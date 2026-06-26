"""Text encoders that turn financial text into dense embeddings.

* :class:`SBERTEncoder` wraps ``sentence-transformers/all-MiniLM-L6-v2`` (384-d).
* :class:`FinBERTEncoder` wraps ``yiyanghkust/finbert-tone`` with mean pooling (768-d).
* ``FineTuned*`` variants load a checkpoint written by the training scripts.
* :class:`FinBERTTripletModel` is the trainable FinBERT module used during
  triplet-loss fine-tuning (SBERT is trained as a plain ``SentenceTransformer``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from finsearch.config import get_config

Texts = Union[str, List[str]]

FINBERT_ENCODER_CONFIG = "encoder_config.json"


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings over non-padding positions.

    Args:
        token_embeddings: ``(batch, seq_len, hidden_dim)``
        attention_mask: ``(batch, seq_len)``

    Returns:
        ``(batch, hidden_dim)`` sentence embeddings.
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / token_counts


class BaseEncoder(ABC):
    """Common interface for every encoder used for indexing, search and evaluation."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        """Dimension of the output embeddings."""
        if self._embedding_dim is None:
            raise ValueError("Embedding dimension unknown: call load() first.")
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    def load(self) -> "BaseEncoder":
        """Load weights into memory and return ``self``."""

    @abstractmethod
    def encode(
        self,
        texts: Texts,
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) into an ``(n_texts, embedding_dim)`` array."""

    @abstractmethod
    def encode_queries(self, queries: Texts, batch_size: int = 32) -> np.ndarray:
        """Encode search queries (models here encode queries and documents identically)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Write the model to ``path``."""

    @abstractmethod
    def load_from_path(self, path: str) -> "BaseEncoder":
        """Load a previously saved model from ``path`` and return ``self``."""

    def to(self, device: str) -> "BaseEncoder":
        """Move the underlying model to ``device``."""
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def similarity(self, embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix of shape ``(len(embeddings_a), len(embeddings_b))``."""
        normalized_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        normalized_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        return np.dot(normalized_a, normalized_b.T)

    def most_similar(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return ``(document_index, cosine_similarity)`` for the ``top_k`` closest documents."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        similarities = self.similarity(query_embedding, document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(index), float(similarities[index])) for index in top_indices]

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(model='{self.model_name}', device='{self.device}', {status})"


class SBERTEncoder(BaseEncoder):
    """Sentence-BERT encoder backed by the ``sentence-transformers`` library."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        config = get_config()
        super().__init__(model_name or config.model.sbert_model_name, device)
        self.max_seq_length = max_seq_length or config.model.max_seq_length
        self._embedding_dim = config.model.sbert_dim

    def _load_sentence_transformer(self, name_or_path: str) -> "SBERTEncoder":
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(name_or_path, device=self.device)
        self._model.max_seq_length = self.max_seq_length
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self

    def load(self) -> "SBERTEncoder":
        return self._load_sentence_transformer(self.model_name)

    def load_from_path(self, path: str) -> "SBERTEncoder":
        return self._load_sentence_transformer(path)

    def encode(
        self,
        texts: Texts,
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

    def encode_queries(self, queries: Texts, batch_size: int = 32) -> np.ndarray:
        return self.encode(queries, batch_size=batch_size, show_progress=False)

    def save(self, path: str) -> None:
        if not self.is_loaded:
            raise ValueError("Cannot save: model is not loaded.")
        self._model.save(path)

    def get_sentence_transformer(self):
        """Return the underlying ``SentenceTransformer`` (loading it if needed)."""
        if not self.is_loaded:
            self.load()
        return self._model


class FineTunedSBERTEncoder(SBERTEncoder):
    """SBERT encoder loaded from a local fine-tuned checkpoint directory."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__(model_name=checkpoint_path, device=device, max_seq_length=max_seq_length)
        self.checkpoint_path = checkpoint_path

    def load(self) -> "FineTunedSBERTEncoder":
        return self.load_from_path(self.checkpoint_path)


class FinBERTEncoder(BaseEncoder):
    """FinBERT encoder producing sentence embeddings by pooling token states.

    Args:
        pooling_strategy: ``"mean"`` (default), ``"cls"`` or ``"max"``.
    """

    POOLING_STRATEGIES = ("mean", "cls", "max")

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        pooling_strategy: str = "mean",
    ):
        config = get_config()
        super().__init__(model_name or config.model.finbert_model_name, device)
        self.max_seq_length = max_seq_length or config.model.max_seq_length
        self._embedding_dim = config.model.finbert_dim
        self.pooling_strategy = pooling_strategy
        self._tokenizer = None

    def _load_transformer(self, name_or_path: Union[str, Path]) -> "FinBERTEncoder":
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self._model = AutoModel.from_pretrained(name_or_path).to(self.device)
        self._model.eval()
        self._embedding_dim = self._model.config.hidden_size
        return self

    def load(self) -> "FinBERTEncoder":
        return self._load_transformer(self.model_name)

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "mean":
            return mean_pooling(token_embeddings, attention_mask)
        if self.pooling_strategy == "cls":
            return token_embeddings[:, 0, :]
        if self.pooling_strategy == "max":
            padding = attention_mask.unsqueeze(-1).expand(token_embeddings.size()) == 0
            return token_embeddings.masked_fill(padding, -1e9).max(dim=1).values
        raise ValueError(
            f"Unknown pooling strategy '{self.pooling_strategy}'. Choose one of {self.POOLING_STRATEGIES}."
        )

    def encode(
        self,
        texts: Texts,
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        if isinstance(texts, str):
            texts = [texts]

        batch_starts = range(0, len(texts), batch_size)
        if show_progress:
            batch_starts = tqdm(batch_starts, desc="Encoding", unit="batch")

        batch_embeddings = []
        with torch.no_grad():
            for start in batch_starts:
                encoded = self._tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
                outputs = self._model(**encoded)
                embeddings = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                batch_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(batch_embeddings)

    def encode_queries(self, queries: Texts, batch_size: int = 32) -> np.ndarray:
        return self.encode(queries, batch_size=batch_size, show_progress=False)

    def save(self, path: str) -> None:
        if not self.is_loaded:
            raise ValueError("Cannot save: model is not loaded.")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        with open(save_path / FINBERT_ENCODER_CONFIG, "w") as handle:
            json.dump(
                {"pooling_strategy": self.pooling_strategy, "max_seq_length": self.max_seq_length},
                handle,
            )

    def load_from_path(self, path: str) -> "FinBERTEncoder":
        load_path = Path(path)
        encoder_config_path = load_path / FINBERT_ENCODER_CONFIG
        if encoder_config_path.exists():
            with open(encoder_config_path) as handle:
                saved = json.load(handle)
            self.pooling_strategy = saved.get("pooling_strategy", "mean")
            self.max_seq_length = saved.get("max_seq_length", self.max_seq_length)
        return self._load_transformer(load_path)

    def get_model_and_tokenizer(self):
        """Return the underlying ``(model, tokenizer)`` (loading them if needed)."""
        if not self.is_loaded:
            self.load()
        return self._model, self._tokenizer


class FineTunedFinBERTEncoder(FinBERTEncoder):
    """FinBERT encoder loaded from a local fine-tuned checkpoint directory."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        pooling_strategy: str = "mean",
    ):
        super().__init__(
            model_name=checkpoint_path,
            device=device,
            max_seq_length=max_seq_length,
            pooling_strategy=pooling_strategy,
        )
        self.checkpoint_path = checkpoint_path

    def load(self) -> "FineTunedFinBERTEncoder":
        return self.load_from_path(self.checkpoint_path)


class FinBERTTripletModel(nn.Module):
    """Trainable FinBERT: tokenise -> transformer -> mean pooling -> L2 normalise.

    ``forward`` takes a list of strings so :class:`EmbeddingTrainer` can call it
    directly. Checkpoints written by :meth:`save` load with
    :class:`FineTunedFinBERTEncoder` (mean pooling, default max length).
    """

    def __init__(self, model_name: str, device: str = "cpu", max_seq_length: Optional[int] = None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.max_seq_length = max_seq_length or get_config().model.max_seq_length
        self.model.to(device)

    def forward(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        outputs = self.model(**encoded)
        embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Inference-mode embeddings (switches the module to eval mode)."""
        self.eval()
        return self.forward(texts)

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
