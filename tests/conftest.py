"""Shared fixtures: tiny offline models so training/search paths run without downloads."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from finsearch.models.encoders import BaseEncoder

TINY_VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
    "apple",
    "tesla",
    "fed",
    "rates",
    "earnings",
    "stock",
    "rise",
    "fall",
    "oil",
    "bitcoin",
    "bank",
    "merger",
    "dividend",
    "ipo",
    "bond",
    "market",
    "price",
    "gold",
    "deal",
    "news",
]


class HashingEncoder(BaseEncoder):
    """Deterministic bag-of-words encoder used to exercise search code offline."""

    def __init__(self, dimension: int = 32):
        super().__init__("hashing-encoder", "cpu")
        self._embedding_dim = dimension
        self._model = object()

    def load(self):
        return self

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._embedding_dim, dtype=np.float32)
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._embedding_dim
            vector[bucket] += 1.0
        return vector

    def encode(self, texts, batch_size=32, show_progress=True, normalize=True):
        texts = [texts] if isinstance(texts, str) else texts
        vectors = np.stack([self._vector(text) for text in texts])
        if normalize:
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        return vectors

    def encode_queries(self, queries, batch_size=32):
        return self.encode(queries, show_progress=False)

    def save(self, path):
        pass

    def load_from_path(self, path):
        return self


@pytest.fixture
def hashing_encoder() -> HashingEncoder:
    return HashingEncoder()


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory):
    """A randomly initialised 2-layer BERT + WordPiece tokenizer saved to disk."""
    transformers = pytest.importorskip("transformers")
    import torch

    directory = tmp_path_factory.mktemp("tiny_bert")
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(TINY_VOCAB) + "\n")

    torch.manual_seed(0)
    model_config = transformers.BertConfig(
        vocab_size=len(TINY_VOCAB),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    transformers.BertModel(model_config).save_pretrained(directory)
    transformers.BertTokenizer(vocab_file=str(vocab_file)).save_pretrained(directory)
    return directory
