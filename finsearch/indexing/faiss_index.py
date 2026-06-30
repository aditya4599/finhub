"""FAISS vector index wrapper.

With ``metric="cosine"`` vectors are L2-normalised and searched by inner
product, so returned scores are cosine similarities in ``[-1, 1]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only when faiss is missing
    faiss = None

INDEX_TYPES = ("flat", "ivf")
METRICS = ("cosine", "l2")


class FaissVectorIndex:
    """Exact (``flat``) or approximate (``ivf``) nearest-neighbour index.

    Args:
        dimension: Embedding dimension.
        index_type: ``"flat"`` (exact) or ``"ivf"`` (inverted file; needs training).
        nlist: Number of IVF clusters. FAISS needs at least ``nlist`` training vectors.
        nprobe: IVF clusters visited per query (higher = more accurate, slower).
        metric: ``"cosine"`` or ``"l2"``.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nlist: int = 1000,
        nprobe: int = 50,
        metric: str = "cosine",
    ):
        if faiss is None:
            raise ImportError("FAISS is not installed. Install it with: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type.lower()
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric.lower()

        if self.index_type not in INDEX_TYPES:
            raise ValueError(f"Unknown index_type '{index_type}'. Choose one of {INDEX_TYPES}.")
        if self.metric not in METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose one of {METRICS}.")

        self._index = None
        self._is_trained = False
        self._n_vectors = 0

    def _create_index(self) -> None:
        if self.metric == "cosine":
            flat_index, faiss_metric = faiss.IndexFlatIP, faiss.METRIC_INNER_PRODUCT
        else:
            flat_index, faiss_metric = faiss.IndexFlatL2, faiss.METRIC_L2

        if self.index_type == "flat":
            self._index = flat_index(self.dimension)
        else:
            quantizer = flat_index(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss_metric)
            self._index.nprobe = self.nprobe

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}."
            )
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-8)
        return vectors

    def train(self, vectors: np.ndarray) -> None:
        """Train the IVF quantiser (no-op for flat indexes)."""
        if self._index is None:
            self._create_index()
        vectors = self._prepare(vectors)
        if self.index_type == "ivf":
            self._index.train(vectors)
        self._is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors; IVF indexes must be trained first."""
        if self._index is None:
            self._create_index()
        if self.index_type == "ivf" and not self._is_trained:
            raise ValueError("IVF index must be trained before adding vectors (call train()).")
        vectors = self._prepare(vectors)
        self._index.add(vectors)
        self._n_vectors += len(vectors)

    def train_and_add(self, vectors: np.ndarray) -> None:
        """Train on ``vectors`` and add them in one step."""
        self.train(vectors)
        self.add(vectors)

    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)``, each of shape ``(n_queries, k)``.

        Scores are cosine similarities for ``metric="cosine"`` and squared L2
        distances for ``metric="l2"``. Missing neighbours have index ``-1``.
        """
        if self._index is None or self._n_vectors == 0:
            raise ValueError("Index is empty: add vectors before searching.")
        return self._index.search(self._prepare(query_vectors), k)

    def save(self, path: str) -> None:
        """Write the FAISS index to ``path`` and its settings to ``path`` with suffix ``.meta``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        with open(path.with_suffix(".meta"), "wb") as handle:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    "metric": self.metric,
                    "n_vectors": self._n_vectors,
                },
                handle,
            )

    def load(self, path: str) -> "FaissVectorIndex":
        """Load an index written by :meth:`save` (only load files you trust: ``.meta`` is a pickle)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")
        self._index = faiss.read_index(str(path))

        meta_path = path.with_suffix(".meta")
        if meta_path.exists():
            with open(meta_path, "rb") as handle:
                meta = pickle.load(handle)
            self.dimension = meta["dimension"]
            self.index_type = meta["index_type"]
            self.nlist = meta["nlist"]
            self.nprobe = meta["nprobe"]
            self.metric = meta["metric"]
            self._n_vectors = meta["n_vectors"]
        else:
            self._n_vectors = self._index.ntotal

        if self.index_type == "ivf":
            self._index.nprobe = self.nprobe
        self._is_trained = True
        return self

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    @property
    def is_empty(self) -> bool:
        return self._n_vectors == 0
