"""Semantic search over an indexed collection of financial headlines."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from finsearch.config import get_config
from finsearch.indexing.faiss_index import FaissVectorIndex
from finsearch.models.encoders import BaseEncoder

DateLike = Union[str, date, datetime]

INDEX_FILENAME = "index.faiss"
DOCUMENTS_FILENAME = "documents.pkl"

# When filters are active and no explicit candidate pool is given, retrieve
# this many times ``k`` candidates before filtering.
FILTER_CANDIDATE_MULTIPLIER = 10


@dataclass
class SearchResult:
    """One ranked search hit."""

    headline: str
    score: float
    rank: int
    date: Optional[datetime] = None
    ticker: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def parse_date(value: DateLike) -> datetime:
    """Convert a ``str``/``date``/``datetime`` into a ``datetime``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    from dateutil.parser import parse

    return parse(value)


class FinancialSearchEngine:
    """Encode headlines, index them with FAISS and answer filtered semantic queries.

    Args:
        encoder: Any :class:`BaseEncoder` (base or fine-tuned SBERT/FinBERT).
        index: Optional pre-built :class:`FaissVectorIndex`.
    """

    def __init__(self, encoder: BaseEncoder, index: Optional[FaissVectorIndex] = None):
        self.encoder = encoder
        self.index = index
        self.documents: List[Dict[str, Any]] = []
        self.document_dates: List[Optional[datetime]] = []
        self.document_tickers: List[Optional[str]] = []

    def build_index(
        self,
        headlines: List[str],
        dates: Optional[List[datetime]] = None,
        tickers: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64,
        index_type: str = "ivf",
        show_progress: bool = True,
    ) -> None:
        """Encode ``headlines`` and build a cosine FAISS index over them.

        ``dates``, ``tickers`` and ``metadata`` are optional per-headline lists
        stored alongside the index and used for filtering / display.
        """
        for name, values in (("dates", dates), ("tickers", tickers), ("metadata", metadata)):
            if values is not None and len(values) != len(headlines):
                raise ValueError(
                    f"'{name}' has {len(values)} entries but there are {len(headlines)} headlines."
                )

        if show_progress:
            print(f"Encoding {len(headlines)} headlines...")
        document_embeddings = self.encoder.encode(
            headlines, batch_size=batch_size, show_progress=show_progress, normalize=True
        )

        search_config = get_config().search
        self.index = FaissVectorIndex(
            dimension=document_embeddings.shape[1],
            index_type=index_type,
            nlist=search_config.nlist,
            nprobe=search_config.nprobe,
            metric="cosine",
        )
        if show_progress:
            print("Building FAISS index...")
        self.index.train_and_add(document_embeddings)

        self.documents = [
            {
                "headline": headline,
                "date": dates[i] if dates else None,
                "ticker": tickers[i] if tickers else None,
                "metadata": metadata[i] if metadata else None,
            }
            for i, headline in enumerate(headlines)
        ]
        self.document_dates = dates or [None] * len(headlines)
        self.document_tickers = tickers or [None] * len(headlines)

        if show_progress:
            print(f"Index built with {self.index.n_vectors} vectors")

    def _candidate_count(self, k: int, has_filters: bool, rerank_k: Optional[int]) -> int:
        if not has_filters:
            candidate_count = k
        elif rerank_k:
            candidate_count = rerank_k
        else:
            candidate_count = k * FILTER_CANDIDATE_MULTIPLIER
        return min(candidate_count, self.index.n_vectors)

    def search(
        self,
        query_text: str,
        k: int = 10,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        ticker: Optional[str] = None,
        rerank_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Return up to ``k`` headlines ranked by cosine similarity to ``query_text``.

        Filters keep documents on/after ``start_date``, on/before ``end_date`` and
        matching ``ticker`` (case-insensitive). Documents with no date/ticker are
        not excluded by that filter. With filters active, ``rerank_k`` candidates
        (default ``10 * k``) are retrieved first and then filtered, so fewer than
        ``k`` results can come back.
        """
        if self.index is None or self.index.is_empty:
            raise ValueError("Index is empty: call build_index() or load() first.")

        has_filters = bool(start_date or end_date or ticker)
        candidate_count = self._candidate_count(k, has_filters, rerank_k)

        query_embeddings = self.encoder.encode_queries(query_text)
        scores, indices = self.index.search(query_embeddings, candidate_count)

        start = parse_date(start_date) if start_date else None
        end = parse_date(end_date) if end_date else None

        retrieval_results: List[SearchResult] = []
        for score, document_index in zip(scores[0], indices[0]):
            if document_index < 0:  # FAISS pads missing neighbours with -1
                continue
            document = self.documents[document_index]

            if start and document["date"] and document["date"] < start:
                continue
            if end and document["date"] and document["date"] > end:
                continue
            if ticker and document["ticker"] and document["ticker"].upper() != ticker.upper():
                continue

            document_metadata = document["metadata"]
            retrieval_results.append(
                SearchResult(
                    headline=document["headline"],
                    score=float(score),
                    rank=len(retrieval_results) + 1,
                    date=document["date"],
                    ticker=document["ticker"],
                    source=(document_metadata or {}).get("source"),
                    metadata=document_metadata,
                )
            )
            if len(retrieval_results) >= k:
                break

        return retrieval_results

    def save(self, directory: str) -> None:
        """Write ``index.faiss`` (+ ``index.meta``) and ``documents.pkl`` to ``directory``."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.index:
            self.index.save(str(save_dir / INDEX_FILENAME))
        with open(save_dir / DOCUMENTS_FILENAME, "wb") as handle:
            pickle.dump(
                {
                    "documents": self.documents,
                    "dates": self.document_dates,
                    "tickers": self.document_tickers,
                },
                handle,
            )

    def load(self, directory: str) -> "FinancialSearchEngine":
        """Load an engine saved with :meth:`save`. ``documents.pkl`` is a pickle:
        only load directories you created or trust."""
        load_dir = Path(directory)
        if not load_dir.is_dir():
            raise FileNotFoundError(f"Index directory not found: {load_dir}")

        index_path = load_dir / INDEX_FILENAME
        if index_path.exists():
            self.index = FaissVectorIndex(self.encoder.embedding_dim).load(str(index_path))

        documents_path = load_dir / DOCUMENTS_FILENAME
        if documents_path.exists():
            with open(documents_path, "rb") as handle:
                stored = pickle.load(handle)
            self.documents = stored["documents"]
            self.document_dates = stored["dates"]
            self.document_tickers = stored["tickers"]
        return self


SAMPLE_HEADLINES = [
    "Apple reports record quarterly revenue of $123.9 billion",
    "Tesla stock plunges 12% after Elon Musk sells shares",
    "Federal Reserve raises interest rates by 0.75%",
    "Bitcoin crashes below $20,000 amid market panic",
    "Amazon announces 10,000 layoffs in cost-cutting move",
    "Microsoft acquires Activision Blizzard for $68.7 billion",
    "Oil prices surge above $100 per barrel on supply concerns",
    "Meta shares jump 20% on strong ad revenue growth",
    "Goldman Sachs reports Q3 earnings beat expectations",
    "Nvidia stock soars on AI chip demand surge",
]
