"""Tests for the FAISS index wrapper and the search engine (offline)."""

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("faiss")

from finsearch.indexing.faiss_index import FaissVectorIndex  # noqa: E402
from finsearch.retrieval.live_news import search_news  # noqa: E402
from finsearch.retrieval.search import FinancialSearchEngine  # noqa: E402


class TestFaissVectorIndex:
    def test_flat_cosine_returns_exact_neighbour(self):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((200, 16)).astype(np.float32)
        index = FaissVectorIndex(16, index_type="flat", metric="cosine")
        index.add(vectors)
        scores, indices = index.search(vectors[7], k=3)
        assert indices[0][0] == 7
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_ivf_requires_training(self):
        index = FaissVectorIndex(8, index_type="ivf", nlist=4)
        with pytest.raises(ValueError, match="trained"):
            index.add(np.ones((10, 8), dtype=np.float32))

    def test_save_load_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((500, 16)).astype(np.float32)
        index = FaissVectorIndex(16, index_type="ivf", nlist=8, nprobe=8)
        index.train_and_add(vectors)
        query = rng.standard_normal((1, 16)).astype(np.float32)
        _, before = index.search(query, k=5)

        index.save(str(tmp_path / "index.faiss"))
        reloaded = FaissVectorIndex(16).load(str(tmp_path / "index.faiss"))
        _, after = reloaded.search(query, k=5)
        assert np.array_equal(before, after)
        assert reloaded.n_vectors == 500 and reloaded.index_type == "ivf"

    def test_dimension_mismatch_message(self):
        index = FaissVectorIndex(8)
        with pytest.raises(ValueError, match="does not match"):
            index.add(np.ones((2, 4), dtype=np.float32))

    def test_invalid_index_type(self):
        with pytest.raises(ValueError, match="index_type"):
            FaissVectorIndex(8, index_type="hnsw")


HEADLINES = [
    "apple earnings rise",
    "tesla stock fall",
    "fed rates rise",
    "oil price fall",
    "apple stock rise",
]


class TestFinancialSearchEngine:
    def _engine(self, encoder):
        engine = FinancialSearchEngine(encoder)
        engine.build_index(
            HEADLINES,
            dates=[datetime(2024, 1, d) for d in range(1, 6)],
            tickers=["AAPL", "TSLA", None, None, "AAPL"],
            metadata=[{"source": "wire"}] * 5,
            index_type="flat",
            show_progress=False,
        )
        return engine

    def test_exact_match_ranks_first(self, hashing_encoder):
        results = self._engine(hashing_encoder).search("apple earnings rise", k=3)
        assert results[0].headline == "apple earnings rise"
        assert [r.rank for r in results] == [1, 2, 3]
        assert results[0].source == "wire"

    def test_filters(self, hashing_encoder):
        engine = self._engine(hashing_encoder)
        by_ticker = engine.search("apple", k=5, ticker="aapl")
        # documents without a ticker are not excluded by the ticker filter
        assert {r.ticker for r in by_ticker} <= {"AAPL", None}
        by_date = engine.search("rise", k=5, start_date="2024-01-03")
        assert all(r.date >= datetime(2024, 1, 3) for r in by_date)

    def test_save_and_load(self, hashing_encoder, tmp_path):
        engine = self._engine(hashing_encoder)
        engine.save(str(tmp_path))
        reloaded = FinancialSearchEngine(hashing_encoder).load(str(tmp_path))
        assert [r.headline for r in reloaded.search("oil price", k=2)] == [
            r.headline for r in engine.search("oil price", k=2)
        ]

    def test_metadata_length_mismatch(self, hashing_encoder):
        with pytest.raises(ValueError, match="tickers"):
            FinancialSearchEngine(hashing_encoder).build_index(
                HEADLINES, tickers=["AAPL"], show_progress=False
            )

    def test_empty_index_error(self, hashing_encoder):
        with pytest.raises(ValueError, match="empty"):
            FinancialSearchEngine(hashing_encoder).search("anything")


class TestLiveNewsSearch:
    def test_skips_empty_headlines_without_misalignment(self, hashing_encoder):
        """Regression: the original mapped result indices back to the unfiltered list."""
        articles = [
            {"headline": "", "source": "A"},
            {"headline": "oil price fall", "source": "B"},
            {"headline": "apple earnings rise", "source": "C"},
        ]
        results = search_news("apple earnings rise", articles, hashing_encoder, k=1)
        assert results[0]["headline"] == "apple earnings rise"
        assert results[0]["source"] == "C"
