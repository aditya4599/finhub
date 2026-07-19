"""
Unit Tests for Evaluation Metrics

Tests for P@K, MRR, nDCG, and other retrieval metrics.
"""

import numpy as np
import pytest

from finsearch.evaluation.metrics import (
    aggregate_metrics,
    average_precision,
    compute_retrieval_metrics,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    """Tests for Precision@K"""

    def test_perfect_ranking(self):
        """All top results are relevant"""
        relevant = [True, True, True, True, True]
        assert precision_at_k(relevant, k=5) == 1.0

    def test_no_relevant(self):
        """No relevant results in top K"""
        relevant = [False, False, False, False, False]
        assert precision_at_k(relevant, k=5) == 0.0

    def test_partial_relevant(self):
        """Some relevant results"""
        relevant = [True, False, True, False, True]
        assert precision_at_k(relevant, k=5) == 0.6

    def test_k_smaller_than_results(self):
        """K is smaller than result list"""
        relevant = [True, True, False, False, False]
        assert precision_at_k(relevant, k=2) == 1.0
        assert precision_at_k(relevant, k=3) == pytest.approx(2 / 3)

    def test_empty_list(self):
        """Empty result list"""
        assert precision_at_k([], k=5) == 0.0

    def test_numpy_array(self):
        """Works with numpy arrays"""
        relevant = np.array([True, False, True, False, True])
        assert precision_at_k(relevant, k=5) == 0.6


class TestMRR:
    """Tests for Mean Reciprocal Rank"""

    def test_first_position(self):
        """Relevant item at position 1"""
        assert mrr([True, False, False]) == 1.0

    def test_second_position(self):
        """Relevant item at position 2"""
        assert mrr([False, True, False]) == 0.5

    def test_third_position(self):
        """Relevant item at position 3"""
        assert mrr([False, False, True]) == pytest.approx(1 / 3)

    def test_multiple_relevant(self):
        """Multiple relevant items - only first matters"""
        assert mrr([False, True, True, True]) == 0.5

    def test_no_relevant(self):
        """No relevant items"""
        assert mrr([False, False, False]) == 0.0

    def test_empty_list(self):
        """Empty result list"""
        assert mrr([]) == 0.0


class TestNDCG:
    """Tests for nDCG@K"""

    def test_perfect_ranking(self):
        """Perfectly ranked results should have nDCG = 1.0"""
        # Best items first
        relevance = [3, 2, 1, 0]
        assert ndcg_at_k(relevance, k=4) == 1.0

    def test_worst_ranking(self):
        """Reversed ranking should have lower nDCG"""
        # Worst items first
        relevance = [0, 1, 2, 3]
        score = ndcg_at_k(relevance, k=4)
        assert 0 < score < 1.0

    def test_binary_relevance(self):
        """Binary relevance (0/1)"""
        relevance = [1, 0, 1, 0, 1]
        score = ndcg_at_k(relevance, k=5)
        assert 0 < score <= 1.0

    def test_all_zeros(self):
        """No relevant items"""
        assert ndcg_at_k([0, 0, 0, 0], k=4) == 0.0

    def test_k_truncation(self):
        """K truncates the list"""
        relevance = [1, 1, 0, 0, 4, 4, 4]  # High scores at end
        score_k2 = ndcg_at_k(relevance, k=2)
        score_k7 = ndcg_at_k(relevance, k=7)
        assert score_k2 != score_k7


class TestDCG:
    """Tests for DCG@K"""

    def test_single_item(self):
        """Single item DCG"""
        # DCG = (2^rel - 1) / log2(2) = rel
        assert dcg_at_k([1], k=1) == pytest.approx(1.0)
        assert dcg_at_k([3], k=1) == pytest.approx(7.0)  # 2^3 - 1

    def test_empty(self):
        """Empty list"""
        assert dcg_at_k([], k=5) == 0.0


class TestAveragePrecision:
    """Tests for Average Precision"""

    def test_perfect_ranking(self):
        """All relevant at top"""
        relevant = [True, True, True, False, False]
        assert average_precision(relevant) == 1.0

    def test_no_relevant(self):
        """No relevant items"""
        assert average_precision([False, False, False]) == 0.0

    def test_alternating(self):
        """Alternating relevant/not relevant"""
        relevant = [True, False, True, False, True]
        ap = average_precision(relevant)
        # P@1=1, P@3=2/3, P@5=3/5 -> AP = (1 + 2/3 + 3/5) / 3
        expected = (1 + 2 / 3 + 3 / 5) / 3
        assert ap == pytest.approx(expected)


class TestComputeRetrievalMetrics:
    """Tests for compute_retrieval_metrics"""

    def test_returns_all_metrics(self):
        """Should return all expected metrics"""
        relevant = [True, False, True, False, True]
        metrics = compute_retrieval_metrics(relevant, k_values=[5, 10])

        assert "mrr" in metrics
        assert "ap" in metrics
        assert "p@5" in metrics
        assert "p@10" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics

    def test_with_total_relevant(self):
        """Should include recall when total_relevant provided"""
        relevant = [True, False, True, False, True]
        metrics = compute_retrieval_metrics(relevant, k_values=[5], total_relevant=5)

        assert "r@5" in metrics


class TestAggregateMetrics:
    """Tests for aggregate_metrics"""

    def test_aggregation(self):
        """Should compute mean and std"""
        metrics_list = [
            {"p@5": 0.6, "mrr": 0.7},
            {"p@5": 0.8, "mrr": 0.8},
            {"p@5": 0.7, "mrr": 0.9},
        ]

        aggregated = aggregate_metrics(metrics_list)

        assert aggregated["p@5"] == pytest.approx(0.7)
        assert aggregated["mrr"] == pytest.approx(0.8)
        assert "p@5_std" in aggregated
        assert "mrr_std" in aggregated

    def test_empty_list(self):
        """Empty list returns empty dict"""
        assert aggregate_metrics([]) == {}


class TestRecallAtK:
    """Tests for Recall@K"""

    def test_partial_recall(self):
        assert recall_at_k([True, False, True, False], total_relevant=4, k=2) == pytest.approx(0.25)

    def test_zero_total_relevant(self):
        assert recall_at_k([True, True], total_relevant=0, k=2) == 0.0
