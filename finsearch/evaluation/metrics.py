"""Information-retrieval metrics computed from a ranked list of relevance labels.

Every function takes the relevance of the ranked results for ONE query,
position 0 being the top result.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

Relevance = Union[Sequence[bool], Sequence[float], np.ndarray]

DEFAULT_K_VALUES = (5, 10, 20)


def precision_at_k(relevant: Relevance, k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    The denominator is ``min(k, len(relevant))``, so a list shorter than ``k`` is
    not penalised for its missing positions. Returns 0.0 for an empty list.

    >>> round(precision_at_k([True, False, True, False, True], k=3), 4)
    0.6667
    """
    if len(relevant) == 0:
        return 0.0
    top_k = np.asarray(relevant)[:k]
    return np.sum(top_k) / min(k, len(top_k))


def recall_at_k(relevant: Relevance, total_relevant: int, k: int) -> float:
    """Relevant results in the top ``k`` divided by ``total_relevant`` (0.0 if that is 0)."""
    if total_relevant == 0:
        return 0.0
    return np.sum(np.asarray(relevant)[:k]) / total_relevant


def average_precision(relevant: Relevance) -> float:
    """Mean of precision@i over every rank i that holds a relevant result.

    Note: this averages over the relevant results *present in the list*, not
    over all relevant documents in the collection.
    """
    relevant = np.asarray(relevant)
    if np.sum(relevant) == 0:
        return 0.0

    precisions = []
    relevant_seen = 0
    for rank_index, is_relevant in enumerate(relevant):
        if is_relevant:
            relevant_seen += 1
            precisions.append(relevant_seen / (rank_index + 1))
    return np.mean(precisions) if precisions else 0.0


def mrr(relevant: Relevance) -> float:
    """Reciprocal rank of the first relevant result for one query (average across
    queries to obtain Mean Reciprocal Rank).

    >>> round(mrr([False, False, True, False]), 4)
    0.3333
    """
    relevant_positions = np.where(np.asarray(relevant))[0]
    if len(relevant_positions) == 0:
        return 0.0
    return 1.0 / (relevant_positions[0] + 1)


def dcg_at_k(relevance_scores: Relevance, k: int) -> float:
    """Discounted cumulative gain: ``sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)``."""
    relevance_scores = np.asarray(relevance_scores)[:k]
    if len(relevance_scores) == 0:
        return 0.0
    gains = np.power(2, relevance_scores) - 1
    discounts = np.log2(np.arange(1, len(relevance_scores) + 1) + 1)
    return np.sum(gains / discounts)


def ndcg_at_k(relevance_scores: Relevance, k: int) -> float:
    """``DCG@k / IDCG@k``, where the ideal ordering is the same list sorted descending.

    The ideal DCG is computed from the retrieved list itself, not from every
    relevant document in the collection.
    """
    relevance_scores = np.asarray(relevance_scores)
    actual_dcg = dcg_at_k(relevance_scores, k)
    if actual_dcg == 0:
        return 0.0
    ideal_dcg = dcg_at_k(np.sort(relevance_scores)[::-1], k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_retrieval_metrics(
    relevant: Relevance,
    k_values: Optional[Sequence[int]] = None,
    total_relevant: Optional[int] = None,
) -> Dict[str, float]:
    """Compute ``mrr``, ``ap`` and ``p@k`` / ``ndcg@k`` (and ``r@k`` when
    ``total_relevant`` is given) for one query."""
    k_values = DEFAULT_K_VALUES if k_values is None else k_values
    relevance_scores = np.asarray(relevant).astype(float)
    relevant_mask = relevance_scores.astype(bool)

    metrics = {
        "mrr": mrr(relevant_mask),
        "ap": average_precision(relevant_mask),
    }
    for k in k_values:
        metrics[f"p@{k}"] = precision_at_k(relevant_mask, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)
        if total_relevant is not None:
            metrics[f"r@{k}"] = recall_at_k(relevant_mask, total_relevant, k)
    return metrics


def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across queries, plus ``<metric>_std`` (population std)."""
    if not per_query_metrics:
        return {}
    aggregated = {}
    for name in per_query_metrics[0]:
        values = [query_metrics[name] for query_metrics in per_query_metrics]
        aggregated[name] = np.mean(values)
        aggregated[f"{name}_std"] = np.std(values)
    return aggregated
