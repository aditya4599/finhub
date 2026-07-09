"""Topic-based retrieval evaluation.

Protocol: every query is a labelled text; a retrieved document counts as
relevant when it has the same topic label as the query. When queries are drawn
from the corpus itself (the default), the query's own document is removed from
its ranking and from the relevant-document count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from finsearch.config import get_config
from finsearch.evaluation.metrics import aggregate_metrics, compute_retrieval_metrics
from finsearch.models.encoders import BaseEncoder


@dataclass
class EvaluationResult:
    """Aggregated and per-query metrics for one encoder."""

    model_name: str
    metrics: Dict[str, float]
    per_query_metrics: List[Dict[str, float]]
    num_queries: int
    corpus_size: int


class RetrievalEvaluator:
    """Evaluate how well an encoder retrieves same-topic documents.

    Args:
        corpus_texts: Documents to search over.
        corpus_labels: Topic label of each document.
        query_texts / query_labels: Separate queries; if omitted the corpus is used.
        k_values: Cut-offs for P@k, nDCG@k and R@k (default ``evaluation.k_values``).
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: List[int],
        query_texts: Optional[List[str]] = None,
        query_labels: Optional[List[int]] = None,
        k_values: Optional[List[int]] = None,
    ):
        if len(corpus_texts) != len(corpus_labels):
            raise ValueError(
                f"corpus_texts ({len(corpus_texts)}) and corpus_labels ({len(corpus_labels)}) differ in length."
            )
        self.corpus_texts = corpus_texts
        self.corpus_labels = np.array(corpus_labels)

        if query_texts is not None:
            if query_labels is None or len(query_labels) != len(query_texts):
                raise ValueError("query_labels must be given with one label per query text.")
            self.query_texts = query_texts
            self.query_labels = np.array(query_labels)
        else:
            self.query_texts = corpus_texts
            self.query_labels = self.corpus_labels

        self.k_values = k_values or get_config().evaluation.k_values

    def _get_relevance(
        self, query_index: int, result_indices: np.ndarray, exclude_self: bool = True
    ) -> np.ndarray:
        """Boolean relevance of ``result_indices`` for the query at ``query_index``."""
        relevant = self.corpus_labels[result_indices] == self.query_labels[query_index]
        if exclude_self:
            relevant[result_indices == query_index] = False
        return relevant

    def evaluate(
        self,
        encoder: BaseEncoder,
        num_queries: Optional[int] = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Encode the corpus, sample ``num_queries`` queries without replacement using
        NumPy's global RNG (seed it beforehand for reproducibility) and score them."""
        num_queries = num_queries or get_config().evaluation.num_eval_queries
        num_queries = min(num_queries, len(self.query_texts))

        if show_progress:
            print(f"Encoding corpus ({len(self.corpus_texts)} documents)...")
        document_embeddings = encoder.encode(
            self.corpus_texts, batch_size=batch_size, show_progress=show_progress, normalize=True
        )

        query_indices = np.random.choice(len(self.query_texts), size=num_queries, replace=False)
        selected_query_texts = [self.query_texts[i] for i in query_indices]

        if show_progress:
            print(f"Encoding queries ({len(selected_query_texts)} queries)...")
        query_embeddings = encoder.encode_queries(selected_query_texts, batch_size=batch_size)

        similarities = np.dot(query_embeddings, document_embeddings.T)
        max_k = max(self.k_values)

        positions = range(len(query_indices))
        if show_progress:
            positions = tqdm(positions, desc="Evaluating queries")

        per_query_metrics = []
        for position in positions:
            query_index = query_indices[position]
            # Retrieve one extra result in case the query itself is ranked.
            ranked_indices = np.argsort(similarities[position])[::-1][: max_k + 1]

            relevant = self._get_relevance(query_index, ranked_indices, exclude_self=True)
            relevant = relevant[ranked_indices != query_index][:max_k]

            total_relevant = np.sum(self.corpus_labels == self.query_labels[query_index]) - 1
            per_query_metrics.append(
                compute_retrieval_metrics(
                    relevant=relevant, k_values=self.k_values, total_relevant=total_relevant
                )
            )

        return EvaluationResult(
            model_name=encoder.model_name,
            metrics=aggregate_metrics(per_query_metrics),
            per_query_metrics=per_query_metrics,
            num_queries=num_queries,
            corpus_size=len(self.corpus_texts),
        )

    def evaluate_multiple(
        self,
        encoders: Dict[str, BaseEncoder],
        num_queries: Optional[int] = None,
        batch_size: int = 64,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate several encoders in sequence and return results keyed by name."""
        evaluation_results = {}
        for name, encoder in encoders.items():
            print(f"\n{'=' * 60}\nEvaluating: {name}\n{'=' * 60}\n")
            result = self.evaluate(encoder=encoder, num_queries=num_queries, batch_size=batch_size)
            result.model_name = name
            evaluation_results[name] = result
        return evaluation_results


def format_results_table(results: Dict[str, EvaluationResult], metrics: Optional[List[str]] = None) -> str:
    """Render mean metrics as a Markdown table (``*_std`` columns hidden by default)."""
    if not results:
        return ""
    if metrics is None:
        first_result = next(iter(results.values()))
        metrics = [name for name in first_result.metrics if not name.endswith("_std")]

    header = "| Model | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["-" * 15] * (len(metrics) + 1)) + "|"
    rows = [
        f"| {name} | " + " | ".join(f"{result.metrics.get(m, 0):.4f}" for m in metrics) + " |"
        for name, result in results.items()
    ]
    return "\n".join([header, separator] + rows)
