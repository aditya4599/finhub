"""Statistics over multi-seed evaluation runs: t-based confidence intervals,
paired t-tests and Cohen's d."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# model name -> seed -> metric name -> value
MultiSeedResults = Dict[str, Dict[int, Dict[str, float]]]


@dataclass
class StatisticalResult:
    """Mean, sample standard deviation and confidence interval of one metric."""

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int


@dataclass
class ComparisonResult:
    """Paired comparison of two models on one metric."""

    model_a: str
    model_b: str
    metric: str
    mean_diff: float
    t_statistic: float
    p_value: float
    is_significant: bool
    effect_size: float  # Cohen's d (pooled standard deviation)


class StatisticalAnalyzer:
    """Confidence intervals and significance tests for per-seed metric values.

    Args:
        significance_level: Alpha used to flag ``is_significant``.
        confidence_level: Coverage of the t-based confidence intervals.
    """

    def __init__(self, significance_level: float = 0.05, confidence_level: float = 0.95):
        self.alpha = significance_level
        self.confidence_level = confidence_level

    def compute_ci(self, values: Sequence[float], confidence: Optional[float] = None) -> StatisticalResult:
        """Mean ± t_{(1+c)/2, n-1} · s/√n (requires at least two values for a finite interval)."""
        confidence = confidence or self.confidence_level
        values = np.asarray(values)
        n = len(values)

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        half_width = t_critical * std / np.sqrt(n)

        return StatisticalResult(
            mean=mean, std=std, ci_lower=mean - half_width, ci_upper=mean + half_width, n_samples=n
        )

    def paired_t_test(
        self, values_a: Sequence[float], values_b: Sequence[float], alternative: str = "two-sided"
    ) -> Tuple[float, float]:
        """Paired t-test; returns ``(t_statistic, p_value)``."""
        t_statistic, p_value = stats.ttest_rel(
            np.asarray(values_a), np.asarray(values_b), alternative=alternative
        )
        return t_statistic, p_value

    def cohens_d(self, values_a: Sequence[float], values_b: Sequence[float]) -> float:
        """Standardised mean difference ``(mean_a - mean_b) / pooled_std`` (0.0 if pooled std is 0).

        Rough guide: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large.
        """
        values_a = np.asarray(values_a)
        values_b = np.asarray(values_b)
        n_a, n_b = len(values_a), len(values_b)
        pooled_std = np.sqrt(
            ((n_a - 1) * np.var(values_a, ddof=1) + (n_b - 1) * np.var(values_b, ddof=1)) / (n_a + n_b - 2)
        )
        if pooled_std == 0:
            return 0.0
        return (np.mean(values_a) - np.mean(values_b)) / pooled_std

    def compare_models(
        self,
        results_a: Sequence[float],
        results_b: Sequence[float],
        model_a_name: str,
        model_b_name: str,
        metric_name: str,
    ) -> ComparisonResult:
        """Compare two models on one metric using values paired by seed."""
        values_a = np.asarray(results_a)
        values_b = np.asarray(results_b)
        t_statistic, p_value = self.paired_t_test(values_a, values_b)
        return ComparisonResult(
            model_a=model_a_name,
            model_b=model_b_name,
            metric=metric_name,
            mean_diff=np.mean(values_a) - np.mean(values_b),
            t_statistic=t_statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            effect_size=self.cohens_d(values_a, values_b),
        )

    def analyze_multi_seed_results(
        self, results: MultiSeedResults, metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, StatisticalResult]]:
        """Return ``model -> metric -> StatisticalResult`` across seeds."""
        analysis: Dict[str, Dict[str, StatisticalResult]] = {}
        for model_name, seed_results in results.items():
            available_metrics = set()
            for seed_metrics in seed_results.values():
                available_metrics.update(seed_metrics.keys())
            if metrics is not None:
                available_metrics &= set(metrics)

            analysis[model_name] = {}
            for metric in available_metrics:
                values = [
                    seed_metrics[metric] for seed_metrics in seed_results.values() if metric in seed_metrics
                ]
                if values:
                    analysis[model_name][metric] = self.compute_ci(values)
        return analysis

    def pairwise_comparisons(
        self,
        results: MultiSeedResults,
        metrics: Optional[List[str]] = None,
        baseline_model: Optional[str] = None,
    ) -> List[ComparisonResult]:
        """Paired comparisons for every model pair (or only pairs involving
        ``baseline_model``) on every shared metric, using seeds common to all models."""
        model_names = list(results.keys())
        common_seeds = sorted(set.intersection(*[set(seed_results) for seed_results in results.values()]))
        if not common_seeds:
            return []

        available_metrics = set()
        for seed_results in results.values():
            for seed_metrics in seed_results.values():
                available_metrics.update(seed_metrics.keys())
        if metrics is not None:
            available_metrics &= set(metrics)

        comparisons = []
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1 :]:
                if baseline_model and baseline_model not in (model_a, model_b):
                    continue
                for metric in sorted(available_metrics):
                    values_a = [
                        results[model_a][seed][metric]
                        for seed in common_seeds
                        if metric in results[model_a].get(seed, {})
                    ]
                    values_b = [
                        results[model_b][seed][metric]
                        for seed in common_seeds
                        if metric in results[model_b].get(seed, {})
                    ]
                    if len(values_a) >= 2 and len(values_b) >= 2:
                        comparisons.append(self.compare_models(values_a, values_b, model_a, model_b, metric))
        return comparisons


def format_statistics_table(
    analysis: Dict[str, Dict[str, StatisticalResult]], metrics: Optional[List[str]] = None
) -> str:
    """Markdown table of ``mean±CI half-width`` per model and metric."""
    if not analysis:
        return ""
    all_metrics = set()
    for model_stats in analysis.values():
        all_metrics.update(model_stats.keys())
    if metrics:
        all_metrics &= set(metrics)
    all_metrics = sorted(all_metrics)

    header = "| Model | " + " | ".join(all_metrics) + " |"
    separator = "|" + "|".join(["-" * 20] * (len(all_metrics) + 1)) + "|"
    rows = []
    for model_name, model_stats in analysis.items():
        cells = []
        for metric in all_metrics:
            if metric in model_stats:
                result = model_stats[metric]
                cells.append(f"{result.mean:.4f}±{(result.ci_upper - result.ci_lower) / 2:.4f}")
            else:
                cells.append("-")
        rows.append(f"| {model_name} | " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


def format_comparison_table(comparisons: List[ComparisonResult]) -> str:
    """Markdown table of pairwise comparisons."""
    if not comparisons:
        return ""
    header = "| Model A | Model B | Metric | Δ Mean | p-value | Effect Size | Sig.? |"
    separator = "|---------|---------|--------|--------|---------|-------------|-------|"
    rows = [
        f"| {c.model_a} | {c.model_b} | {c.metric} | {c.mean_diff:+.4f} | {c.p_value:.4f} | "
        f"{c.effect_size:.2f} | {'✓' if c.is_significant else '-'} |"
        for c in comparisons
    ]
    return "\n".join([header, separator] + rows)
