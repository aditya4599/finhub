#!/usr/bin/env python3
"""Evaluate base and fine-tuned SBERT / FinBERT on topic-based retrieval.

For every seed, NumPy's global RNG is seeded once and each model in turn
samples ``--num_queries`` queries from the held-out test split (so models are
evaluated on different query samples within a seed). Fine-tuned models are
loaded from a single checkpoint each; seeds vary the query sample, not training.

Usage:
    python scripts/evaluate_all.py --seeds 42 123 456 789 1024 --num_queries 200 --no_cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from finsearch.config import get_config, load_config, resolve_device
from finsearch.data.datasets import load_twitter_financial_news
from finsearch.evaluation.analysis import (
    StatisticalAnalyzer,
    format_comparison_table,
    format_statistics_table,
)
from finsearch.evaluation.evaluator import RetrievalEvaluator
from finsearch.models.encoders import (
    BaseEncoder,
    FinBERTEncoder,
    FineTunedFinBERTEncoder,
    FineTunedSBERTEncoder,
    SBERTEncoder,
)

SUMMARY_METRICS = ["p@5", "p@10", "mrr", "ndcg@10"]
COMPARISON_METRICS = ["p@5", "mrr", "ndcg@10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all models")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Query-sampling seeds")
    parser.add_argument("--num_queries", type=int, default=500, help="Queries per model per seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--sbert_path", type=str, default=None, help="Fine-tuned SBERT checkpoint")
    parser.add_argument("--finbert_path", type=str, default=None, help="Fine-tuned FinBERT checkpoint")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Results JSON (default: <results_dir>/evaluation_results.json)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    return parser.parse_args()


def _resolve_checkpoint(explicit_path: Optional[str], model_family: str) -> Optional[Path]:
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)
    default_path = get_config().paths.default_checkpoint(model_family)
    return default_path if default_path.exists() else None


def load_models(args: argparse.Namespace, device: str) -> Dict[str, BaseEncoder]:
    """Load the base encoders and whichever fine-tuned checkpoints exist."""
    print("\nLoading models...")
    models: Dict[str, BaseEncoder] = {
        "Base SBERT": SBERTEncoder(device=device).load(),
        "Base FinBERT": FinBERTEncoder(device=device).load(),
    }

    sbert_checkpoint = _resolve_checkpoint(args.sbert_path, "sbert")
    if sbert_checkpoint:
        print(f"  Loading fine-tuned SBERT from {sbert_checkpoint}...")
        models["Fine-Tuned SBERT"] = FineTunedSBERTEncoder(str(sbert_checkpoint), device=device).load()
    else:
        print("  [SKIP] Fine-tuned SBERT checkpoint not found")

    finbert_checkpoint = _resolve_checkpoint(args.finbert_path, "finbert")
    if finbert_checkpoint:
        print(f"  Loading fine-tuned FinBERT from {finbert_checkpoint}...")
        models["Fine-Tuned FinBERT"] = FineTunedFinBERTEncoder(str(finbert_checkpoint), device=device).load()
    else:
        print("  [SKIP] Fine-tuned FinBERT checkpoint not found")

    return models


def evaluate_single_seed(
    models: Dict[str, BaseEncoder], evaluator: RetrievalEvaluator, seed: int, args: argparse.Namespace
) -> Dict[str, Dict[str, float]]:
    """Seed the RNGs once, then evaluate every model in order."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    seed_results = {}
    for model_name, encoder in models.items():
        print(f"\n  Evaluating {model_name}...")
        result = evaluator.evaluate(
            encoder=encoder, num_queries=args.num_queries, batch_size=args.batch_size, show_progress=False
        )
        evaluation_metrics = {
            name: value for name, value in result.metrics.items() if not name.endswith("_std")
        }
        seed_results[model_name] = evaluation_metrics
        print(
            f"    P@5: {evaluation_metrics.get('p@5', 0):.4f}, MRR: {evaluation_metrics.get('mrr', 0):.4f}, "
            f"nDCG@10: {evaluation_metrics.get('ndcg@10', 0):.4f}"
        )
    return seed_results


def main() -> None:
    args = parse_args()
    if args.config:
        load_config(args.config)
    config = get_config()

    print("=" * 60 + "\nModel Evaluation: 4-Way Comparison\n" + "=" * 60)
    device = resolve_device(args.no_cuda)
    print(f"\nDevice: {device}\nSeeds: {args.seeds}\nQueries per seed: {args.num_queries}")

    print("\nLoading test data...")
    dataset = load_twitter_financial_news()
    k_values = list(config.evaluation.k_values)
    evaluator = RetrievalEvaluator(
        corpus_texts=dataset.test["text"].tolist(),
        corpus_labels=dataset.test["label"].tolist(),
        k_values=k_values,
    )
    print(f"Test corpus size: {len(dataset.test)}")

    models = load_models(args, device)
    print(f"\nModels to evaluate: {list(models)}")

    all_results: Dict[str, Dict[int, Dict[str, float]]] = {model_name: {} for model_name in models}
    for seed in args.seeds:
        print(f"\n{'=' * 40}\nSeed {seed}\n{'=' * 40}")
        for model_name, evaluation_metrics in evaluate_single_seed(models, evaluator, seed, args).items():
            all_results[model_name][seed] = evaluation_metrics

    print(f"\n{'=' * 60}\nStatistical Analysis\n{'=' * 60}")
    analyzer = StatisticalAnalyzer(confidence_level=config.evaluation.confidence_level)
    analysis = analyzer.analyze_multi_seed_results(all_results)
    print(f"\nResults with {config.evaluation.confidence_level:.0%} confidence intervals:")
    print(format_statistics_table(analysis, metrics=SUMMARY_METRICS))

    comparisons = []
    if len(models) > 1 and len(args.seeds) >= 2:
        comparisons = analyzer.pairwise_comparisons(all_results, metrics=COMPARISON_METRICS)
        print("\nPairwise comparisons (paired t-test across seeds):")
        print(format_comparison_table(comparisons))

    output_path = Path(args.output) if args.output else config.paths.results_dir / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {"seeds": args.seeds, "num_queries": args.num_queries, "k_values": k_values},
        "results": all_results,
        "analysis": {
            model_name: {
                metric: {
                    "mean": stat.mean,
                    "std": stat.std,
                    "ci_lower": stat.ci_lower,
                    "ci_upper": stat.ci_upper,
                }
                for metric, stat in model_stats.items()
            }
            for model_name, model_stats in analysis.items()
        },
        "comparisons": [
            {
                "model_a": c.model_a,
                "model_b": c.model_b,
                "metric": c.metric,
                "mean_diff": float(c.mean_diff),
                "t_statistic": float(c.t_statistic),
                "p_value": float(c.p_value),
                "effect_size": float(c.effect_size),
            }
            for c in comparisons
        ],
    }
    with open(output_path, "w") as handle:
        json.dump(output_data, handle, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for metric in COMPARISON_METRICS:
        candidates = {name: stats[metric] for name, stats in analysis.items() if metric in stats}
        if candidates:
            best_model = max(candidates, key=lambda name: candidates[name].mean)
            best = candidates[best_model]
            print(
                f"Best {metric}: {best_model} ({best.mean:.4f} ± {(best.ci_upper - best.ci_lower) / 2:.4f})"
            )


if __name__ == "__main__":
    main()
