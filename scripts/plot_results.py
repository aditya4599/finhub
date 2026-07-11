#!/usr/bin/env python3
"""Create figures for training losses and retrieval evaluation metrics.

Examples:
    python scripts/plot_results.py
    python scripts/plot_results.py --results results/inherited/evaluation_results.json \
        --output-dir results/figures

The loss values below are historical values supplied with the project; they
are not read from training logs. Treat those curves as illustrative unless you
can independently verify them against the original logs or checkpoints. The
performance chart is calculated from the selected evaluation-results JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finsearch.config import get_config

# Historical values supplied with the project. They have not been independently
# verified against training logs or checkpoints.
LOSS_EPOCHS = (1, 2, 3)
LOSS_SERIES = {
    "SBERT": {
        "train": (0.2675, 0.1301, 0.0818),
        "val": (0.2068, 0.1599, 0.1436),
    },
    "FinBERT": {
        "train": (0.2511, 0.0789, 0.0408),
        "val": (0.1709, 0.1471, 0.1419),
    },
}
PERFORMANCE_METRICS = {"MRR": "mrr", "P@5": "p@5"}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    paths = get_config().paths
    parser = argparse.ArgumentParser(
        description="Plot historical loss curves and retrieval evaluation metrics."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=paths.results_dir / "inherited" / "evaluation_results.json",
        help="JSON file containing per-seed evaluation results and summary statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.results_dir / "figures",
        help="Directory where PNG figures will be written.",
    )
    return parser.parse_args()


def plot_loss_curves(split: str, title: str, output_path: Path) -> None:
    """Plot the supplied training or validation loss series."""
    import matplotlib.pyplot as plt

    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = ("b-o", "r-s")
    for (model_name, losses), style in zip(LOSS_SERIES.items(), styles):
        ax.plot(LOSS_EPOCHS, losses[split], style, label=model_name, linewidth=2)

    ax.set_title(f"{title} (historical values)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triplet loss")
    ax.set_xticks(LOSS_EPOCHS)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def _load_analysis(results_path: Path) -> dict[str, Any]:
    """Read and validate the summary section of an evaluation JSON file."""
    try:
        with results_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read results file: {results_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Results file is not valid JSON: {results_path}") from exc

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Results JSON must contain an 'analysis' object")
    return analysis


def plot_performance(results_path: Path, output_path: Path) -> None:
    """Plot mean MRR and precision-at-five for each model in a results file."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    analysis = _load_analysis(results_path)
    rows = []
    for model_name, model_stats in analysis.items():
        for label, metric_key in PERFORMANCE_METRICS.items():
            metric = model_stats.get(metric_key)
            if isinstance(metric, dict) and isinstance(metric.get("mean"), (int, float)):
                rows.append({"Model": model_name, "Metric": label, "Score": metric["mean"]})

    if not rows:
        raise ValueError(
            "No supported metrics found. Expected analysis entries containing "
            "numeric means for 'mrr' and/or 'p@5'."
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(rows), x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title(f"Retrieval performance (mean across seeds)\nSource: {results_path.name}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def main() -> None:
    """Generate the two loss plots and the evaluation-performance plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    plot_loss_curves("train", "Training loss", args.output_dir / "training_loss_comparison.png")
    plot_loss_curves("val", "Validation loss", args.output_dir / "validation_loss_comparison.png")
    plot_performance(args.results, args.output_dir / "model_performance.png")


if __name__ == "__main__":
    main()
