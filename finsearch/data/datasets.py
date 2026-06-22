"""Labelled training data: the Twitter Financial News Topic dataset.

Source: https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic
(financial tweets labelled with one of 20 topics). The Hub "validation" split
is divided 90/10 (stratified, ``random_state=42``) into validation and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from finsearch.config import get_config


@dataclass
class FinancialTextDataset:
    """Train / validation / test DataFrames with ``text``, ``label`` and ``topic_name`` columns."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    topic_labels: Dict[int, str]

    @property
    def num_topics(self) -> int:
        return len(self.topic_labels)

    @property
    def total_samples(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def get_topic_distribution(self, split: str = "train") -> Dict[str, int]:
        """Return ``{topic_name: count}`` for one split."""
        split_df = getattr(self, split)
        counts = split_df["label"].value_counts().to_dict()
        return {self.topic_labels[label]: count for label, count in counts.items()}


def load_twitter_financial_news(
    test_size: float = 0.1,
    cache_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> FinancialTextDataset:
    """Download (or reuse the cache of) the Twitter Financial News Topic dataset.

    Args:
        test_size: Fraction of the Hub validation split held out as the test set.
        cache_dir: Hugging Face datasets cache directory (default: ``paths.data_dir``).
        force_reload: Re-download even if a cached copy exists.
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    config = get_config()
    cache_dir = cache_dir or config.paths.data_dir

    hub_dataset = load_dataset(
        config.dataset_name,
        cache_dir=str(cache_dir),
        download_mode="force_redownload" if force_reload else "reuse_dataset_if_exists",
    )

    train_df = hub_dataset["train"].to_pandas()
    validation_df = hub_dataset["validation"].to_pandas()

    validation_df, test_df = train_test_split(
        validation_df,
        test_size=test_size,
        stratify=validation_df["label"],
        random_state=42,
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for split_df in (train_df, validation_df, test_df):
        split_df["topic_name"] = split_df["label"].map(config.topic_labels)

    return FinancialTextDataset(
        train=train_df,
        validation=validation_df,
        test=test_df,
        topic_labels=config.topic_labels,
    )


def stratified_subset(split_df: pd.DataFrame, samples_per_label: int, seed: int) -> pd.DataFrame:
    """Sample up to ``samples_per_label`` rows from every label (used to subset training data)."""
    return (
        split_df.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), samples_per_label), random_state=seed))
        .reset_index(drop=True)
    )


def select_training_subset(
    dataset: FinancialTextDataset,
    subset_size: Optional[int],
    seed: int,
    quick: bool = False,
    quick_train_size: int = 500,
    quick_validation_size: int = 100,
) -> FinancialTextDataset:
    """Shrink the train/validation splits in place for faster fine-tuning.

    * ``quick``: keep the first ``quick_train_size`` / ``quick_validation_size`` rows.
    * otherwise, if ``subset_size`` is smaller than the training split: sample
      ``subset_size // num_topics`` rows per topic from train (seeded) and keep the
      first ``subset_size // 4`` validation rows.
    """
    if quick:
        dataset.train = dataset.train.head(quick_train_size)
        dataset.validation = dataset.validation.head(quick_validation_size)
    elif subset_size and subset_size < len(dataset.train):
        dataset.train = stratified_subset(dataset.train, subset_size // dataset.num_topics, seed)
        dataset.validation = dataset.validation.head(min(len(dataset.validation), subset_size // 4))
    return dataset


def get_samples_by_topic(
    dataset: FinancialTextDataset,
    topic_id: int,
    split: str = "train",
    n_samples: Optional[int] = None,
) -> List[str]:
    """Return the texts of one topic from one split (optionally truncated)."""
    split_df = getattr(dataset, split)
    samples = split_df[split_df["label"] == topic_id]["text"].tolist()
    return samples if n_samples is None else samples[:n_samples]


def get_topic_pairs(dataset: FinancialTextDataset) -> List[Tuple[int, int]]:
    """Return every unordered pair of distinct topic ids."""
    return list(combinations(dataset.topic_labels.keys(), 2))
