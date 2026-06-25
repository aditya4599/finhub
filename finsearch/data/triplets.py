"""Triplet construction for contrastive (triplet-loss) training.

For each anchor text:
* the positive is a random text with the SAME topic label (a different row
  when one exists, otherwise the anchor itself);
* the negative is a random text from a randomly chosen DIFFERENT topic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


@dataclass
class Triplet:
    """One (anchor, positive, negative) training example with topic labels."""

    anchor: str
    positive: str
    negative: str
    anchor_label: int
    positive_label: int
    negative_label: int


class TripletGenerator:
    """Sample label-based triplets from a labelled text collection.

    Args:
        texts: Text samples.
        labels: Integer topic label for each text.
        seed: Seed for the generator's private ``random.Random`` instance.
    """

    def __init__(self, texts: List[str], labels: List[int], seed: int = 42):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length (got {len(texts)} and {len(labels)})"
            )
        self.texts = texts
        self.labels = labels
        self.seed = seed
        self.rng = random.Random(seed)

        self.topic_to_indices: Dict[int, List[int]] = {}
        for index, label in enumerate(labels):
            self.topic_to_indices.setdefault(label, []).append(index)
        self.all_topics = list(self.topic_to_indices.keys())

    def _sample_positive_index(self, anchor_index: int) -> int:
        anchor_label = self.labels[anchor_index]
        same_topic_indices = [i for i in self.topic_to_indices[anchor_label] if i != anchor_index]
        return self.rng.choice(same_topic_indices) if same_topic_indices else anchor_index

    def generate_triplet(self, anchor_index: int) -> Triplet:
        """Build one triplet for the text at ``anchor_index``."""
        anchor_label = self.labels[anchor_index]
        positive_index = self._sample_positive_index(anchor_index)

        other_topics = [topic for topic in self.all_topics if topic != anchor_label]
        if not other_topics:
            raise ValueError("Triplet generation needs at least two distinct topic labels.")
        negative_topic = self.rng.choice(other_topics)
        negative_index = self.rng.choice(self.topic_to_indices[negative_topic])

        return Triplet(
            anchor=self.texts[anchor_index],
            positive=self.texts[positive_index],
            negative=self.texts[negative_index],
            anchor_label=anchor_label,
            positive_label=self.labels[positive_index],
            negative_label=self.labels[negative_index],
        )

    def generate_triplets(
        self,
        n_triplets: Optional[int] = None,
        triplets_per_anchor: int = 1,
    ) -> List[Triplet]:
        """Generate triplets.

        Args:
            n_triplets: If given, sample exactly this many anchors (with replacement).
                If ``None``, use every text as an anchor ``triplets_per_anchor`` times.
            triplets_per_anchor: Triplets per anchor when ``n_triplets`` is ``None``.
        """
        if n_triplets is None:
            return [
                self.generate_triplet(anchor_index)
                for anchor_index in range(len(self.texts))
                for _ in range(triplets_per_anchor)
            ]
        anchor_indices = self.rng.choices(range(len(self.texts)), k=n_triplets)
        return [self.generate_triplet(anchor_index) for anchor_index in anchor_indices]

    def generate_hard_negatives(
        self,
        anchor_index: int,
        embeddings: np.ndarray,
        n_candidates: int = 10,
    ) -> Triplet:
        """Build a triplet whose negative is one of the ``n_candidates`` different-topic
        texts closest (by cosine similarity) to the anchor in ``embeddings``."""
        anchor_label = self.labels[anchor_index]
        anchor_embedding = embeddings[anchor_index]

        different_topic_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        candidate_embeddings = embeddings[different_topic_indices]
        similarities = np.dot(candidate_embeddings, anchor_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(anchor_embedding)
        )
        hardest_local_indices = np.argsort(similarities)[-n_candidates:]
        negative_index = different_topic_indices[self.rng.choice(hardest_local_indices)]

        positive_index = self._sample_positive_index(anchor_index)

        return Triplet(
            anchor=self.texts[anchor_index],
            positive=self.texts[positive_index],
            negative=self.texts[negative_index],
            anchor_label=anchor_label,
            positive_label=self.labels[positive_index],
            negative_label=self.labels[negative_index],
        )


class TripletDataset(Dataset):
    """PyTorch dataset over :class:`Triplet` objects (items are dicts of strings/labels)."""

    def __init__(self, triplets: List[Triplet], return_labels: bool = False):
        self.triplets = triplets
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> Dict[str, object]:
        triplet = self.triplets[index]
        item: Dict[str, object] = {
            "anchor": triplet.anchor,
            "positive": triplet.positive,
            "negative": triplet.negative,
        }
        if self.return_labels:
            item.update(
                anchor_label=triplet.anchor_label,
                positive_label=triplet.positive_label,
                negative_label=triplet.negative_label,
            )
        return item


def triplet_collate_fn(batch: List[dict]) -> Dict[str, list]:
    """Collate a list of labelled triplet dicts into a dict of lists."""
    keys = ("anchor", "positive", "negative", "anchor_label", "positive_label", "negative_label")
    return {key: [item[key] for item in batch] for key in keys}


def create_triplet_dataloader(
    split_df: pd.DataFrame,
    batch_size: int = 32,
    n_triplets: Optional[int] = None,
    triplets_per_anchor: int = 1,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Generate triplets from a DataFrame with ``text`` and ``label`` columns and wrap
    them in a DataLoader that yields dicts of lists."""
    generator = TripletGenerator(
        texts=split_df["text"].tolist(),
        labels=split_df["label"].tolist(),
        seed=seed,
    )
    triplets = generator.generate_triplets(
        n_triplets=n_triplets,
        triplets_per_anchor=triplets_per_anchor,
    )
    return DataLoader(
        TripletDataset(triplets, return_labels=True),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=triplet_collate_fn,
    )
