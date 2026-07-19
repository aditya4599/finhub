#!/usr/bin/env python3
"""Generate five triplets from the training split and check their labels."""

from finsearch.data.datasets import load_twitter_financial_news
from finsearch.data.triplets import TripletGenerator

dataset = load_twitter_financial_news()
generator = TripletGenerator(
    texts=dataset.train["text"].tolist()[:500],
    labels=dataset.train["label"].tolist()[:500],
    seed=42,
)
for number, triplet in enumerate(generator.generate_triplets(n_triplets=5), start=1):
    print(f"\n--- Triplet {number} ---")
    print(f"ANCHOR   [{dataset.topic_labels[triplet.anchor_label]}]: {triplet.anchor[:70]}")
    print(f"POSITIVE [{dataset.topic_labels[triplet.positive_label]}]: {triplet.positive[:70]}")
    print(f"NEGATIVE [{dataset.topic_labels[triplet.negative_label]}]: {triplet.negative[:70]}")
    assert triplet.anchor_label == triplet.positive_label, "positive has a different topic"
    assert triplet.anchor_label != triplet.negative_label, "negative has the same topic"
print("\nAll triplets have correct labels.")
