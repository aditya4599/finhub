#!/usr/bin/env python3
"""Download the Twitter Financial News Topic dataset and print split sizes and samples."""

from finsearch.data.datasets import load_twitter_financial_news

dataset = load_twitter_financial_news()
print(f"Train: {len(dataset.train)} | Validation: {len(dataset.validation)} | Test: {len(dataset.test)}")
print(f"Topics: {dataset.num_topics}")
for label, name in dataset.topic_labels.items():
    print(f"  {label}: {name}")
print("\nSamples:")
for _, row in dataset.train.head(5).iterrows():
    print(f"  [{row['topic_name']}] {row['text'][:70]}...")
