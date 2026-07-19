#!/usr/bin/env python3
"""Print the active configuration (no downloads)."""

from finsearch.config import get_config

config = get_config()
print("=== CONFIG ===")
print(f"Learning rate: {config.training.learning_rate}")
print(f"Epochs:        {config.training.epochs}")
print(f"Margin:        {config.training.margin}")
print(f"Batch size:    {config.training.batch_size}")
print(f"SBERT model:   {config.model.sbert_model_name}")
print(f"FinBERT model: {config.model.finbert_model_name}")
print(f"Device:        {config.training.device}")
print(f"Project root:  {config.paths.project_root}")
