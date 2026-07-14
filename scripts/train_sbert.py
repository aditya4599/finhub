#!/usr/bin/env python3
"""Fine-tune Sentence-BERT (all-MiniLM-L6-v2) with triplet loss on the Twitter
Financial News Topic dataset.

Usage:
    python scripts/train_sbert.py --subset 4000 --epochs 3 --no_cuda
    python scripts/train_sbert.py --seeds 42 123 456 --quick

Flags override the matching YAML values (epochs, batch size, learning rate, margin).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from finsearch.config import get_config, load_config
from finsearch.data.datasets import load_twitter_financial_news, select_training_subset
from finsearch.data.triplets import create_triplet_dataloader
from finsearch.training.callbacks import EarlyStopping, ModelCheckpoint, TensorBoardCallback
from finsearch.training.trainer import EmbeddingTrainer, TrainingResult, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Sentence-BERT with triplet loss")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet loss margin")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Checkpoint root (default: <models_dir>/sbert)"
    )
    parser.add_argument("--subset", type=int, default=4000, help="Approximate number of training samples")
    parser.add_argument("--quick", action="store_true", help="Smoke run on 500 train / 100 validation rows")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    return parser.parse_args()


def train_single_seed(args: argparse.Namespace, seed: int) -> TrainingResult:
    """Train one SBERT model for ``seed`` and save its best checkpoint."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 60}\nTraining with seed {seed}\n{'=' * 60}\n")
    set_global_seed(seed)

    config = get_config()
    training_config = config.training
    training_config.epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.lr
    training_config.margin = args.margin
    if args.no_cuda:
        training_config.device = "cpu"
    print(f"Device: {training_config.device}")

    print("Loading Twitter Financial News dataset...")
    dataset = select_training_subset(
        load_twitter_financial_news(), args.subset, seed, quick=args.quick, quick_train_size=500
    )
    print(f"Train samples: {len(dataset.train)}")
    print(f"Validation samples: {len(dataset.validation)}")

    train_loader = create_triplet_dataloader(
        dataset.train, batch_size=training_config.batch_size, seed=seed, shuffle=True
    )
    val_loader = create_triplet_dataloader(
        dataset.validation, batch_size=training_config.eval_batch_size, seed=seed, shuffle=False
    )
    print(f"Train triplets: {len(train_loader.dataset)}")
    print(f"Validation triplets: {len(val_loader.dataset)}")

    print("\nLoading Sentence-BERT model...")
    model = SentenceTransformer(config.model.sbert_model_name, device=training_config.device)

    save_root = Path(args.save_dir) if args.save_dir else config.paths.models_dir / "sbert"
    seed_dir = save_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    trainer = EmbeddingTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=[
            EarlyStopping(
                patience=training_config.patience,
                min_delta=training_config.min_delta,
                monitor="val_loss",
                verbose=True,
            ),
            ModelCheckpoint(
                save_dir=str(seed_dir),
                monitor="val_loss",
                save_best_only=True,
                filename="best_model",
                verbose=True,
            ),
            TensorBoardCallback(log_dir=str(config.paths.logs_dir / "sbert")),
        ],
        device=training_config.device,
        config=training_config,
    )

    print("\nStarting training...")
    result = trainer.train()
    print("\nTraining complete!")
    print(f"Best val loss: {result.best_val_loss:.4f} at epoch {result.best_epoch + 1}")
    print(f"Training time: {result.training_time:.1f}s")
    print(f"Model saved to: {seed_dir / 'best_model'}")
    return result


def main() -> None:
    args = parse_args()
    if args.config:
        load_config(args.config)

    print("=" * 60 + "\nSentence-BERT Fine-Tuning with Triplet Loss\n" + "=" * 60)
    print(
        f"\nEpochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr} | "
        f"Margin: {args.margin} | Seeds: {args.seeds}"
    )

    results = {seed: train_single_seed(args, seed) for seed in args.seeds}

    if len(args.seeds) > 1:
        val_losses = [result.best_val_loss for result in results.values()]
        print(f"\n{'=' * 60}\nMulti-seed summary\n{'=' * 60}")
        print(f"Val loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        for seed, result in results.items():
            print(f"  Seed {seed}: {result.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
