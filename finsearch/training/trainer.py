"""Triplet-loss fine-tuning loop for embedding models.

Two model types are supported:

* ``sentence_transformers.SentenceTransformer`` (SBERT): batches are tokenised
  with the model's tokenizer and passed through its module pipeline so that
  gradients flow (``SentenceTransformer.encode`` returns detached tensors).
* any ``nn.Module`` whose ``forward`` accepts a list of strings and returns
  embeddings, e.g. :class:`finsearch.models.FinBERTTripletModel`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from finsearch.config import get_config
from finsearch.config.settings import TrainingConfig
from finsearch.training.callbacks import (
    Callback,
    CallbackHandler,
    EarlyStopping,
    ModelCheckpoint,
    TrainingState,
)
from finsearch.training.objectives import TripletMarginLoss

MAX_GRAD_NORM = 1.0


def _is_sentence_transformer(model: Any) -> bool:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False
    return isinstance(model, SentenceTransformer)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingResult:
    """Summary returned by :meth:`EmbeddingTrainer.train`."""

    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    training_time: float
    history: Dict[str, List[float]]
    model_path: Optional[str] = None


class EmbeddingTrainer:
    """Fine-tune an embedding model on (anchor, positive, negative) batches.

    Args:
        model: ``SentenceTransformer`` or a string-in/embedding-out ``nn.Module``.
        train_dataloader: Yields dicts with ``anchor``/``positive``/``negative`` lists.
        val_dataloader: Same format; validation loss is ``inf`` when omitted.
        loss_fn: Defaults to cosine :class:`TripletMarginLoss` with ``training_config.margin``.
        optimizer: Defaults to AdamW (``learning_rate``, ``weight_decay``).
        scheduler: Defaults to linear warm-up for ``warmup_ratio`` of steps, then constant LR.
        callbacks: Early stopping, checkpointing, logging, ...
        device: Device for tokenised inputs (default ``training_config.device``).
        config: A :class:`TrainingConfig` (default: the global one).
    """

    def __init__(
        self,
        model: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.training_config = config or get_config().training
        self.device = device or self.training_config.device

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self._is_sentence_transformer = _is_sentence_transformer(model)

        if loss_fn is None:
            loss_fn = TripletMarginLoss(margin=self.training_config.margin, distance_metric="cosine")
        self.loss_fn = loss_fn

        if optimizer is None:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else self._create_scheduler()

        self.callback_handler = CallbackHandler(callbacks or [])
        self.state = TrainingState()
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _create_scheduler(self) -> SequentialLR:
        total_steps = len(self.train_dataloader) * self.training_config.epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        constant = ConstantLR(self.optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
        return SequentialLR(self.optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Embed a batch of texts with gradient tracking."""
        if not self._is_sentence_transformer:
            return self.model(texts)

        features = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model.max_seq_length,
            return_tensors="pt",
        )
        features = {name: tensor.to(self.device) for name, tensor in features.items()}
        for module in self.model:
            features = module(features)
        return features["sentence_embedding"]

    def _batch_loss(self, batch: Dict[str, list]) -> torch.Tensor:
        anchor_embeddings = self._encode_batch(batch["anchor"])
        positive_embeddings = self._encode_batch(batch["positive"])
        negative_embeddings = self._encode_batch(batch["negative"])
        return self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        accumulation_steps = self.training_config.gradient_accumulation_steps
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.training_config.epochs}",
            leave=True,
        )
        for batch_idx, batch in enumerate(progress):
            self.callback_handler.on_batch_begin(self.state, batch_idx)

            loss = self._batch_loss(batch) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            batch_loss = loss.item() * accumulation_steps
            total_loss += batch_loss
            num_batches += 1
            self.state.step += 1
            progress.set_postfix({"loss": f"{batch_loss:.4f}"})

            self.callback_handler.on_batch_end(self.state, batch_idx, batch_loss)

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self) -> float:
        if self.val_dataloader is None:
            return float("inf")
        self.model.eval()
        losses = [self._batch_loss(batch).item() for batch in self.val_dataloader]
        return sum(losses) / len(losses)

    def train(self) -> TrainingResult:
        """Run training for ``training_config.epochs`` epochs (or until early stopping)."""
        start_time = time.time()
        self.callback_handler.on_train_begin(self.state)

        for epoch in range(self.training_config.epochs):
            self.state.epoch = epoch
            self.callback_handler.on_epoch_begin(self.state, epoch)

            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            self.callback_handler.on_epoch_end(
                self.state, epoch, {"train_loss": train_loss, "val_loss": val_loss}, model=self.model
            )
            if self.state.should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.callback_handler.on_train_end(self.state)

        model_path = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ModelCheckpoint):
                model_path = str(callback.best_model_path) if callback.best_model_path else None
                break

        best_val_loss = self.state.best_metric
        if best_val_loss == float("inf"):
            best_val_loss = min(self.history["val_loss"])

        return TrainingResult(
            best_epoch=self.state.best_epoch,
            best_val_loss=best_val_loss,
            final_train_loss=self.history["train_loss"][-1],
            training_time=time.time() - start_time,
            history=self.history,
            model_path=model_path,
        )


def train_with_seed(
    model_factory: Callable[[], Any],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    seed: int,
    save_dir: str,
    config: Optional[TrainingConfig] = None,
) -> Tuple[TrainingResult, str]:
    """Seed all RNGs, build triplet loaders, train a fresh model and return
    ``(result, checkpoint_directory)``."""
    from finsearch.data.triplets import create_triplet_dataloader

    set_global_seed(seed)
    training_config = config or get_config().training

    train_loader = create_triplet_dataloader(
        train_data, batch_size=training_config.batch_size, seed=seed, shuffle=True
    )
    val_loader = create_triplet_dataloader(
        val_data, batch_size=training_config.eval_batch_size, seed=seed, shuffle=False
    )

    seed_save_dir = Path(save_dir) / f"seed_{seed}"
    trainer = EmbeddingTrainer(
        model=model_factory(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=[
            EarlyStopping(patience=training_config.patience, min_delta=training_config.min_delta),
            ModelCheckpoint(str(seed_save_dir), monitor="val_loss", save_best_only=True),
        ],
        config=training_config,
    )
    return trainer.train(), str(seed_save_dir)


def multi_seed_training(
    model_factory: Callable[[], Any],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    save_dir: str,
    seeds: Optional[List[int]] = None,
    config: Optional[TrainingConfig] = None,
) -> Dict[int, TrainingResult]:
    """Run :func:`train_with_seed` for each seed and print the mean/std of best val loss."""
    training_config = config or get_config().training
    seeds = seeds or training_config.random_seeds

    results: Dict[int, TrainingResult] = {}
    for position, seed in enumerate(seeds, start=1):
        print(f"\n{'=' * 60}\nTraining with seed {seed} ({position}/{len(seeds)})\n{'=' * 60}\n")
        result, model_path = train_with_seed(
            model_factory=model_factory,
            train_data=train_data,
            val_data=val_data,
            seed=seed,
            save_dir=save_dir,
            config=training_config,
        )
        result.model_path = model_path
        results[seed] = result
        print(f"\nSeed {seed} complete: best_val_loss={result.best_val_loss:.4f}")

    val_losses = [result.best_val_loss for result in results.values()]
    print(f"\n{'=' * 60}\nMulti-seed training summary\n{'=' * 60}")
    print(f"Val loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    return results
