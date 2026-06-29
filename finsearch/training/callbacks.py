"""Training callbacks: early stopping, checkpointing and TensorBoard logging."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainingState:
    """Mutable state shared between the trainer and its callbacks."""

    epoch: int = 0
    step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0
    should_stop: bool = False


class Callback:
    """No-op base class; override the hooks you need."""

    def on_train_begin(self, state: TrainingState) -> None:
        pass

    def on_train_end(self, state: TrainingState) -> None:
        pass

    def on_epoch_begin(self, state: TrainingState, epoch: int) -> None:
        pass

    def on_epoch_end(self, state: TrainingState, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_batch_begin(self, state: TrainingState, batch_idx: int) -> None:
        pass

    def on_batch_end(self, state: TrainingState, batch_idx: int, loss: float) -> None:
        pass


def _improves(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


class EarlyStopping(Callback):
    """Set ``state.should_stop`` after ``patience`` epochs without improvement.

    Args:
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change that counts as an improvement.
        monitor: Metric name to watch (default ``"val_loss"``).
        mode: ``"min"`` if lower is better, ``"max"`` otherwise.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self._best_value: Optional[float] = None
        self._epochs_without_improvement = 0

    def on_train_begin(self, state: TrainingState) -> None:
        self._best_value = None
        self._epochs_without_improvement = 0

    def on_epoch_end(self, state: TrainingState, epoch: int, metrics: Dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]

        if self._best_value is None or _improves(current, self._best_value, self.mode, self.min_delta):
            self._epochs_without_improvement = 0
            self._best_value = current
            state.best_metric = current
            state.best_epoch = epoch
            return

        self._epochs_without_improvement += 1
        state.patience_counter = self._epochs_without_improvement
        if self.verbose:
            print(
                f"EarlyStopping: no improvement for {self._epochs_without_improvement}/{self.patience} epochs"
            )
        if self._epochs_without_improvement >= self.patience:
            state.should_stop = True
            if self.verbose:
                print(f"EarlyStopping: stopping at epoch {epoch}")


class ModelCheckpoint(Callback):
    """Save the model when the monitored metric improves (or every epoch).

    Args:
        save_dir: Directory for checkpoints.
        monitor: Metric name to watch.
        mode: ``"min"`` or ``"max"``.
        save_best_only: Keep only the best checkpoint.
        filename: Checkpoint name template; ``{epoch}`` is substituted.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        filename: str = "checkpoint_epoch_{epoch}",
        verbose: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename = filename
        self.verbose = verbose
        self._best_value: Optional[float] = None
        self._best_path: Optional[Path] = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(
        self,
        state: TrainingState,
        epoch: int,
        metrics: Dict[str, float],
        model: Any,
        is_best: bool = False,
    ) -> Path:
        filename = self.filename.format(epoch=epoch)
        save_path = self.save_dir / filename

        if hasattr(model, "save"):
            model.save(str(save_path))
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(str(save_path))
        else:
            torch.save(model.state_dict(), save_path)

        with open(self.save_dir / f"{filename}_state.json", "w") as handle:
            json.dump(
                {"epoch": epoch, "step": state.step, "metrics": metrics, "is_best": is_best},
                handle,
                indent=2,
            )

        if self.verbose:
            print(f"{'[BEST] ' if is_best else ''}Saved checkpoint: {save_path}")
        return save_path

    def on_train_begin(self, state: TrainingState) -> None:
        self._best_value = None
        self._best_path = None

    def on_epoch_end(
        self,
        state: TrainingState,
        epoch: int,
        metrics: Dict[str, float],
        model: Any = None,
    ) -> None:
        if model is None:
            return

        if self.monitor not in metrics:
            if not self.save_best_only:
                self._save_checkpoint(state, epoch, metrics, model, is_best=False)
            return

        current = metrics[self.monitor]
        is_best = self._best_value is None or _improves(current, self._best_value, self.mode)

        if is_best:
            if self._best_value is not None and self.save_best_only and self._best_path is not None:
                if self._best_path.is_dir():
                    shutil.rmtree(self._best_path)
                elif self._best_path.exists():
                    self._best_path.unlink()
            self._best_value = current

        if is_best or not self.save_best_only:
            save_path = self._save_checkpoint(state, epoch, metrics, model, is_best)
            if is_best:
                self._best_path = save_path

    @property
    def best_model_path(self) -> Optional[Path]:
        """Path of the best checkpoint saved so far."""
        return self._best_path


class TensorBoardCallback(Callback):
    """Log per-batch loss and per-epoch metrics to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def on_batch_end(self, state: TrainingState, batch_idx: int, loss: float) -> None:
        self.writer.add_scalar("train/batch_loss", loss, state.step)

    def on_epoch_end(self, state: TrainingState, epoch: int, metrics: Dict[str, float]) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(f"metrics/{name}", value, epoch)

    def on_train_end(self, state: TrainingState) -> None:
        self.writer.close()


class CallbackHandler:
    """Dispatch trainer events to a list of callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(state)

    def on_train_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_train_end(state)

    def on_epoch_begin(self, state: TrainingState, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(state, epoch)

    def on_epoch_end(
        self, state: TrainingState, epoch: int, metrics: Dict[str, float], **kwargs: Any
    ) -> None:
        """``kwargs`` (e.g. ``model=``) are forwarded only to :class:`ModelCheckpoint`."""
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.on_epoch_end(state, epoch, metrics, **kwargs)
            else:
                callback.on_epoch_end(state, epoch, metrics)

    def on_batch_begin(self, state: TrainingState, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(state, batch_idx)

    def on_batch_end(self, state: TrainingState, batch_idx: int, loss: float) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(state, batch_idx, loss)
