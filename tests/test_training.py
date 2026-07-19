"""Tests for objectives, callbacks and the training loop (offline, tiny models)."""

import pandas as pd
import pytest
import torch

from finsearch.config.settings import TrainingConfig
from finsearch.data.triplets import create_triplet_dataloader
from finsearch.models.encoders import FinBERTTripletModel
from finsearch.training.callbacks import CallbackHandler, EarlyStopping, ModelCheckpoint, TrainingState
from finsearch.training.objectives import OnlineTripletLoss, TripletMarginLoss
from finsearch.training.trainer import EmbeddingTrainer


class TestTripletMarginLoss:
    def test_zero_when_negative_far_enough(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[-1.0, 0.0]])
        # d(a,p)=0, d(a,n)=2 -> max(0, 0 - 2 + 0.5) = 0
        assert TripletMarginLoss(margin=0.5)(anchor, positive, negative).item() == pytest.approx(0.0)

    def test_value_matches_formula(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.0, 1.0]])  # cosine distance 1
        negative = torch.tensor([[1.0, 0.0]])  # cosine distance 0
        assert TripletMarginLoss(margin=0.5)(anchor, positive, negative).item() == pytest.approx(1.5)

    def test_euclidean_and_reduction_none(self):
        loss_fn = TripletMarginLoss(margin=1.0, distance_metric="euclidean", reduction="none")
        losses = loss_fn(torch.zeros(3, 2), torch.zeros(3, 2), torch.ones(3, 2))
        assert losses.shape == (3,)

    def test_unknown_metric(self):
        with pytest.raises(ValueError):
            TripletMarginLoss(distance_metric="manhattan")(
                torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)
            )


class TestOnlineTripletLoss:
    @pytest.mark.parametrize("strategy", ["hard", "all"])
    def test_runs_and_is_non_negative(self, strategy):
        torch.manual_seed(0)
        loss = OnlineTripletLoss(margin=0.5, mining_strategy=strategy)(
            torch.randn(8, 4), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        )
        assert loss.item() >= 0


class TestCallbacks:
    def test_early_stopping_sequence(self):
        """Same scenario as the original callbacks demo."""
        state = TrainingState()
        handler = CallbackHandler([EarlyStopping(patience=2, min_delta=0.01, verbose=False)])
        handler.on_train_begin(state)
        stopped_at = None
        for epoch, val_loss in enumerate([1.0, 0.8, 0.6, 0.59, 0.58, 0.58, 0.58]):
            handler.on_epoch_end(state, epoch, {"val_loss": val_loss})
            if state.should_stop:
                stopped_at = epoch
                break
        # 0.59 is within min_delta of 0.6 (no improvement); 0.58 improves on 0.6;
        # the two following 0.58s do not -> stop at epoch 6.
        assert stopped_at == 6
        assert state.best_metric == pytest.approx(0.58)
        assert state.best_epoch == 4

    def test_checkpoint_keeps_only_best(self, tmp_path):
        class Saveable:
            def save(self, path):
                open(path, "w").close()

        checkpoint = ModelCheckpoint(str(tmp_path), save_best_only=True, verbose=False)
        state = TrainingState()
        checkpoint.on_train_begin(state)
        for epoch, val_loss in enumerate([1.0, 0.5, 0.7]):
            checkpoint.on_epoch_end(state, epoch, {"val_loss": val_loss}, model=Saveable())
        assert checkpoint.best_model_path == tmp_path / "checkpoint_epoch_1"
        assert not (tmp_path / "checkpoint_epoch_0").exists()
        assert not (tmp_path / "checkpoint_epoch_2").exists()


def _toy_split(n_per_label: int = 6) -> pd.DataFrame:
    words = {0: "apple earnings rise", 1: "fed rates bond", 2: "oil price gold"}
    rows = [{"text": f"{words[label]} {i}", "label": label} for label in words for i in range(n_per_label)]
    return pd.DataFrame(rows)


class TestEmbeddingTrainer:
    def test_finbert_style_module_trains_and_checkpoints(self, tiny_bert_dir, tmp_path):
        """Regression test: the original FinBERT training path could not run."""
        torch.manual_seed(0)
        training_config = TrainingConfig(epochs=2, batch_size=4, device="cpu", gradient_accumulation_steps=2)
        loader = create_triplet_dataloader(_toy_split(), batch_size=4, seed=0)
        model = FinBERTTripletModel(str(tiny_bert_dir), device="cpu")
        initial = [p.detach().clone() for p in model.parameters()]

        trainer = EmbeddingTrainer(
            model=model,
            train_dataloader=loader,
            val_dataloader=loader,
            callbacks=[ModelCheckpoint(str(tmp_path), filename="best_model", verbose=False)],
            config=training_config,
        )
        result = trainer.train()

        assert len(result.history["train_loss"]) == 2
        assert result.model_path == str(tmp_path / "best_model")
        assert (tmp_path / "best_model" / "config.json").exists()
        assert any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))

    def test_sentence_transformer_trains(self, tiny_bert_dir):
        st = pytest.importorskip("sentence_transformers")
        from sentence_transformers import models as st_models

        torch.manual_seed(0)
        transformer = st_models.Transformer(str(tiny_bert_dir), max_seq_length=32)
        get_dimension = (
            getattr(transformer, "get_embedding_dimension", None) or transformer.get_word_embedding_dimension
        )
        pooling = st_models.Pooling(get_dimension())
        model = st.SentenceTransformer(modules=[transformer, pooling], device="cpu")

        training_config = TrainingConfig(epochs=1, batch_size=4, device="cpu")
        loader = create_triplet_dataloader(_toy_split(), batch_size=4, seed=0)
        result = EmbeddingTrainer(model=model, train_dataloader=loader, config=training_config).train()
        assert result.history["train_loss"][0] >= 0
