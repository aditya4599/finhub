"""Tests for the encoder wrappers."""

import numpy as np
import pytest
import torch

from finsearch.models.encoders import BaseEncoder, FinBERTEncoder, FinBERTTripletModel, mean_pooling


class MockEncoder(BaseEncoder):
    def load(self):
        return self

    def encode(self, texts, **kwargs):
        pass

    def encode_queries(self, queries, **kwargs):
        pass

    def save(self, path):
        pass

    def load_from_path(self, path):
        return self


class TestBaseEncoder:
    def test_similarity_computation(self):
        encoder = MockEncoder("test", "cpu")
        v1 = np.array([[1, 0, 0], [0, 1, 0]])
        v2 = np.array([[1, 0, 0], [0, 0, 1]])
        sim = encoder.similarity(v1, v2)
        assert sim.shape == (2, 2)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[1, 1] == pytest.approx(0.0)

    def test_most_similar(self):
        encoder = MockEncoder("test", "cpu")
        query = np.array([1, 0, 0])
        corpus = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]])
        results = encoder.most_similar(query, corpus, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0)
        assert results[1][0] == 1


class TestMeanPooling:
    def test_ignores_padding(self):
        token_embeddings = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])
        pooled = mean_pooling(token_embeddings, attention_mask)
        assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


class TestFinBERTOffline:
    """FinBERT code paths exercised with a tiny random BERT (no download)."""

    @pytest.mark.parametrize("pooling", ["mean", "cls", "max"])
    def test_encode_shapes_and_norms(self, tiny_bert_dir, pooling):
        encoder = FinBERTEncoder(model_name=str(tiny_bert_dir), device="cpu", pooling_strategy=pooling).load()
        embeddings = encoder.encode(["apple earnings rise", "fed rates"], show_progress=False)
        assert embeddings.shape == (2, 16)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

    def test_unknown_pooling_raises(self, tiny_bert_dir):
        encoder = FinBERTEncoder(
            model_name=str(tiny_bert_dir), device="cpu", pooling_strategy="median"
        ).load()
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            encoder.encode(["apple"], show_progress=False)

    def test_save_and_reload_round_trip(self, tiny_bert_dir, tmp_path):
        encoder = FinBERTEncoder(model_name=str(tiny_bert_dir), device="cpu").load()
        before = encoder.encode(["tesla stock fall"], show_progress=False)
        encoder.save(str(tmp_path / "ckpt"))
        reloaded = FinBERTEncoder(device="cpu").load_from_path(str(tmp_path / "ckpt"))
        after = reloaded.encode(["tesla stock fall"], show_progress=False)
        assert np.allclose(before, after, atol=1e-6)

    def test_triplet_model_matches_encoder(self, tiny_bert_dir):
        """The training module and the inference encoder produce the same embeddings."""
        triplet_model = FinBERTTripletModel(str(tiny_bert_dir), device="cpu")
        encoder = FinBERTEncoder(model_name=str(tiny_bert_dir), device="cpu").load()
        texts = ["oil price rise", "bank merger deal"]
        training_embeddings = triplet_model.encode(texts).numpy()
        inference_embeddings = encoder.encode(texts, show_progress=False)
        assert np.allclose(training_embeddings, inference_embeddings, atol=1e-5)


# Integration tests that download real models from the Hugging Face Hub.
@pytest.mark.slow
class TestSBERTEncoderDownload:
    def test_load_and_encode(self):
        from finsearch.models.encoders import SBERTEncoder

        encoder = SBERTEncoder().load()
        embeddings = encoder.encode(["Hello world", "Financial news"], show_progress=False)
        assert embeddings.shape == (2, 384)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
        assert encoder.embedding_dim == 384


@pytest.mark.slow
class TestFinBERTEncoderDownload:
    def test_load_and_encode(self):
        encoder = FinBERTEncoder().load()
        embeddings = encoder.encode(["Stock market rises", "Earnings beat expectations"], show_progress=False)
        assert embeddings.shape == (2, 768)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
