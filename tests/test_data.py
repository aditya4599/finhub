"""
Tests for Data Pipeline

Tests for dataset loading and triplet generation.
"""


class TestTripletGenerator:
    """Tests for TripletGenerator"""

    def test_triplet_structure(self):
        """Triplets have correct structure"""
        from finsearch.data.triplets import Triplet, TripletGenerator

        texts = ["text1", "text2", "text3", "text4", "text5", "text6"]
        labels = [0, 0, 0, 1, 1, 1]

        gen = TripletGenerator(texts, labels, seed=42)
        triplets = gen.generate_triplets(n_triplets=5)

        assert len(triplets) == 5
        for t in triplets:
            assert isinstance(t, Triplet)
            assert t.anchor in texts
            assert t.positive in texts
            assert t.negative in texts

    def test_positive_same_label(self):
        """Positive should have same label as anchor"""
        from finsearch.data.triplets import TripletGenerator

        texts = ["a1", "a2", "b1", "b2", "c1", "c2"]
        labels = [0, 0, 1, 1, 2, 2]

        gen = TripletGenerator(texts, labels, seed=42)
        triplets = gen.generate_triplets(n_triplets=10)

        for t in triplets:
            assert t.anchor_label == t.positive_label

    def test_negative_different_label(self):
        """Negative should have different label from anchor"""
        from finsearch.data.triplets import TripletGenerator

        texts = ["a1", "a2", "b1", "b2", "c1", "c2"]
        labels = [0, 0, 1, 1, 2, 2]

        gen = TripletGenerator(texts, labels, seed=42)
        triplets = gen.generate_triplets(n_triplets=10)

        for t in triplets:
            assert t.anchor_label != t.negative_label

    def test_deterministic_with_seed(self):
        """Same seed should produce same triplets"""
        from finsearch.data.triplets import TripletGenerator

        texts = ["a", "b", "c", "d"]
        labels = [0, 0, 1, 1]

        gen1 = TripletGenerator(texts, labels, seed=42)
        gen2 = TripletGenerator(texts, labels, seed=42)

        t1 = gen1.generate_triplets(n_triplets=5)
        t2 = gen2.generate_triplets(n_triplets=5)

        for i in range(5):
            assert t1[i].anchor == t2[i].anchor
            assert t1[i].positive == t2[i].positive
            assert t1[i].negative == t2[i].negative


class TestTripletDataset:
    """Tests for TripletDataset"""

    def test_len(self):
        """Dataset length matches triplet count"""
        from finsearch.data.triplets import TripletDataset, TripletGenerator

        texts = ["a", "b", "c", "d"]
        labels = [0, 0, 1, 1]

        gen = TripletGenerator(texts, labels, seed=42)
        triplets = gen.generate_triplets(n_triplets=10)
        dataset = TripletDataset(triplets)

        assert len(dataset) == 10

    def test_getitem(self):
        """Can retrieve items by index"""
        from finsearch.data.triplets import TripletDataset, TripletGenerator

        texts = ["a", "b", "c", "d"]
        labels = [0, 0, 1, 1]

        gen = TripletGenerator(texts, labels, seed=42)
        triplets = gen.generate_triplets(n_triplets=5)
        dataset = TripletDataset(triplets, return_labels=False)

        item = dataset[0]
        assert "anchor" in item
        assert "positive" in item
        assert "negative" in item

    def test_return_labels(self):
        """Can optionally return labels"""
        from finsearch.data.triplets import TripletDataset, TripletGenerator

        texts = ["a", "b", "c", "d"]
        labels = [0, 0, 1, 1]

        gen = TripletGenerator(texts, labels, seed=42)
        triplets = gen.generate_triplets(n_triplets=5)
        dataset = TripletDataset(triplets, return_labels=True)

        item = dataset[0]
        assert "anchor_label" in item
        assert "positive_label" in item
        assert "negative_label" in item
