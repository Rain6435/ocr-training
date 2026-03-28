import numpy as np
import pytest
import tensorflow as tf

from src.classifier.model import build_difficulty_classifier
from src.classifier.dataset import simulate_degradation


class TestClassifierModel:
    def test_build_model(self):
        model = build_difficulty_classifier()
        assert model is not None
        assert model.output_shape == (None, 3)

    def test_model_inference(self):
        model = build_difficulty_classifier()
        dummy = np.random.rand(1, 128, 128, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        assert output.shape == (1, 3)
        assert abs(np.sum(output) - 1.0) < 1e-5  # softmax sums to 1

    def test_model_param_count(self):
        model = build_difficulty_classifier()
        total_params = model.count_params()
        assert total_params < 500_000  # Should be ~200K


class TestDegradation:
    def test_simulate_degradation(self):
        img = np.full((128, 128), 200, dtype=np.uint8)
        degraded = simulate_degradation(img)
        assert degraded.shape == img.shape
        assert degraded.dtype == np.uint8

    def test_degradation_changes_image(self):
        img = np.full((128, 128), 200, dtype=np.uint8)
        rng = np.random.default_rng(42)
        degraded = simulate_degradation(img, rng=rng)
        # Should be different from original (with high probability)
        assert not np.array_equal(img, degraded)
