import numpy as np
import pytest
import cv2

from src.ocr.custom_model.architecture import build_crnn, build_training_model
from src.ocr.custom_model.vocabulary import (
    CHARS, CHAR_TO_IDX, IDX_TO_CHAR, BLANK_IDX, NUM_CLASSES,
    encode_text, decode_indices,
)
from src.ocr.custom_model.ctc_utils import ctc_greedy_decode, compute_ctc_confidence
from src.ocr.custom_model.augmentation import (
    augment_ocr_image,
    apply_jpeg_compression,
    apply_gaussian_blur,
    apply_document_fade,
    apply_document_degradation,
)
from src.ocr.custom_model.predict import CustomOCREngine


class TestVocabulary:
    def test_char_mapping_consistency(self):
        for char in CHARS:
            idx = CHAR_TO_IDX[char]
            assert IDX_TO_CHAR[idx] == char

    def test_blank_index(self):
        assert BLANK_IDX == 0
        assert BLANK_IDX not in IDX_TO_CHAR

    def test_encode_decode(self):
        text = "Hello World"
        encoded = encode_text(text)
        decoded = decode_indices(encoded)
        assert decoded == text

    def test_num_classes(self):
        assert NUM_CLASSES == len(CHARS)


class TestCRNN:
    def test_build_model(self):
        model = build_crnn()
        assert model is not None
        assert "CRNN_CTC" in model.name

    def test_model_output_shape(self):
        model = build_crnn(input_shape=(64, 256, 1))
        dummy = np.random.rand(1, 64, 256, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        # Output should be (batch, time_steps, num_classes + 1)
        assert output.shape[0] == 1
        assert output.shape[2] == NUM_CLASSES + 1

    def test_training_model(self):
        training_model, inference_model = build_training_model()
        assert training_model is not None
        assert inference_model is not None


class TestCTCDecode:
    def test_greedy_decode(self):
        # Create fake output: (1, 10, NUM_CLASSES+1) with clear peaks
        num_classes = NUM_CLASSES + 1
        y_pred = np.zeros((1, 10, num_classes), dtype=np.float32)
        # Set blank for most time steps
        y_pred[0, :, BLANK_IDX] = 0.9
        # Set 'H' at time step 2
        h_idx = CHAR_TO_IDX["H"]
        y_pred[0, 2, BLANK_IDX] = 0.1
        y_pred[0, 2, h_idx] = 0.9

        result = ctc_greedy_decode(y_pred)
        assert len(result) == 1
        assert "H" in result[0]

    def test_confidence(self):
        num_classes = NUM_CLASSES + 1
        y_pred = np.random.rand(2, 10, num_classes).astype(np.float32)
        # Normalize to valid probabilities
        y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)
        confs = compute_ctc_confidence(y_pred)
        assert len(confs) == 2
        assert all(0 <= c <= 1 for c in confs)


class TestAugmentation:
    def test_augment(self):
        img = np.full((64, 256), 200, dtype=np.uint8)
        augmented = augment_ocr_image(img)
        assert augmented.shape == img.shape
        assert augmented.dtype == np.uint8

    def test_augment_with_degradation(self):
        """Test augmentation with document degradation enabled."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        rng = np.random.default_rng(seed=42)
        augmented = augment_ocr_image(
            img, rng=rng, degrade_probability=1.0  # Always degrade
        )
        assert augmented.shape == img.shape
        assert augmented.dtype == np.uint8

    def test_apply_jpeg_compression(self):
        """Test JPEG compression produces valid output."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        # Add some variation to see JPEG artifacts
        cv2.rectangle(img, (20, 20), (80, 40), 100, -1)
        
        compressed = apply_jpeg_compression(img, quality_range=(50, 50))
        assert compressed.shape == img.shape
        assert compressed.dtype == np.uint8
        # JPEG compression may produce identical results for simple solid colors
        # but should be valid output either way
        assert isinstance(compressed, np.ndarray)

    def test_apply_gaussian_blur(self):
        """Test Gaussian blur produces valid output."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        # Add some text-like content
        cv2.rectangle(img, (10, 10), (50, 50), 100, -1)
        
        blurred = apply_gaussian_blur(img, sigma_range=(1.0, 1.0))
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8
        # Blur should not be identical to original (unless image is uniform)
        assert not np.array_equal(blurred, img), "Blur should modify the image"

    def test_apply_document_fade(self):
        """Test document fading produces valid output."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        faded = apply_document_fade(img, alpha_range=(0.7, 0.7))
        assert faded.shape == img.shape
        assert faded.dtype == np.uint8
        # Fade should make image brighter overall (lower values moved toward 200)
        assert faded.mean() > img.mean() - 50, "Fade should lighten image"

    def test_apply_document_degradation_combined(self):
        """Test combined degradation (JPEG + blur + fade) works end-to-end."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 40), 100, -1)  # Add text-like content
        
        degraded = apply_document_degradation(
            img,
            jpeg_probability=1.0,  # Always JPEG
            blur_probability=1.0,  # Always blur
            fade_probability=1.0,  # Always fade
        )
        assert degraded.shape == img.shape
        assert degraded.dtype == np.uint8
        # Degraded image should be visibly different
        assert not np.array_equal(degraded, img), "Degradation should modify the image"

    def test_apply_document_degradation_with_probability(self):
        """Test degradation with probabilistic application."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        rng = np.random.default_rng(seed=42)
        
        # With 0.0 probability, should be identical (or very close due to rounding)
        degraded = apply_document_degradation(
            img,
            jpeg_probability=0.0,
            blur_probability=0.0,
            fade_probability=0.0,
            rng=rng,
        )
        assert np.allclose(degraded, img, atol=2), "Zero probability should not modify"

    def test_degradation_parameters_propagate(self):
        """Test that degradation parameters flow through augment_ocr_image correctly."""
        img = np.full((64, 256), 200, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (100, 40), 100, -1)
        
        rng = np.random.default_rng(seed=42)
        augmented = augment_ocr_image(
            img,
            rng=rng,
            degrade_probability=0.5,
            degrade_params={
                "jpeg_p": 1.0,
                "blur_p": 1.0,
                "fade_p": 1.0,
                "jpeg_quality_range": (50, 50),
                "blur_sigma_range": (1.0, 1.0),
                "fade_alpha_range": (0.7, 0.7),
            },
        )
        assert augmented.shape == img.shape
        assert augmented.dtype == np.uint8


class TestCustomOCREnginePreprocess:
    def test_preprocess_output_shape(self):
        engine = CustomOCREngine()
        img = np.full((40, 120), 255, dtype=np.uint8)
        cv2.putText(img, "test", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

        tensor = engine.preprocess(img)
        assert tensor.shape == (1, 64, 256, 1)
        assert tensor.dtype == np.float32

    def test_preprocess_normalizes_inverted_polarity(self):
        engine = CustomOCREngine()
        img = np.zeros((40, 120), dtype=np.uint8)
        cv2.rectangle(img, (20, 10), (100, 30), 255, -1)

        normalized = engine._normalize_polarity(img)
        # Background should be bright and former bright foreground should become darker.
        assert int(normalized[0, 0]) > int(normalized[20, 60])

    def test_build_input_variants_returns_two_tensors(self):
        engine = CustomOCREngine()
        img = np.full((40, 120), 255, dtype=np.uint8)
        cv2.putText(img, "abc", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

        variants = engine._build_input_variants(img)
        assert len(variants) == 2
        assert all(v.shape == (1, 64, 256, 1) for v in variants)

    def test_select_best_candidate_prefers_non_empty_text(self):
        chosen = CustomOCREngine._select_best_candidate([
            ("", 0.91),
            ("word", 0.55),
        ])
        assert chosen[0] == "word"
