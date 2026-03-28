import numpy as np
import pytest

from src.ocr.custom_model.architecture import build_crnn, build_training_model
from src.ocr.custom_model.vocabulary import (
    CHARS, CHAR_TO_IDX, IDX_TO_CHAR, BLANK_IDX, NUM_CLASSES,
    encode_text, decode_indices,
)
from src.ocr.custom_model.ctc_utils import ctc_greedy_decode, compute_ctc_confidence
from src.ocr.custom_model.augmentation import augment_ocr_image


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
