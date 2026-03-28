import numpy as np
import cv2
import tensorflow as tf

from src.ocr.custom_model.ctc_utils import (
    ctc_greedy_decode,
    ctc_beam_search_decode,
    compute_ctc_confidence,
)


class CustomOCREngine:
    """Wrapper for the custom CRNN-CTC model (medium documents)."""

    def __init__(
        self,
        model_path: str = "models/ocr_custom/inference_model.keras",
        use_tflite: bool = False,
        tflite_path: str = "models/ocr_tflite/ocr_model.tflite",
        img_height: int = 64,
        img_width: int = 256,
        beam_width: int = 10,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.beam_width = beam_width
        self.use_tflite = use_tflite

        self.model_path = model_path
        self.tflite_path = tflite_path
        self.model = None
        self.interpreter = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return
        if self.use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(self.model_path)
            dummy = np.zeros((1, self.img_height, self.img_width, 1), dtype=np.float32)
            self.model.predict(dummy, verbose=0)
        self._loaded = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, pad, and normalize image for model input."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        scale = self.img_height / h
        new_w = min(int(w * scale), self.img_width)
        resized = cv2.resize(gray, (new_w, self.img_height), interpolation=cv2.INTER_AREA)

        # Pad to target width
        if new_w < self.img_width:
            pad = np.full((self.img_height, self.img_width - new_w), 255, dtype=np.uint8)
            resized = np.concatenate([resized, pad], axis=1)
        else:
            resized = resized[:, :self.img_width]

        # Normalize and add batch + channel dims
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, self.img_height, self.img_width, 1)

    def recognize(self, image: np.ndarray) -> dict:
        """
        Run OCR on a single image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "engine": "custom_crnn",
                "cost": 0.001,
            }
        """
        self._ensure_loaded()
        input_tensor = self.preprocess(image)

        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            y_pred = self.interpreter.get_tensor(self.output_details[0]["index"])
        else:
            y_pred = self.model.predict(input_tensor, verbose=0)

        # Decode
        if self.beam_width > 1:
            texts = ctc_beam_search_decode(y_pred, beam_width=self.beam_width)
        else:
            texts = ctc_greedy_decode(y_pred)

        confidences = compute_ctc_confidence(y_pred)

        return {
            "text": texts[0] if texts else "",
            "confidence": confidences[0] if confidences else 0.0,
            "engine": "custom_crnn",
            "cost": 0.001,
        }
