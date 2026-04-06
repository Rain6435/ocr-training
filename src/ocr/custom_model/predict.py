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
        second_pass_threshold: float = 0.72,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.beam_width = beam_width
        self.second_pass_threshold = second_pass_threshold
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

    @staticmethod
    def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
        """Normalize image to black text on white background for CRNN stability."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_count = int(np.count_nonzero(binary == 255))
        black_count = int(np.count_nonzero(binary == 0))

        background_is_white = white_count >= black_count
        return gray if background_is_white else cv2.bitwise_not(gray)

    @staticmethod
    def _crop_foreground(gray: np.ndarray, pad: int = 2) -> np.ndarray:
        """Crop around foreground strokes to reduce empty margins before resize."""
        inv = 255 - gray
        ys, xs = np.where(inv > 20)
        if ys.size == 0 or xs.size == 0:
            return gray

        y1 = max(int(ys.min()) - pad, 0)
        y2 = min(int(ys.max()) + pad + 1, gray.shape[0])
        x1 = max(int(xs.min()) - pad, 0)
        x2 = min(int(xs.max()) + pad + 1, gray.shape[1])
        cropped = gray[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else gray

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, pad, and normalize image for model input."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = self._normalize_polarity(gray)
        gray = self._crop_foreground(gray)

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

    @staticmethod
    def _enhance_variant(gray: np.ndarray) -> np.ndarray:
        """Create a lightly enhanced variant to recover faint strokes."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    @staticmethod
    def _denoise_variant(gray: np.ndarray) -> np.ndarray:
        """Apply light denoising to reduce compression and sensor noise artifacts."""
        denoised = cv2.medianBlur(gray, 3)
        return denoised

    @staticmethod
    def _sharpen_variant(gray: np.ndarray) -> np.ndarray:
        """Apply mild sharpening to improve weak stroke boundaries."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _prepare_base_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare base grayscale image with normalized polarity and tight crop."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        base = self._normalize_polarity(gray)
        base = self._crop_foreground(base)
        return base

    def _build_input_variants(self, image: np.ndarray) -> list[np.ndarray]:
        """Build normalized model-ready input variants for robust decoding."""
        base = self._prepare_base_image(image)

        enhanced = self._enhance_variant(base)

        variants = [self.preprocess(base), self.preprocess(enhanced)]
        return variants

    def _decode_prediction(self, y_pred: np.ndarray) -> tuple[str, float]:
        """Decode logits and return (text, confidence) tuple."""
        if self.beam_width > 1:
            texts = ctc_beam_search_decode(y_pred, beam_width=self.beam_width)
        else:
            texts = ctc_greedy_decode(y_pred)

        confidences = compute_ctc_confidence(y_pred)
        text = texts[0] if texts else ""
        confidence = confidences[0] if confidences else 0.0
        return text, float(confidence)

    def _predict_logits(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run model inference and return logits/probabilities tensor."""
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]["index"])
        return self.model.predict(input_tensor, verbose=0)

    @staticmethod
    def _select_best_candidate(candidates: list[tuple[str, float]]) -> tuple[str, float]:
        """Prefer non-empty decode, then highest confidence among candidates."""
        non_empty = [c for c in candidates if c[0].strip()]
        pool = non_empty if non_empty else candidates
        return max(pool, key=lambda item: item[1])

    @staticmethod
    def _apply_text_quality_penalty(text: str, confidence: float) -> float:
        """Penalize confidence for outputs that look like OCR gibberish."""
        t = text.strip()
        if not t:
            return 0.0

        penalty = 0.0
        chars = list(t)
        alnum_ratio = sum(ch.isalnum() for ch in chars) / max(len(chars), 1)

        # Too little alphanumeric content often indicates noisy decode.
        if len(t) >= 10 and alnum_ratio < 0.45:
            penalty += 0.10

        # Penalize long repeated-character runs.
        max_run = 1
        run = 1
        for i in range(1, len(chars)):
            if chars[i] == chars[i - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run >= 4:
            penalty += 0.12

        # Penalize very low unique-character diversity on long strings.
        if len(t) >= 12:
            diversity = len(set(chars)) / len(chars)
            if diversity < 0.33:
                penalty += 0.08

        return max(0.0, float(confidence) - penalty)

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
        base = self._prepare_base_image(image)
        base_tensor = self.preprocess(base)
        base_text, base_confidence = self._decode_prediction(self._predict_logits(base_tensor))

        needs_second_pass = (not base_text.strip()) or (base_confidence < self.second_pass_threshold)
        if needs_second_pass:
            enhanced = self._enhance_variant(base)
            denoised = self._denoise_variant(base)
            sharpened = self._sharpen_variant(denoised)

            enhanced_text, enhanced_confidence = self._decode_prediction(
                self._predict_logits(self.preprocess(enhanced))
            )
            sharpened_text, sharpened_confidence = self._decode_prediction(
                self._predict_logits(self.preprocess(sharpened))
            )
            best_text, best_confidence = self._select_best_candidate(
                [
                    (base_text, base_confidence),
                    (enhanced_text, enhanced_confidence),
                    (sharpened_text, sharpened_confidence),
                ]
            )
        else:
            best_text, best_confidence = base_text, base_confidence

        best_confidence = self._apply_text_quality_penalty(best_text, best_confidence)

        return {
            "text": best_text,
            "confidence": best_confidence,
            "engine": "custom_crnn",
            "cost": 0.001,
        }
