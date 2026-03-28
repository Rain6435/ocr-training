import numpy as np
import pytesseract
import cv2


class TesseractEngine:
    """Wrapper for Tesseract OCR (easy documents)."""

    def __init__(self, lang: str = "eng", config: str = "--oem 3 --psm 6"):
        self.lang = lang
        self.config = config

    def recognize(self, image: np.ndarray) -> dict:
        """
        Run Tesseract OCR on an image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "engine": "tesseract",
                "cost": 0.0,
            }
        """
        # Ensure proper format for pytesseract
        if len(image.shape) == 2:
            img = image
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get per-word data with confidences
        data = pytesseract.image_to_data(
            img, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
        )

        # Extract text and compute mean confidence
        words = []
        confidences = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            conf = int(data["conf"][i])
            if text and conf >= 0:
                words.append(text)
                confidences.append(conf)

        full_text = " ".join(words)
        avg_confidence = float(np.mean(confidences)) / 100.0 if confidences else 0.0

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "engine": "tesseract",
            "cost": 0.0,
        }
