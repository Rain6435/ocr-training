import numpy as np
import pytesseract
import cv2
import os

# Configure pytesseract to find Tesseract on Windows
# CRITICAL: Set this FIRST before any pytesseract calls
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Default Windows install
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",  # 32-bit install
]

for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.pytesseract_cmd = path
        print(f"[TesseractEngine] Configured pytesseract to use: {path}")
        break

# Also add to PATH environment variable for subprocess calls
if r"C:\Program Files\Tesseract-OCR" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = r"C:\Program Files\Tesseract-OCR" + os.pathsep + os.environ.get("PATH", "")


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
