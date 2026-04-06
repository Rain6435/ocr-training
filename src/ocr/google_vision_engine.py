import os
from typing import Any

import cv2
import numpy as np


class GoogleVisionEngine:
    """Wrapper for Google Cloud Vision OCR (external API baseline)."""

    def __init__(self, language_hints: list[str] | None = None, timeout_s: int = 30):
        from google.cloud import vision

        self.client = vision.ImageAnnotatorClient()
        self.vision = vision
        self.language_hints = language_hints or ["en"]
        self.timeout_s = timeout_s
        self.cost_per_page = float(os.getenv("GOOGLE_VISION_COST_PER_PAGE", "0.015"))

    def _build_image(self, image: np.ndarray) -> Any:
        if len(image.shape) == 2:
            ok, buf = cv2.imencode(".png", image)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".png", rgb)
        if not ok:
            raise RuntimeError("Failed to encode image for Google Vision request")
        return self.vision.Image(content=buf.tobytes())

    def _extract_confidence(self, response: Any) -> float:
        doc = getattr(response, "full_text_annotation", None)
        if not doc or not getattr(doc, "pages", None):
            return 0.0

        confs: list[float] = []
        for page in doc.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        if hasattr(word, "confidence"):
                            confs.append(float(word.confidence))
        if not confs:
            return 0.0
        return float(np.mean(confs))

    def recognize(self, image: np.ndarray) -> dict:
        """
        Run Google Vision OCR on an image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "engine": "google_vision",
                "cost": float,
            }
        """
        request_image = self._build_image(image)
        image_context = self.vision.ImageContext(language_hints=self.language_hints)

        response = self.client.document_text_detection(
            image=request_image,
            image_context=image_context,
            timeout=self.timeout_s,
        )

        if response.error.message:
            raise RuntimeError(response.error.message)

        text = ""
        if response.full_text_annotation and response.full_text_annotation.text:
            text = response.full_text_annotation.text
        elif response.text_annotations:
            text = response.text_annotations[0].description

        return {
            "text": text.strip(),
            "confidence": self._extract_confidence(response),
            "engine": "google_vision",
            "cost": self.cost_per_page,
        }
