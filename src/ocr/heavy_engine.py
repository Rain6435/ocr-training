import numpy as np
import cv2
from PIL import Image


class TrOCREngine:
    """Wrapper for TrOCR-large (hard documents)."""

    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def recognize(self, image: np.ndarray) -> dict:
        """
        Run TrOCR on an image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "engine": "trocr-large",
                "cost": 0.05,
            }
        """
        # Convert to PIL RGB
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Process
        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        generated = self.model.generate(
            pixel_values,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True,
        )

        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]

        # Estimate confidence from generation scores
        import torch
        if generated.scores:
            probs = [torch.softmax(s, dim=-1).max().item() for s in generated.scores]
            confidence = float(np.mean(probs)) if probs else 0.5
        else:
            confidence = 0.5

        return {
            "text": text,
            "confidence": confidence,
            "engine": "trocr-large",
            "cost": 0.05,
        }


class PaddleOCREngine:
    """Wrapper for PaddleOCR (alternative heavy engine)."""

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def recognize(self, image: np.ndarray) -> dict:
        """
        Run PaddleOCR on an image.

        Returns:
            {
                "text": str,
                "confidence": float,
                "engine": "paddleocr",
                "cost": 0.02,
            }
        """
        result = self.ocr.ocr(image, cls=True)

        texts = []
        confidences = []

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                texts.append(text)
                confidences.append(conf)

        full_text = " ".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "engine": "paddleocr",
            "cost": 0.02,
        }
