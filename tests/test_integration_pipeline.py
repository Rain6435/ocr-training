import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.api.dependencies import (
    get_confidence_scorer,
    get_preprocessing_pipeline,
    get_router,
    get_spell_corrector,
)
from src.ocr.custom_model.predict import CustomOCREngine
from src.ocr.page_pipeline import process_page


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_IMAGE = PROJECT_ROOT / "data" / "processed" / "hard_paragraph_test" / "hard_paragraph_condensed.png"


def _load_sample_image() -> np.ndarray:
    if DEFAULT_SAMPLE_IMAGE.exists():
        img = cv2.imread(str(DEFAULT_SAMPLE_IMAGE), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

    # Synthetic fallback for CI/local environments where benchmark assets are absent.
    img = np.full((192, 640), 245, dtype=np.uint8)
    cv2.putText(img, "historical text sample", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 40, 2, cv2.LINE_AA)
    cv2.putText(img, "line two for OCR", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 40, 2, cv2.LINE_AA)
    return img


def test_custom_engine_real_inference_smoke():
    model_path = PROJECT_ROOT / "models" / "ocr_custom" / "inference_model.keras"
    if not model_path.exists():
        pytest.skip("Custom inference model not found; skipping integration smoke test")

    img = _load_sample_image()
    engine = CustomOCREngine(model_path=str(model_path))
    result = engine.recognize(img)

    assert isinstance(result.get("text", ""), str)
    assert 0.0 <= float(result.get("confidence", 0.0)) <= 1.0
    assert result.get("engine") == "custom_crnn"


def test_page_pipeline_real_components_smoke():
    model_path = PROJECT_ROOT / "models" / "ocr_custom" / "inference_model.keras"
    if not model_path.exists():
        pytest.skip("Custom inference model not found; skipping integration smoke test")

    img = _load_sample_image()

    result = process_page(
        image=img,
        preprocessing_pipeline=get_preprocessing_pipeline(),
        ocr_router=get_router(),
        spell_corrector=get_spell_corrector(),
        confidence_scorer=get_confidence_scorer(),
        profile="handwritten",
        force_engine="custom",
        segmentation_mode="projection",
    )

    assert "lines" in result
    assert isinstance(result["lines"], list)
    assert result["num_lines"] >= 1
    assert "confidence" in result

    first = result["lines"][0]
    assert "curved_attempted" in first
    assert "curved_used" in first
    assert "word_mode_attempted" in first
    assert "word_mode_used" in first
    assert "word_count" in first
