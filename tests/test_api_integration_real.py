from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_IMAGE = PROJECT_ROOT / "data" / "processed" / "hard_paragraph_test" / "hard_paragraph_condensed.png"


def _build_test_image_bytes() -> bytes:
    if DEFAULT_SAMPLE_IMAGE.exists():
        data = DEFAULT_SAMPLE_IMAGE.read_bytes()
        if data:
            return data

    # Synthetic fallback so the test can run when benchmark assets are unavailable.
    img = np.full((192, 720), 245, dtype=np.uint8)
    cv2.putText(img, "historical text sample", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "line two for OCR", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode synthetic image")
    return encoded.tobytes()


@pytest.mark.integration
def test_page_endpoint_real_components_custom_force():
    model_path = PROJECT_ROOT / "models" / "ocr_custom" / "inference_model.keras"
    if not model_path.exists():
        pytest.skip("Custom inference model not found; skipping real API integration test")

    client = TestClient(app)
    image_bytes = _build_test_image_bytes()

    response = client.post(
        "/api/v1/ocr/page",
        params={
            "profile": "handwritten",
            "force_engine": "custom",
            "segmentation_mode": "projection",
            "output_format": "json",
        },
        files={"file": ("sample.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert isinstance(payload.get("text", ""), str)
    assert isinstance(payload.get("lines", []), list)
    assert int(payload.get("num_lines", 0)) >= 1
    assert 0.0 <= float(payload.get("confidence", 0.0)) <= 1.0

    first = payload["lines"][0]
    assert "bbox" in first
    assert "engine_used" in first
    assert "processing_time_ms" in first
    assert "curved_attempted" in first
    assert "curved_used" in first
    assert "word_mode_attempted" in first
    assert "word_mode_used" in first
    assert "word_count" in first
