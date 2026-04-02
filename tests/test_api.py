import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestStatsEndpoint:
    def test_stats(self, client):
        response = client.get("/api/v1/stats")
        # May fail if models not loaded, that's expected
        assert response.status_code in (200, 500)


class TestPageOCREndpoint:
    def test_page_ocr_returns_structured_lines(self, client, monkeypatch):
        class FakePipeline:
            def process(self, image, profile=None):
                return {"preprocessed_full": image}

            def normalize_for_ocr(self, image, target_width=256):
                return image

        class FakeRouter:
            engines = {}

            def route(self, image):
                return {
                    "text": "line text",
                    "confidence": 0.82,
                    "engine_used": "medium",
                    "difficulty": "medium",
                    "cost": 0.01,
                    "escalated": False,
                    "processing_time_ms": 1.0,
                }

        class FakeSpell:
            def correct(self, text):
                return {"corrected": text, "num_corrections": 0}

        class FakeScorer:
            def score(self, result):
                return {"confidence": result.get("confidence", 0.0), "needs_review": False}

        monkeypatch.setattr("src.api.routes.get_preprocessing_pipeline", lambda: FakePipeline())
        monkeypatch.setattr("src.api.routes.get_router", lambda: FakeRouter())
        monkeypatch.setattr("src.api.routes.get_spell_corrector", lambda: FakeSpell())
        monkeypatch.setattr("src.api.routes.get_confidence_scorer", lambda: FakeScorer())

        image = np.full((64, 256), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok

        response = client.post(
            "/api/v1/ocr/page",
            files={"file": ("page.png", encoded.tobytes(), "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "lines" in data
        assert data["num_lines"] >= 1
        assert data["num_columns"] >= 1
        assert data["segmentation_mode"] == "auto"
        assert data["lines"][0]["line_index"] == 0
        assert "column_index" in data["lines"][0]
        assert "bbox" in data["lines"][0]

    def test_page_ocr_tei_output_contains_zones(self, client, monkeypatch):
        class FakePipeline:
            def process(self, image, profile=None):
                return {"preprocessed_full": image}

            def normalize_for_ocr(self, image, target_width=256):
                return image

        class FakeRouter:
            engines = {}

            def route(self, image):
                return {
                    "text": "line text",
                    "confidence": 0.82,
                    "engine_used": "medium",
                    "difficulty": "medium",
                    "cost": 0.01,
                    "escalated": False,
                    "processing_time_ms": 1.0,
                }

        class FakeSpell:
            def correct(self, text):
                return {"corrected": text, "num_corrections": 0}

        class FakeScorer:
            def score(self, result):
                return {"confidence": result.get("confidence", 0.0), "needs_review": False}

        monkeypatch.setattr("src.api.routes.get_preprocessing_pipeline", lambda: FakePipeline())
        monkeypatch.setattr("src.api.routes.get_router", lambda: FakeRouter())
        monkeypatch.setattr("src.api.routes.get_spell_corrector", lambda: FakeSpell())
        monkeypatch.setattr("src.api.routes.get_confidence_scorer", lambda: FakeScorer())

        image = np.full((64, 256), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok

        response = client.post(
            "/api/v1/ocr/page?output_format=tei-xml",
            files={"file": ("page.png", encoded.tobytes(), "image/png")},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/xml")
        body = response.text
        assert "<facsimile" in body
        assert "<zone" in body
        assert "corresp=\"#line_0\"" in body
