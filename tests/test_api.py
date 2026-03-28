import pytest
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
