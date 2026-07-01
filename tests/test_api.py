from fastapi.testclient import TestClient

from src.api import app


def test_api_health_and_models():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    models = client.get("/models")
    assert models.status_code == 200
    assert "efficientnet_b0" in models.json()["supported_cnn_models"]


def test_api_quality_upload(synthetic_retina):
    client = TestClient(app)
    with synthetic_retina.open("rb") as handle:
        response = client.post("/quality", files={"image": ("retina.png", handle, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["retina_visibility_score"] > 0