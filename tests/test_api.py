from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_root():
    r = client.get("/")
    assert r.status_code == 200

def test_model_info():
    r = client.get("/api/v1/model-info")
    assert r.status_code == 200
    assert "model" in r.json()