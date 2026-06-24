"""
Test src methods
Run from project root: pytest -v
"""

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "blood_glucose": 140,
    "physical_activity": 20,
    "diet": 1,
    "medication_adherence": 0,
    "stress_level": 2,
    "sleep_hours": 6,
    "hydration_level": 1,
    "bmi": 31.5
}

def test_valid_input_returns_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in ("Low Risk", "Moderate Risk", "High Risk")
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_version" in body

def test_invalid_input_missing_field_returns_422():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["medication_adherence"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_edge_case_extreme_value_does_not_crash():
    edge_payload = {
        "blood_glucose": 1e6,
        "physical_activity": 1e6,
        "diet": 1e6,
        "medication_adherence": 1e6,
        "stress_level": 1e6,
        "sleep_hours": 1e6,
        "hydration_level": 1e6,
        "bmi": 1e6
    }
    response = client.post("/predict", json=edge_payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0