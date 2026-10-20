import joblib
import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from tests.conftest import SAMPLE_PATIENT


@pytest.fixture()
def client(fitted_model, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)
    monkeypatch.setattr(app_module, "MODEL_PATH", path)
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence(client):
    resp = client.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


@pytest.mark.parametrize("field,value", [("age", -5), ("sex", 3), ("thal", 5), ("oldpeak", 99)])
def test_predict_rejects_invalid_values(client, field, value):
    resp = client.post("/predict", json={**SAMPLE_PATIENT, field: value})
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in SAMPLE_PATIENT.items() if k != "chol"}
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
