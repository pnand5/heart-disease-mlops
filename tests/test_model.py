import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src import data
from src.predict import predict
from tests.conftest import SAMPLE_PATIENT


def test_model_beats_chance_in_cross_validation(clean_df, fitted_model):
    X, y = data.split_xy(clean_df)
    proba = cross_val_predict(fitted_model, X, y, cv=5, method="predict_proba")[:, 1]
    assert roc_auc_score(y, proba) > 0.8


def test_predict_output_contract(fitted_model):
    result = predict([SAMPLE_PATIENT], model=fitted_model)[0]
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_disease"] <= 1.0
    assert result["diagnosis"] in ("heart disease", "no heart disease")


def test_predict_is_deterministic(fitted_model):
    a = predict([SAMPLE_PATIENT], model=fitted_model)
    b = predict([SAMPLE_PATIENT], model=fitted_model)
    assert a == b


def test_saved_model_loads_and_predicts():
    from src.predict import DEFAULT_MODEL_PATH, load_model

    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("models/model.joblib not trained yet")
    result = predict([SAMPLE_PATIENT], model=load_model())[0]
    assert result["prediction"] in (0, 1)
