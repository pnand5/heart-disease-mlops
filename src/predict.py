"""Inference helpers: load the saved pipeline and score patient records."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_model(path: Path = DEFAULT_MODEL_PATH):
    return joblib.load(path)


def predict(records: list[dict], model=None) -> list[dict]:
    """Score records; `confidence` is the probability of the predicted class."""
    model = model or load_model()
    frame = pd.DataFrame(records)[FEATURES]
    p_disease = model.predict_proba(frame)[:, 1]
    results = []
    for p in p_disease:
        label = int(p >= 0.5)
        results.append({
            "prediction": label,
            "diagnosis": "heart disease" if label else "no heart disease",
            "confidence": round(float(p if label else 1 - p), 4),
            "probability_disease": round(float(p), 4),
        })
    return results
