import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import data
from src.features import build_preprocessor

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


@pytest.fixture(scope="session")
def clean_df():
    return data.load_clean()


@pytest.fixture(scope="session")
def fitted_model(clean_df):
    X, y = data.split_xy(clean_df)
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
    return pipe.fit(X, y)
