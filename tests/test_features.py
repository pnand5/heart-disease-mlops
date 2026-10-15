import numpy as np

from src import data
from src.features import build_preprocessor


def test_output_is_numeric_and_finite(clean_df):
    X, _ = data.split_xy(clean_df)
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape[0] == len(X)
    assert np.isfinite(out).all()


def test_numeric_columns_are_standardised(clean_df):
    X, _ = data.split_xy(clean_df)
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    n_num = len(data.NUMERIC_FEATURES)
    assert np.allclose(out[:, :n_num].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out[:, :n_num].std(axis=0), 1, atol=1e-6)


def test_unseen_category_and_missing_value_do_not_crash(clean_df):
    X, _ = data.split_xy(clean_df)
    prep = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["thal"] = 99.0  # category never seen in training
    row["chol"] = np.nan
    out = prep.transform(row)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert np.isfinite(out).all()
