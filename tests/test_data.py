import numpy as np
import pandas as pd

from src import data


def make_raw(n=10):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 4, size=(n, len(data.COLUMNS))), columns=data.COLUMNS)
    df["age"] = rng.integers(30, 80, size=n)  # keeps rows unique
    df["num"] = [0, 1, 2, 3, 4] * (n // 5)
    return df.astype(float)


def test_clean_binarises_target():
    cleaned = data.clean(make_raw())
    assert set(cleaned[data.TARGET]) <= {0, 1}
    assert "num" not in cleaned.columns
    # num of 0 -> 0, anything above 0 -> 1
    assert cleaned[data.TARGET].sum() == 8


def test_clean_imputes_missing_with_mode():
    raw = make_raw()
    raw.loc[0, "ca"] = np.nan
    raw.loc[1, "thal"] = np.nan
    cleaned = data.clean(raw)
    assert cleaned.isna().sum().sum() == 0


def test_load_raw_treats_question_mark_as_nan(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = data.load_raw(path)
    assert df.shape == (1, 14)
    assert np.isnan(df.loc[0, "ca"])


def test_prepare_writes_clean_csv(tmp_path):
    raw_path = tmp_path / "raw.data"
    make_raw().to_csv(raw_path, header=False, index=False)
    out = tmp_path / "out" / "clean.csv"
    df = data.prepare(raw_path, out)
    assert out.exists()
    assert list(pd.read_csv(out).columns) == list(df.columns)


def test_split_xy_shapes():
    cleaned = data.clean(make_raw())
    X, y = data.split_xy(cleaned)
    assert list(X.columns) == data.FEATURES
    assert len(X) == len(y)
