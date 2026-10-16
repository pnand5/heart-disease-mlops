"""Loading and cleaning of the UCI Heart Disease (Cleveland) dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target and impute missing values.

    The original label `num` is 0 (no disease) or 1-4 (disease severity); the
    task is binary so any value > 0 becomes 1. The few missing values (`ca`,
    `thal`) are categorical, so they are filled with the column mode.
    """
    df = df.copy()
    df[TARGET] = (df["num"] > 1).astype(int)
    df = df.drop(columns="num")
    for col in df.columns.drop(TARGET):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df.drop_duplicates().reset_index(drop=True)


def load_clean(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Raw file -> cleaned CSV on disk (and returned)."""
    df = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET].to_numpy(dtype=np.int64)


if __name__ == "__main__":
    cleaned = prepare()
    print(f"Wrote {PROCESSED_PATH} {cleaned.shape}")
    print(cleaned[TARGET].value_counts().to_string())
