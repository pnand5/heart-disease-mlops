"""Experiments beyond the baseline model: feature selection, feature engineering,
decision threshold and probability calibration.

Everything here uses the TRAINING split only (the same split as src/train.py). Selectors,
imputers and scalers sit inside the cross-validated pipeline, so nothing is fitted on
validation folds. The test split is used once, to illustrate the consequence of the chosen
threshold.

Usage: python -m src.experiments
(logs to the MLflow experiment `feature-and-threshold-experiments`)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import data

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ENGINEERED_NUMERIC = ("hr_reserve_pct",)
ENGINEERED_FLAGS = ("any_vessel", "st_depression", "high_bp", "high_chol")
K_SELECTED = 14  # about half of the 28 encoded columns, for the filter and RFE variants


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-motivated features derived from the raw columns (stateless, so safe inside CV).

    hr_reserve_pct  share of the age-predicted maximum heart rate (220 - age) reached
                    in the exercise test
    any_vessel      at least one major vessel coloured by fluoroscopy (ca > 0)
    st_depression   any exercise-induced ST depression (oldpeak > 0)
    high_bp         resting systolic pressure >= 140 mm Hg (hypertension threshold)
    high_chol       serum cholesterol >= 240 mg/dl (high threshold)
    """
    out = df.copy()
    out["hr_reserve_pct"] = out["thalach"] / (220 - out["age"])
    out["any_vessel"] = (out["ca"] > 0).astype(int)
    out["st_depression"] = (out["oldpeak"] > 0).astype(int)
    out["high_bp"] = (out["trestbps"] >= 140).astype(int)
    out["high_chol"] = (out["chol"] >= 240).astype(int)
    return out


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    drop: tuple = ()
    add_numeric: tuple = ()
    add_flags: tuple = ()
    selector: str | None = None


VARIANTS = [
    Variant("baseline_all13", "all 13 original features (the production model)"),
    Variant(
        "drop_fbs_chol",
        "drop fbs and chol (EDA correlation with target < 0.1)",
        drop=("fbs", "chol"),
    ),
    Variant(
        "drop_4_weakest",
        "also drop trestbps and restecg (correlation < 0.2)",
        drop=("fbs", "chol", "trestbps", "restecg"),
    ),
    Variant("filter_kbest14", "ANOVA F-test keeps the 14 best encoded columns", selector="kbest"),
    Variant("l1_select", "L1-regularised logistic regression picks the columns", selector="l1"),
    Variant("rfe14", "recursive feature elimination down to 14 columns", selector="rfe"),
    Variant("eng_hr_reserve", "baseline + heart-rate reserve %", add_numeric=("hr_reserve_pct",)),
    Variant("eng_flags", "baseline + 4 clinical threshold flags", add_flags=ENGINEERED_FLAGS),
    Variant(
        "eng_all",
        "baseline + heart-rate reserve + 4 flags",
        add_numeric=ENGINEERED_NUMERIC,
        add_flags=ENGINEERED_FLAGS,
    ),
    Variant(
        "eng_all_drop_weak",
        "eng_all, then drop fbs and chol",
        drop=("fbs", "chol"),
        add_numeric=ENGINEERED_NUMERIC,
        add_flags=ENGINEERED_FLAGS,
    ),
]
BASELINE = VARIANTS[0]


def models() -> dict:
    """Production hyper-parameters from src/train.py, so variants are compared on equal footing."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=1, random_state=SEED
        ),
    }


def build_pipeline(variant: Variant, model) -> Pipeline:
    numeric = [c for c in data.NUMERIC_FEATURES if c not in variant.drop] + list(
        variant.add_numeric
    )
    categorical = [c for c in data.CATEGORICAL_FEATURES if c not in variant.drop] + list(
        variant.add_flags
    )
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    steps = [("features", FunctionTransformer(add_clinical_features)), ("prep", prep)]
    if variant.selector == "kbest":
        steps.append(("select", SelectKBest(f_classif, k=K_SELECTED)))
    elif variant.selector == "l1":
        steps.append(
            (
                "select",
                SelectFromModel(
                    LogisticRegression(penalty="l1", solver="liblinear", C=0.3, random_state=SEED)
                ),
            )
        )
    elif variant.selector == "rfe":
        steps.append(
            (
                "select",
                RFE(
                    LogisticRegression(max_iter=2000, random_state=SEED),
                    n_features_to_select=K_SELECTED,
                ),
            )
        )
    steps.append(("clf", clone(model)))
    return Pipeline(steps)


def feature_counts(variant: Variant, X: pd.DataFrame, y) -> tuple[int, int]:
    """(encoded columns before selection, columns that reach the classifier)."""
    pipe = build_pipeline(variant, LogisticRegression(max_iter=2000)).fit(X, y)
    n_in = len(pipe.named_steps["prep"].get_feature_names_out())
    sel = pipe.named_steps.get("select")
    return n_in, int(sel.get_support().sum()) if sel is not None else n_in


def split_train_test():
    X, y = data.split_xy(data.load_clean())
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def run_feature_experiments(X_train, y_train, n_repeats: int = 5) -> pd.DataFrame:
    """Repeated stratified 5-fold CV ROC-AUC per variant and model, paired with the baseline."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows, baseline_by_repeat = [], {}
    for model_name, model in models().items():
        for variant in VARIANTS:
            scores = cross_val_score(
                build_pipeline(variant, model),
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            per_repeat = scores.reshape(n_repeats, 5).mean(axis=1)  # folds come grouped by repeat
            if variant is BASELINE:
                baseline_by_repeat[model_name] = per_repeat
            delta = per_repeat - baseline_by_repeat[model_name]
            n_in, n_used = feature_counts(variant, X_train, y_train)
            rows.append(
                {
                    "model": model_name,
                    "variant": variant.name,
                    "description": variant.description,
                    "cv_auc": scores.mean(),
                    "fold_std": scores.std(ddof=1),
                    "delta_vs_baseline": delta.mean(),
                    "delta_se": delta.std(ddof=1) / np.sqrt(n_repeats)
                    if variant is not BASELINE
                    else 0.0,
                    "encoded_columns": n_in,
                    "columns_used": n_used,
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, bins: int = 5):
    """Equal-frequency bins -> (ECE, per-bin table of mean predicted vs observed rate)."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    table = pd.DataFrame(
        {
            "mean_predicted": [p[c].mean() for c in chunks],
            "observed_rate": [y[c].mean() for c in chunks],
            "n": [len(c) for c in chunks],
        }
    )
    ece = float(
        (table["n"] * (table["mean_predicted"] - table["observed_rate"]).abs()).sum() / len(p)
    )
    return ece, table


def threshold_and_calibration(
    X_train, y_train, X_test, y_test, target_recall: float = 0.90
) -> dict:
    """Out-of-fold probabilities (training split) -> threshold trade-off and calibration."""
    pipe = build_pipeline(BASELINE, models()["logistic_regression"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
        pred = oof >= t
        tp = int((pred & (y_train == 1)).sum())
        fp = int((pred & (y_train == 0)).sum())
        fn = int((~pred & (y_train == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn),
                "flagged_share": pred.mean(),
            }
        )
    curve = pd.DataFrame(rows)
    ok = curve[curve["recall"] >= target_recall]
    screening_t = float(ok["threshold"].max()) if len(ok) else 0.5

    fitted = pipe.fit(X_train, y_train)
    p_test = fitted.predict_proba(X_test)[:, 1]
    outcomes = {}
    for label, t in (("default_0.5", 0.5), ("screening", screening_t)):
        tn, fp, fn, tp = confusion_matrix(y_test, p_test >= t).ravel()
        outcomes[label] = {
            "threshold": t,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    ece, bins = expected_calibration_error(y_train, oof)
    return {
        "curve": curve,
        "screening_threshold": screening_t,
        "target_recall": target_recall,
        "test_outcomes": outcomes,
        "brier": float(brier_score_loss(y_train, oof)),
        "brier_prevalence_only": float(
            brier_score_loss(y_train, np.full(len(y_train), y_train.mean()))
        ),
        "ece": ece,
        "calibration_bins": bins,
        "oof": oof,
    }


def main(tracking_uri: str, n_repeats: int) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("feature-and-threshold-experiments")
    X_train, X_test, y_train, y_test = split_train_test()

    results = run_feature_experiments(X_train, y_train, n_repeats)
    for r in results.itertuples():
        with mlflow.start_run(run_name=f"{r.model}:{r.variant}"):
            mlflow.log_params(
                {
                    "model": r.model,
                    "variant": r.variant,
                    "description": r.description,
                    "encoded_columns": r.encoded_columns,
                    "columns_used": r.columns_used,
                    "cv": f"repeated stratified 5-fold x {n_repeats}",
                }
            )
            mlflow.log_metrics(
                {
                    "cv_roc_auc": r.cv_auc,
                    "fold_std": r.fold_std,
                    "delta_vs_baseline": r.delta_vs_baseline,
                    "delta_se": r.delta_se,
                }
            )
    out = ROOT / "artifacts" / "feature_experiments.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)

    tc = threshold_and_calibration(X_train, y_train, X_test, y_test)
    with mlflow.start_run(run_name="threshold_and_calibration"):
        mlflow.log_params(
            {"target_recall": tc["target_recall"], "screening_threshold": tc["screening_threshold"]}
        )
        mlflow.log_metrics(
            {
                "brier_oof": tc["brier"],
                "brier_prevalence_only": tc["brier_prevalence_only"],
                "ece_oof": tc["ece"],
            }
        )
        mlflow.log_artifact(str(out))

    show = results.pivot(index="variant", columns="model", values="cv_auc").reindex(
        [v.name for v in VARIANTS]
    )
    print(show.round(4).to_string())
    print(
        f"screening threshold (OOF recall >= {tc['target_recall']:.2f}): "
        f"{tc['screening_threshold']:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    main(args.tracking_uri, args.repeats)
