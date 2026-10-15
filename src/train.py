"""Train, tune, evaluate and track candidate models; save the best pipeline.

Usage: python -m src.train
"""
import argparse
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import data  # noqa: E402
from src.features import build_preprocessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
SEED = 42
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 4, 8],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
    ),
}


def test_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def log_plots(model, X, y, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, cmap="Blues")
        ax.set_title(f"{name} - confusion matrix (test)")
        fig.tight_layout()
        fig.savefig(Path(tmp) / "confusion_matrix.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(model, X, y, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"{name} - ROC curve (test)")
        fig.tight_layout()
        fig.savefig(Path(tmp) / "roc_curve.png", dpi=120)
        plt.close(fig)
        mlflow.log_artifacts(tmp, artifact_path="plots")


def run_candidate(name, estimator, grid, X_train, X_test, y_train, y_test):
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    with mlflow.start_run(run_name=name) as run:
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Unbiased CV estimate of the tuned configuration, on the training split only.
        cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING)
        cv_metrics = {f"cv_{m}": float(cv_res[f"test_{m}"].mean()) for m in CV_SCORING}
        cv_metrics.update({f"cv_{m}_std": float(cv_res[f"test_{m}"].std()) for m in CV_SCORING})
        held_out = {f"test_{k}": v for k, v in test_metrics(best, X_test, y_test).items()}

        mlflow.log_param("model", name)
        mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_param(
            "search_grid", json.dumps({k: [str(x) for x in v] for k, v in grid.items()})
        )
        mlflow.log_metrics({**cv_metrics, **held_out})
        log_plots(best, X_test, y_test, name)
        mlflow.sklearn.log_model(
            best,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(X_train, best.predict(X_train)),
            input_example=X_train.head(3),
        )
        return {
            "name": name,
            "run_id": run.info.run_id,
            "model": best,
            "params": search.best_params_,
            **cv_metrics,
            **held_out,
        }


def main(experiment: str, tracking_uri: str) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    df = data.prepare() if not data.PROCESSED_PATH.exists() else data.load_clean()
    X, y = data.split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    results = [
        run_candidate(n, est, grid, X_train, X_test, y_train, y_test)
        for n, (est, grid) in CANDIDATES.items()
    ]
    # Select on CV ROC-AUC (training data only) so the test set stays untouched by selection.
    best = max(results, key=lambda r: r["cv_roc_auc"])
    mlflow.MlflowClient().set_tag(best["run_id"], "selected", "true")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    summary = {
        "selected_model": best["name"],
        "selected_run_id": best["run_id"],
        "models": {
            r["name"]: {k: v for k, v in r.items() if k not in ("model", "run_id", "name")}
            for r in results
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(summary, indent=2, default=str))

    print(f"{'model':<22}{'cv_auc':>8}{'test_auc':>10}{'test_acc':>10}{'test_rec':>10}")
    for r in results:
        mark = " <- selected" if r is best else ""
        print(f"{r['name']:<22}{r['cv_roc_auc']:>8.3f}{r['test_roc_auc']:>10.3f}"
              f"{r['test_accuracy']:>10.3f}{r['test_recall']:>10.3f}{mark}")
    print(f"Saved {MODEL_PATH}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="heart-disease-classification")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()
    main(args.experiment, args.tracking_uri)
