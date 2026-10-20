"""Build the assignment report (PDF + editable DOCX) from one content definition.

    python report/build_report.py

Screenshots are read from ../screenshots. A missing screenshot becomes a clearly
marked placeholder; drop the file in with the expected name and re-run.
PDF needs Google Chrome (headless); DOCX is written with the standard library only.
Placeholders written as [[like this]] are highlighted - replace them in this file.
"""
import html
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHOTS = ROOT / "screenshots"
FIGS = HERE / "figures"
STEM = "Heart_Disease_MLOps_Report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO_URL = "https://github.com/pnand5/heart-disease-mlops"
MISSING: list[str] = []


# ----------------------------------------------------------------- helpers
def img_crop(path: Path, name: str, right: float = 1.0) -> Path | None:
    """Trim empty space from the right of a screenshot; the original file is left as is."""
    if not path.exists():
        MISSING.append(path.name)
        return None
    im = Image.open(path).convert("RGB")
    out = FIGS / name
    im.crop((0, 0, int(im.width * right), im.height)).save(out)
    return out


def img(path: Path) -> Path | None:
    if path.exists():
        return path
    MISSING.append(path.name)
    return None


# ----------------------------------------------------------------- content
def blocks() -> list:
    B: list = []
    add = B.append

    add(("title", "Heart Disease Risk Prediction",
         "An end-to-end MLOps pipeline: experiment tracking, CI/CD, containerisation, Kubernetes and monitoring"))
    add(("meta", [
        ("Course", "Machine Learning Operations (MLOps), AIMLCZG523 - Assignment 1"),
        ("Student", "Prasanna Nandalal    BITS ID: 2025AF05091"),
        ("Code repository", REPO_URL),
        ("Deployed API", "http://localhost:8080 (Docker Desktop Kubernetes, local; access steps in section 9)"),
    ]))

    # 1 ---------------------------------------------------------------
    add(("h1", "1. Overview"))
    add(("p", "The goal is a reproducible, automated and observable machine-learning service that predicts whether a "
              "patient has heart disease from routine clinical measurements. I used the UCI Heart Disease (Cleveland) "
              "dataset: 303 patients, 13 predictors and a target that I binarised into *no disease* versus *disease*. "
              "The final model is a **Logistic Regression** inside a single scikit-learn pipeline, served by a FastAPI "
              "container on a local Kubernetes cluster, with request logging, Prometheus metrics and a Grafana dashboard."))
    add(("p", "Beyond the baseline I asked four questions of my own: does dropping weak features or adding "
              "clinically motivated ones help (section 5.1), is 0.5 the right decision threshold for a screening tool, and "
              "can the reported probability be trusted (section 5.2). The honest answer to the first is *no, not "
              "robustly*, and I kept the simpler all-feature model."))
    add(("table", ["Model", "CV ROC-AUC (5-fold)", "Test ROC-AUC", "Test accuracy", "Test recall"], [
        ["**Logistic Regression (selected)**", "**0.902**", "0.966", "0.885", "0.929"],
        ["Random Forest", "0.901", "0.944", "0.836", "0.821"],
        ["Gradient Boosting", "0.875", "0.951", "0.885", "0.929"],
    ], [34, 22, 16, 16, 12]))
    add(("p", "Logistic Regression and Random Forest are statistically tied: a more reliable repeated 5x5 cross-validation "
              "gives 0.892 and 0.893. Individual folds vary by about 0.04 and the test split holds only 61 patients, so "
              "the model was chosen for simplicity and interpretability, not for a meaningful accuracy gap."))

    # 2 ---------------------------------------------------------------
    add(("h1", "2. Architecture"))
    add(("fig", img(FIGS / "architecture.png"),
         "End-to-end architecture: data and model development, CI/CD on GitHub Actions, and containerised serving "
         "with monitoring.", 80))
    add(("p", "Code flows from the developer to GitHub, where Actions lint, test, train and smoke-test a container on every "
              "push. The same Dockerfile produces the image that Kubernetes runs; Prometheus scrapes its `/metrics` "
              "endpoint and Grafana visualises them."))

    # 3 ---------------------------------------------------------------
    add(("h1", "3. Setup and installation"))
    add(("p", "Requirements: Python 3.10; Docker Desktop (with Kubernetes enabled) for the container, cluster and "
              "monitoring steps. Everything below runs from a clean clone using only `requirements.txt`."))
    add(("code",
         f"git clone {REPO_URL}.git && cd heart-disease-mlops\n"
         "python3 -m venv .venv && source .venv/bin/activate\n"
         "pip install -r requirements.txt\n\n"
         "python scripts/download_data.py   # UCI file -> data/raw/\n"
         "python -m src.data                # clean -> data/processed/heart_clean.csv\n"
         "python -m src.train               # tune, evaluate, track (MLflow), save models/model.joblib\n"
         "python -m src.experiments         # feature-selection / threshold / calibration experiments\n"
         "pytest -v && ruff check .         # 37 tests + lint\n"
         "make mlflow-ui                    # tracking UI on http://localhost:5001"))
    add(("p", "The Makefile wraps these (`make setup data train test lint`). Serving, Kubernetes and monitoring commands "
              "are in sections 9-10 and in the repository README."))

    # 4 ---------------------------------------------------------------
    add(("h1", "4. Data acquisition and exploratory analysis"))
    add(("p", "`scripts/download_data.py` fetches `processed.cleveland.data` from the UCI repository. The file has no header "
              "and marks missing values with `?`; `src/data.py` names the 14 columns, parses `?` as missing, and "
              "binarises the label (`num` 0 -> *no disease*, 1-4 -> *disease*). I binarised because the task is binary and "
              "the disease grades are too sparse to learn separately (55, 36, 35 and 13 patients against 164 healthy). Only six "
              "values are missing (`ca`: 4, `thal`: 2). Both columns are categorical, and dropping six rows would discard "
              "2% of a very small dataset, so I filled them with the column mode. The cleaned file has 303 rows and no "
              "missing values and is committed to the repository."))
    add(("figrow", [
        (img(FIGS / "class_balance.png"), "Class balance: 164 no disease (54%), 139 disease (46%)."),
        (img(FIGS / "correlation_heatmap.png"), "Correlation heatmap including the target."),
    ], [38, 62]))
    add(("ul", [
        "**Balanced target** (54% / 46%), so accuracy is meaningful and no resampling is needed; recall and ROC-AUC are "
        "still reported because missing a sick patient is the costly error.",
        "**Strongest linear signals** with the target: `thal` (0.52), `ca` (0.46), `exang` (0.43), `oldpeak` (0.42), "
        "`thalach` (-0.42: a higher maximum heart rate means lower risk) and `cp` (0.41). `fbs` (0.03) and `chol` (0.09) "
        "are nearly uninformative on their own - which is what prompted the feature experiments in section 5.1.",
        "`oldpeak` and `slope` are correlated (0.58): a mild collinearity that matters for linear-model coefficients, not "
        "for tree models.",
        "Numeric features sit on different scales, so scaling is required for Logistic Regression. Codes such as `cp`, "
        "`thal` and `slope` are nominal labels, not quantities (`thal` takes the values 3, 6 and 7), so they are one-hot "
        "encoded rather than treated as ordered numbers.",
    ]))
    add(("fig", img(FIGS / "numeric_histograms.png"),
         "Distributions of the numeric features, split by target (full notebook: notebooks/01_eda.ipynb).", 56))

    # 5 ---------------------------------------------------------------
    add(("h1", "5. Feature engineering and model development"))
    add(("p", "Preprocessing lives in one `ColumnTransformer` (`src/features.py`) that is part of the model pipeline, so "
              "during cross-validation it is re-fitted on each training fold and never sees validation data. Numeric "
              "columns: median imputation then standardisation. Categorical columns: most-frequent imputation then one-hot "
              "encoding with `handle_unknown=\"ignore\"`. The imputers are redundant on the cleaned CSV but make the "
              "served model robust to a missing or unseen value."))
    add(("p", "**Protocol.** A stratified 80/20 split (242 training, 61 test patients, seed 42). Each candidate is tuned "
              "with `GridSearchCV` (stratified 5-fold, scoring ROC-AUC) on the training split only; the tuned configuration "
              "is then re-scored with 5-fold cross-validation on accuracy, precision, recall, F1 and ROC-AUC. The model is "
              "**selected on cross-validated ROC-AUC**, and the test split is used once, for reporting."))
    add(("table", ["Model", "Grid searched", "Best parameters"], [
        ["Logistic Regression", "C in {0.01, 0.1, 1, 10}", "C = 1"],
        ["Random Forest", "n_estimators {100, 300} x max_depth {None, 4, 8} x min_samples_leaf {1, 3}",
         "100 trees, max_depth 4, min_samples_leaf 1"],
        ["Gradient Boosting", "n_estimators {100, 200} x learning_rate {0.05, 0.1} x max_depth {2, 3}",
         "100 trees, learning_rate 0.05, max_depth 3"],
    ], [20, 48, 32]))
    add(("table", ["Model", "CV accuracy", "CV precision", "CV recall", "CV F1", "CV ROC-AUC",
                   "Test acc.", "Test recall", "Test AUC"], [
        ["Logistic Regression", "0.847", "0.877", "0.783", "0.825", "0.902", "0.885", "0.929", "0.966"],
        ["Random Forest", "0.826", "0.844", "0.774", "0.802", "0.901", "0.836", "0.821", "0.944"],
        ["Gradient Boosting", "0.793", "0.771", "0.783", "0.775", "0.875", "0.885", "0.929", "0.951"],
    ], [20, 10, 10, 10, 9, 11, 10, 10, 10]))
    add(("p", "**Why Logistic Regression.** It is marginally ahead on the logged cross-validation (0.902 vs 0.901) but "
              "marginally behind under repeated 5x5 CV (0.892 vs 0.893): a tie. I chose it because it is the simplest, its "
              "coefficients can be inspected, its probabilities turned out to be well calibrated (section 5.2) and the "
              "artifact is tiny - not because it is more accurate. The held-out numbers are higher than the cross-validated "
              "ones for every model; with 61 patients one error moves accuracy by about 1.6 points, so the cross-validated "
              "figures (242 patients, many folds) are the more trustworthy estimate."))
    add(("figrow", [
        (img(FIGS / "cv_comparison.png"), "Cross-validated ROC-AUC, single 5-fold run (error bars: std of 5 folds)."),
        (img(FIGS / "roc_comparison.png"), "ROC curves on the held-out test split."),
    ], [50, 50]))
    add(("figrow", [
        (img(FIGS / "confusion_matrix.png"), "Confusion matrix on the test split (selected model)."),
        (img(FIGS / "lr_coefficients.png"), "Largest Logistic Regression coefficients."),
    ], [38, 62]))
    add(("p", "The largest coefficients agree with the EDA: no coloured vessels (`ca_0`) is the strongest protective signal, "
              "while asymptomatic chest pain (`cp_4`) and a reversible thalassemia defect (`thal_7`) raise risk the most. "
              "Typical angina (`cp_1`) is counter-intuitively associated with *lower* predicted risk in this dataset - a "
              "pattern in the data, not clinical advice."))

    add(("h2", "5.1 Do other feature sets help? Selection and engineering experiments"))
    add(("p", "The EDA suggested `fbs` and `chol` carry almost no signal, and clinical knowledge suggested features the "
              "raw columns do not express directly: the share of the age-predicted maximum heart rate reached "
              "(`thalach / (220 - age)`), \"any vessel narrowed\" (`ca > 0`), any ST depression, and conventional "
              "hypertension and cholesterol cut-offs. I tested ten feature-set variants on both Logistic Regression and "
              "Random Forest (`src/experiments.py`). All comparisons use repeated stratified 5-fold CV (5 repeats, 25 folds) "
              "on the training split with the same folds for every variant, so differences are paired, and every selector "
              "and scaler sits inside the pipeline so nothing is fitted on validation data. The baseline (all 13 features) "
              "scores 0.892 (LR) and 0.893 (RF)."))
    add(("table", ["Variant", "What changes", "LR change in AUC", "RF change in AUC", "Verdict"], [
        ["drop_fbs_chol", "drop `fbs`, `chol`", "+0.006", "+0.004", "free simplification (3 fewer columns)"],
        ["drop_4_weakest", "also drop `trestbps`, `restecg`", "+0.005", "+0.008", "within noise"],
        ["filter_kbest14", "ANOVA filter keeps 14 of 28 columns", "+0.013", "+0.003", "best of 10, not adopted"],
        ["l1_select", "L1 regularisation picks 15 columns", "+0.005", "-0.010", "inconsistent across models"],
        ["rfe14", "recursive elimination to 14 columns", "-0.008", "-0.006", "hurts"],
        ["eng_hr_reserve", "+ heart-rate reserve %", "0.000", "+0.001", "no gain"],
        ["eng_flags", "+ 4 clinical threshold flags", "0.000", "+0.006", "helps only Random Forest"],
        ["eng_all", "+ reserve % and the 4 flags", "0.000", "+0.002", "no gain"],
        ["eng_all_drop_weak", "eng_all, then drop `fbs`, `chol`", "+0.003", "+0.006", "no gain"],
    ], [17, 31, 13, 13, 26]))
    add(("fig", img(FIGS / "feature_experiments.png"),
         "Change in cross-validated ROC-AUC against the baseline (+/- 2 SE). Grey band: gaps under 0.01, which I treat as noise "
         "(fold-to-fold spread is about 0.04).", 86))
    add(("p", "**Conclusion: keep all 13 original features.** Every variant lands within about 0.013 AUC of the baseline. "
              "The biggest gain (+0.013, the 14-column filter on Logistic Regression) is the best of ten comparisons, which "
              "overstates its benefit, and it does not transfer to Random Forest. Automatic selection can even hurt (RFE, "
              "L1 on Random Forest) because a 242-patient training set has little to spare. The clinical features add "
              "nothing for Logistic Regression, whose one-hot `ca`/`thal` and linear `age`/`thalach` already carry the "
              "same information. The one robust finding is that `fbs` and `chol` can be dropped at no cost, which would "
              "simplify data collection if the model were ever redeployed."))

    add(("h2", "5.2 Decision threshold and probability calibration"))
    add(("p", "The API labels a patient *disease* when the probability reaches 0.5, a convention rather than a clinical "
              "decision. Using out-of-fold probabilities on the training split, the default threshold gives recall 0.78 and "
              "precision 0.87, so about one sick patient in five is missed. Lowering the threshold to **0.25** (the highest "
              "value that keeps recall at 90%) gives recall 0.90 and precision 0.69. On the held-out split (28 diseased, "
              "33 healthy) that finds all 28 sick patients (0 missed instead of 2) at the cost of 10 false alarms instead "
              "of 5 - an illustration, given only 61 patients."))
    add(("fig", img(FIGS / "threshold_calibration.png"),
         "Left: out-of-fold precision and recall against the threshold. Right: reliability diagram of the predicted "
         "probabilities (five equal-size bins).", 86))
    add(("p", "The probabilities are well calibrated: Brier score 0.121 against 0.248 for always predicting the base "
              "rate, and an expected calibration error of 0.030. So `confidence` and `probability_disease` can be read as risk "
              "estimates. I left the deployed cut-off at 0.5 because the operating point is a policy decision; the API "
              "returns `probability_disease`, so a screening client can apply 0.25 without a redeploy."))

    # 6 ---------------------------------------------------------------
    add(("h1", "6. Experiment tracking with MLflow"))
    add(("p", "`src/train.py` wraps every candidate in an MLflow run (SQLite backend) inside the experiment "
              "`heart-disease-classification`. Each run logs: the tuned hyper-parameters and the searched grid, 5-fold CV "
              "metrics (mean and standard deviation for five metrics), held-out test metrics, the confusion-matrix and ROC "
              "plots as artifacts, and the fitted pipeline as an MLflow model with an inferred signature and an input "
              "example. The selected run carries the tag `selected=true`. Because training is deterministic, "
              "`notebooks/02_training.ipynb` refits each model from its logged parameters and reproduces the logged test "
              "AUC exactly (0.9665, 0.9437, 0.9513). `python -m src.experiments` logs a second experiment, "
              "`feature-and-threshold-experiments`: one run per variant and model (20) with the paired change in AUC, plus one "
              "run for the threshold and calibration analysis."))
    add(("fig", img(SHOTS / "01_mlflow_experiment_runs.png"),
         "MLflow experiment: one run per model, each with a logged model artifact.", 70))
    add(("figrow", [
        (img(SHOTS / "02_mlflow_compare_runs.png"), "Run comparison: parameters, metrics and the selected tag."),
        (img(SHOTS / "04_mlflow_run_artifacts.png"), "A logged plot artifact (ROC curve)."),
    ], [46, 54]))

    # 7 ---------------------------------------------------------------
    add(("h1", "7. Model packaging and reproducibility"))
    add(("ul", [
        "**One artifact for train and serve.** `models/model.joblib` is the *whole* pipeline (imputation, scaling, "
        "one-hot encoding, classifier). Inference takes raw patient fields; there is no separate preprocessing code that "
        "could drift from training. The same pipeline is also stored as an MLflow model.",
        "**Pinned environment.** `requirements.txt` pins every package; `requirements-serve.txt` holds only the runtime "
        "subset the container needs. Random seeds are fixed (42).",
        "**Verified from a clean setup.** The CI runner starts from an empty machine and installs only from "
        "`requirements.txt` before downloading the data, testing and training, and I also repeated the whole pipeline "
        "from a fresh clone and a new virtual environment locally with identical results.",
    ]))

    # 8 ---------------------------------------------------------------
    add(("h1", "8. CI/CD pipeline and automated testing"))
    add(("p", "The suite has **37 pytest tests**: data cleaning and parsing (5), the preprocessing transformers (3: finite "
              "output, standardised numeric columns, tolerance of unseen categories and missing values), the model (4: "
              "cross-validated AUC above 0.8, output contract, determinism, saved model loads), the API (8: health, "
              "prediction contract, four invalid-value cases, a missing field, metrics exposure) and the experiments (17: "
              "feature derivations, every variant fits and predicts, selectors keep the expected columns, threshold and "
              "calibration properties)."))
    add(("p", "`.github/workflows/ci.yml` runs on every push and on demand. Job **build-test-train**: install dependencies, "
              "`ruff` lint, download the data, `pytest`, train the models with MLflow, then upload the JUnit test report "
              "and the model, metrics and MLflow runs as workflow artifacts. Job **docker** (only if the first passes): "
              "build the image, start the container, and call `/health` and `/predict`. Steps are ordered cheapest-first "
              "and each depends on the previous one, so a failure stops the pipeline and marks the run red."))
    add(("table", ["Run", "Commit", "Branch", "Purpose", "Result"], [
        ["#3", "6bd7a67", "mlops-pipeline", "Finished branch", "**Passed** - lint, tests, training, Docker build and smoke test"],
        ["#4", "1e90cec", "ci-failure-demo", "Deliberate regression: target threshold changed to `> 1`",
         "**Failed at Unit tests**; training and Docker jobs skipped"],
        ["#5", "1916dbc", "ci-failure-demo", "Deliberate lint error (unused import)",
         "**Failed at Lint**; tests, training, Docker skipped"],
        ["#8", "adaf477", "main", "Merge of pull request #1", "**Passed**"],
    ], [7, 11, 17, 30, 35]))
    add(("p", "Runs #4 and #5 were pushed on a throwaway branch to show that the pipeline really fails on code and test "
              "errors; the failing assertion (`assert 6 == 8`) was reproduced locally first."))
    add(("figrow", [
        (img(SHOTS / "10_github_actions_run_success.png"), "Run #3: both jobs pass."),
        (img(SHOTS / "09_github_actions_runs_list.png"), "Workflow runs: green and deliberately red."),
    ], [50, 50]))
    add(("figrow", [
        (img_crop(SHOTS / "11_github_actions_failed_unit_tests.png", "ci_failed_unit_tests.png", 0.74),
         "Run #4 stops at Unit tests: `assert 6 == 8`."),
        (img_crop(SHOTS / "12_github_actions_failed_lint.png", "ci_failed_lint.png", 0.50),
         "Run #5 stops at Lint: 7 errors."),
    ], [60, 40]))

    # 9 ---------------------------------------------------------------
    add(("h1", "9. Containerisation and production deployment"))
    add(("p", "**Image.** The Dockerfile starts from `python:3.10-slim`, installs only `requirements-serve.txt` (no MLflow "
              "or notebook tooling), copies just `src/`, `api/` and the model, runs as a non-root user and defines a "
              "`HEALTHCHECK`. The image is 473 MB."))
    add(("p", "**API.** `POST /predict` takes the 13 clinical fields as JSON and returns the prediction and the "
              "confidence (the probability of the predicted class). Inputs are validated (ranges and allowed codes); "
              "invalid or missing fields return HTTP 422 instead of a guess. `GET /health` is used by the probes and "
              "`GET /metrics` exposes Prometheus metrics."))
    add(("fig", img(SHOTS / "14_docker_build_and_run.png"),
         "Docker build, isolated run, health check, a prediction and a rejected invalid request.", 82))
    add(("p", "**Kubernetes.** The target is the local Docker Desktop cluster (context `docker-desktop`), chosen over a "
              "paid cloud so the deployment is free to reproduce. `k8s/` holds a Namespace, a Deployment (2 replicas, "
              "readiness and liveness probes on `/health`, CPU/memory requests and limits) and a `LoadBalancer` Service "
              "mapping port 8080 to the container's 8000. The image uses `imagePullPolicy: Never` because it is built "
              "locally; on GKE/EKS/AKS it would be pushed to a registry and the policy changed."))
    add(("p", "**Access instructions (local testing).** There is no public URL; the API is reachable at "
              "`http://localhost:8080` on a machine with Docker Desktop and Kubernetes enabled:"))
    add(("code",
         "docker build -t heart-disease-api:latest .                  # image is built locally\n"
         "kubectl --context docker-desktop apply -f k8s/\n"
         "kubectl --context docker-desktop -n heart-disease get deploy,pods,svc\n"
         "curl http://localhost:8080/health\n"
         "curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d @scripts/sample_patient.json\n"
         "# interactive docs: http://localhost:8080/docs"))
    add(("fig", img(SHOTS / "13_kubectl_pods_services.png"),
         "Two replicas Running, the LoadBalancer service on port 8080, and a live prediction through it.", 82))
    add(("fig", img(SHOTS / "08_kubernetes_api_swagger_localhost8080.png"),
         "API documentation served by the Kubernetes deployment at localhost:8080.", 48))

    # 10 --------------------------------------------------------------
    add(("h1", "10. Monitoring and logging"))
    add(("p", "Every request is logged to stdout with method, path, status and latency, and every prediction with its "
              "input and result (visible with `kubectl logs`). The service exposes four Prometheus series: "
              "`api_requests_total{method,path,status}`, `api_request_latency_seconds` (histogram), "
              "`model_predictions_total{diagnosis}` and `model_prediction_errors_total`. `docker-compose.yml` starts the API, "
              "Prometheus (scraping every 5 s) and Grafana, which loads a provisioned dashboard: request rate by path "
              "and status, p50/p95 latency, predictions by outcome, and 4xx/5xx counts. I counted predictions by outcome "
              "because a sudden shift in the share of *disease* predictions is the cheapest available signal of input drift."))
    add(("p", "`scripts/generate_traffic.py` replays real dataset rows (with about 12% deliberately invalid requests) to "
              "exercise the dashboard. Under that local load the p95 `/predict` latency was about 24 ms. The 4xx counter "
              "rises with the invalid requests while 5xx stays at zero."))
    add(("fig", img(SHOTS / "07_grafana_dashboard.png"), "Grafana dashboard during generated traffic.", 82))
    add(("fig", img(SHOTS / "06_prometheus_targets.png"), "Prometheus target `api:8000/metrics` is UP.", 62))

    # 11 --------------------------------------------------------------
    add(("h1", "11. Design decisions"))
    add(("table", ["Decision", "Alternative considered", "Why this way"], [
        ["Binary target (`num` > 0)", "Keep the 5 disease grades", "Task is binary; grades 1-4 have 55/36/35/13 patients"],
        ["Mode-impute 6 missing values", "Drop those rows", "Both columns categorical; keeps 2% more data"],
        ["One-hot encode `cp`, `thal`, `slope`, ...", "Integer codes as numbers", "They are labels, not quantities"],
        ["Tune and select on CV; test set used once", "Choose by test score", "61 test patients would overfit the choice"],
        ["Logistic Regression", "Random Forest / Gradient Boosting", "Tied on accuracy; simpler, interpretable, calibrated"],
        ["Keep all 13 features", "Selected / engineered sets", "Ten variants tried; none robustly better (5.1)"],
        ["API cut-off 0.5, probability exposed", "Fixed screening cut-off 0.25", "Operating point is a policy choice (5.2)"],
        ["Whole pipeline in one artifact", "Model only plus separate preprocessing", "Inference cannot drift from training"],
        ["FastAPI with typed schema", "Flask", "Validation gives clear 422s; docs for free"],
        ["CI order: lint, test, train, then Docker", "One combined job", "Cheapest checks first; Docker job only if they pass"],
        ["Serving image with runtime deps only", "Reuse `requirements.txt`", "Smaller image, fewer packages to patch"],
        ["Docker Desktop Kubernetes", "Minikube or a cloud cluster", "Free, no account, same manifests would run on GKE"],
        ["Prometheus + Grafana, own dashboard", "Logs only", "Request, latency, outcome and error series; outcome share hints at drift"],
    ], [29, 27, 44]))

    # 12 --------------------------------------------------------------
    add(("h1", "12. Limitations, future work and reflection"))
    add(("ul", [
        "**Small data.** 303 patients give wide uncertainty; fold-to-fold AUC varies by about 0.04. This is a coursework "
        "model on a historical dataset and not a medical device.",
        "**Local deployment.** There is no public URL; the model file is committed so images build reproducibly, where a "
        "production system would load it from a model registry.",
        "**Not covered:** authentication and TLS, autoscaling, input-drift monitoring beyond the outcome counter, and "
        "automated retraining and continuous deployment (CI stops at a verified image).",
    ]))
    add(("p", "**Reflection.** [[Write 3-5 sentences in your own words: what surprised you (for example that nothing beat the "
              "all-features baseline), what you would do differently with more data or time, and what you learned about "
              "MLOps from building this.]]"))
    return number_figures(B)


def number_figures(B: list) -> list:
    """Prefix captions with 'Figure N - ' in order of appearance."""
    n, out = 0, []
    for b in B:
        if b[0] == "fig":
            n += 1
            out.append(("fig", b[1], f"Figure {n} - {b[2]}", b[3]))
        elif b[0] == "figrow":
            cells = []
            for p, c in b[1]:
                n += 1
                cells.append((p, f"Figure {n} - {c}"))
            out.append(("figrow", cells, b[2]))
        else:
            out.append(b)
    return out


# ----------------------------------------------------------------- inline markup
TOKEN = re.compile(r"\*\*(.+?)\*\*|`(.+?)`|\[\[(.+?)\]\]|\*(.+?)\*")


def runs(text: str):
    """Yield (text, style) where style is one of '', 'b', 'code', 'hl', 'i'."""
    pos = 0
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            yield text[pos:m.start()], ""
        if m.group(1) is not None:
            yield m.group(1), "b"
        elif m.group(2) is not None:
            yield m.group(2), "code"
        elif m.group(3) is not None:
            yield m.group(3), "hl"
        else:
            yield m.group(4), "i"
        pos = m.end()
    if pos < len(text):
        yield text[pos:], ""


def h(text: str) -> str:
    tag = {"b": "strong", "code": "code", "hl": 'mark', "i": "em"}
    return "".join(f"<{tag[s]}>{html.escape(t)}</{tag[s]}>" if s else html.escape(t) for t, s in runs(text))


# ----------------------------------------------------------------- HTML
CSS = """
@page { size: A4; margin: 14mm 14mm 15mm 14mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 9.5pt; line-height: 1.35; color: #1c2833; }
h1 { font-size: 13.5pt; color: #1a5276; border-bottom: 1.5px solid #1a5276; padding-bottom: 2px; margin: 15px 0 7px;
     break-after: avoid; }
h2 { font-size: 11pt; color: #1a5276; margin: 11px 0 4px; break-after: avoid; }
h1.title { font-size: 22pt; border: 0; margin: 0 0 2px; color: #154360; }
.sub { font-size: 11pt; color: #566573; margin-bottom: 8px; }
table.meta td { border: 0; padding: 1px 10px 1px 0; font-size: 9.6pt; }
p { margin: 4px 0 6px; text-align: left; }
ul { margin: 3px 0 6px 0; padding-left: 18px; } li { margin-bottom: 2px; text-align: left; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 8.6pt; background: #f2f4f4; padding: 0 2px; border-radius: 2px; }
pre { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 7.4pt; line-height: 1.3; background: #f4f6f7;
      border: 1px solid #d5d8dc; border-radius: 3px; padding: 5px 7px; margin: 5px 0 7px; white-space: pre-wrap;
      word-break: break-all; }
mark { background: #fcf3cf; padding: 0 2px; }
table.data { border-collapse: separate; border-spacing: 1px; background: #ccd1d1; width: 100%; margin: 5px 0 8px;
             font-size: 8.6pt; }
table.data th { background: #1a5276; color: white; text-align: left; padding: 3px 5px; }
table.data td { background: #ffffff; padding: 3px 5px; vertical-align: top; }
table.data tr { break-inside: avoid; }
figure { margin: 6px 0 8px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e5e8e8; }
figcaption { font-size: 8.2pt; color: #566573; margin-top: 2px; font-style: italic; }
table.row { width: 100%; border-collapse: separate; border-spacing: 0; margin: 4px 0 8px; break-inside: avoid; }
table.row td { vertical-align: top; text-align: center; padding: 0 3px; }
table.row img { max-width: 100%; border: 1px solid #e5e8e8; }
.ph { border: 1.5px dashed #c0392b; background: #fdedec; color: #922b21; padding: 14px 6px; font-size: 8.4pt; }
"""


def fig_html(path, caption, width):
    if path is None:
        inner = f'<div class="ph">Screenshot to insert: <code>{html.escape(caption.split(" - ")[0])}</code> (see submission checklist)</div>'
    else:
        inner = f'<img src="{html.escape(rel(path))}" style="width:{width}%">'
    return f"<figure>{inner}<figcaption>{h(caption)}</figcaption></figure>"


def rel(path: Path) -> str:
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def to_html(B) -> str:
    out = ['<!doctype html><html><head><meta charset="utf-8"><title>Heart Disease MLOps Report</title>'
           f"<style>{CSS}</style></head><body>"]
    for b in B:
        k = b[0]
        if k == "title":
            out.append(f'<h1 class="title">{h(b[1])}</h1><div class="sub">{h(b[2])}</div>')
        elif k == "meta":
            rows = "".join(f"<tr><td><b>{h(a)}</b></td><td>{h(v)}</td></tr>" for a, v in b[1])
            out.append(f'<table class="meta">{rows}</table>')
        elif k == "h1":
            out.append(f"<h1>{h(b[1])}</h1>")
        elif k == "h2":
            out.append(f"<h2>{h(b[1])}</h2>")
        elif k == "p":
            out.append(f"<p>{h(b[1])}</p>")
        elif k == "ul":
            out.append("<ul>" + "".join(f"<li>{h(x)}</li>" for x in b[1]) + "</ul>")
        elif k == "code":
            out.append(f"<pre>{html.escape(b[1])}</pre>")
        elif k == "table":
            head = "".join(f"<th>{h(c)}</th>" for c in b[1])
            body = "".join("<tr>" + "".join(f"<td>{h(c)}</td>" for c in r) + "</tr>" for r in b[2])
            out.append(f'<table class="data"><tr>{head}</tr>{body}</table>')
        elif k == "fig":
            out.append(fig_html(b[1], b[2], b[3]))
        elif k == "figrow":
            cells = "".join(f'<td style="width:{w}%">{fig_html(p, c, 100)}</td>' for (p, c), w in zip(b[1], b[2]))
            out.append(f'<table class="row"><tr>{cells}</tr></table>')
    out.append("</body></html>")
    return "\n".join(out)


# ----------------------------------------------------------------- DOCX (stdlib only)
W_NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')
CONTENT_W = 9906          # A4 text width in twips (margins 1000)
EMU_PER_TWIP = 635


class Docx:
    def __init__(self):
        self.body: list[str] = []
        self.media: list[tuple[str, Path]] = []
        self.n_img = 0

    # run/paragraph builders
    @staticmethod
    def run(text, style="", size=None):
        rpr = ""
        if style == "code":
            rpr += '<w:rFonts w:ascii="Menlo" w:hAnsi="Menlo" w:cs="Menlo"/>'
        if style == "b":
            rpr += "<w:b/>"
        if style == "i":
            rpr += "<w:i/>"
        if style == "code":
            rpr += '<w:sz w:val="17"/>'
        elif size:
            rpr += f'<w:sz w:val="{size}"/>'
        if style == "hl":
            rpr += '<w:highlight w:val="yellow"/>'
        return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

    def rich(self, text, size=None):
        return "".join(self.run(t, s, size) for t, s in runs(text))

    def para(self, inner, style=None, jc=None, extra=""):
        ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + extra + (f'<w:jc w:val="{jc}"/>' if jc else "")
        return f"<w:p><w:pPr>{ppr}</w:pPr>{inner}</w:p>"

    def image(self, path: Path, frac: float):
        im = Image.open(path)
        cx = int(frac * CONTENT_W * EMU_PER_TWIP)
        cy = int(cx * im.height / im.width)
        max_cy = int(9.2 * 914400 * 0.9)          # keep a figure within one page
        if cy > max_cy:
            cx, cy = int(cx * max_cy / cy), max_cy
        self.n_img += 1
        rid, name = f"rIdImg{self.n_img}", f"image{self.n_img}.png"
        self.media.append((name, path))
        return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                f'<wp:docPr id="{self.n_img}" name="{name}"/><a:graphic><a:graphicData uri='
                f'"http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" '
                f'name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch>'
                f'<a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" '
                f'cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
                f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')

    def figure_xml(self, path, caption, frac):
        if path is None:
            ph = self.run("[Screenshot to insert: " + caption.split(" - ")[0] + "]", "hl")
            return self.para(ph, jc="center") + self.para(self.rich(caption, 16), "Caption", "center")
        return self.para(self.image(path, frac), jc="center") + self.para(self.rich(caption, 16), "Caption", "center")

    def table(self, rows, widths, header=True, borders=True, font=17):
        total = CONTENT_W
        grid = "".join(f'<w:gridCol w:w="{int(total * w / sum(widths))}"/>' for w in widths)
        bd = ('<w:tblBorders>' + "".join(f'<w:{s} w:val="single" w:sz="4" w:color="B2BABB"/>'
                                          for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
              + '</w:tblBorders>') if borders else ""
        xml = [f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>{bd}<w:tblCellMar><w:left w:w="70" w:type="dxa"/>'
               f'<w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
        for i, row in enumerate(rows):
            xml.append("<w:tr>")
            for c, w in zip(row, widths):
                tcw = int(total * w / sum(widths))
                shade = '<w:shd w:val="clear" w:color="auto" w:fill="1A5276"/>' if header and i == 0 else ""
                if isinstance(c, str):
                    txt = (self.run(c, "b", font) if header and i == 0 else self.rich(c, font))
                    if header and i == 0:
                        txt = txt.replace("<w:b/>", '<w:b/><w:color w:val="FFFFFF"/>')
                    content = self.para(txt, extra='<w:spacing w:before="20" w:after="20"/>')
                else:
                    content = c[0]                       # pre-built paragraphs (figure cell)
                xml.append(f'<w:tc><w:tcPr><w:tcW w:w="{tcw}" w:type="dxa"/>{shade}</w:tcPr>{content}</w:tc>')
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        return "".join(xml) + self.para("", extra='<w:spacing w:before="0" w:after="40"/>')

    def add(self, b):
        k = b[0]
        if k == "title":
            self.body += [self.para(self.run(b[1], "b", 44), "Title"), self.para(self.rich(b[2], 22), "Subtitle")]
        elif k == "meta":
            self.body.append(self.table([[f"**{a}**", v] for a, v in b[1]], [16, 84], header=False, borders=False, font=19))
        elif k == "h1":
            self.body.append(self.para(self.run(b[1]), "Heading1"))
        elif k == "h2":
            self.body.append(self.para(self.run(b[1]), "Heading2"))
        elif k == "p":
            self.body.append(self.para(self.rich(b[1]), jc="both"))
        elif k == "ul":
            for x in b[1]:
                self.body.append(self.para(self.run("•\t") + self.rich(x), jc="both",
                                           extra='<w:ind w:left="360" w:hanging="260"/>'))
        elif k == "code":
            for ln in b[1].split("\n"):
                self.body.append(self.para(
                    f'<w:r><w:rPr><w:rFonts w:ascii="Menlo" w:hAnsi="Menlo" w:cs="Menlo"/><w:sz w:val="14"/></w:rPr>'
                    f'<w:t xml:space="preserve">{escape(ln) or " "}</w:t></w:r>',
                    extra='<w:shd w:val="clear" w:color="auto" w:fill="F4F6F7"/><w:spacing w:before="0" w:after="0" w:line="220" w:lineRule="exact"/>'))
            self.body.append(self.para("", extra='<w:spacing w:before="0" w:after="60"/>'))
        elif k == "table":
            self.body.append(self.table([b[1]] + b[2], b[3]))
        elif k == "fig":
            self.body.append(self.figure_xml(b[1], b[2], b[3] / 100))
        elif k == "figrow":
            cells = []
            for (p, c), w in zip(b[1], b[2]):
                frac = 0.94 * w / 100
                cells.append((self.figure_xml(p, c, frac),))
            self.body.append(self.table([cells], b[2], header=False, borders=False))

    def save(self, path: Path):
        styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="20"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:before="40" w:after="80" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="40"/></w:pPr><w:rPr><w:color w:val="154360"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="566573"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:pBdr><w:bottom w:val="single" w:sz="8" w:space="1" w:color="1A5276"/></w:pBdr><w:spacing w:before="280" w:after="100"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="1A5276"/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="60"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="1A5276"/><w:sz w:val="23"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="20" w:after="140"/></w:pPr><w:rPr><w:i/><w:color w:val="566573"/><w:sz w:val="16"/></w:rPr></w:style>
</w:styles>"""
        doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W_NS}><w:body>'
               + "".join(self.body)
               + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1000" w:right="1000" w:bottom="1000" '
                 'w:left="1000" w:header="500" w:footer="500" w:gutter="0"/></w:sectPr></w:body></w:document>')
        rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                + "".join(f'<Relationship Id="rIdImg{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{n}"/>'
                          for i, (n, _) in enumerate(self.media, 1)) + "</Relationships>")
        ctypes = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                  '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                  '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
                  '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                  '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                     '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", ctypes)
            z.writestr("_rels/.rels", root_rels)
            z.writestr("word/document.xml", doc)
            z.writestr("word/styles.xml", styles)
            z.writestr("word/_rels/document.xml.rels", rels)
            for name, src in self.media:
                z.write(src, f"word/media/{name}")


# ----------------------------------------------------------------- main
def _texts(x):
    """Yield every string nested anywhere inside a content block."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _texts(y)


def main() -> int:
    B = blocks()
    html_path = HERE / f"{STEM}.html"
    html_path.write_text(to_html(B), encoding="utf-8")

    d = Docx()
    for b in B:
        d.add(b)
    d.save(HERE / f"{STEM}.docx")
    print(f"wrote {STEM}.docx")

    pdf_path = HERE / f"{STEM}.pdf"
    if Path(CHROME).exists():
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                        f"--print-to-pdf={pdf_path}", html_path.as_uri()], capture_output=True, timeout=180)
        try:
            from pypdf import PdfReader
            print(f"wrote {pdf_path.name} ({len(PdfReader(str(pdf_path)).pages)} pages)")
        except ImportError:
            print(f"wrote {pdf_path.name}")
    else:
        print("Chrome not found - PDF skipped (open the .html and print to PDF).")
    # the HTML is only an intermediate; keep the folder tidy
    html_path.unlink(missing_ok=True)

    left = sorted({m.group(1) for t in _texts(B) for m in re.finditer(r"\[\[(.+?)\]\]", t)})
    if left:
        print("\nPlaceholders still to fill in (highlighted yellow in the report):")
        for t in left:
            print("  -", t[:90])
    missing = sorted(set(MISSING))
    if missing:
        print("\nMissing screenshots (placeholders were inserted):")
        for m in missing:
            print("  -", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
