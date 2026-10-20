import numpy as np
import pandas as pd
import pytest

from src import data
from src import experiments as ex


@pytest.fixture(scope="module")
def split():
    return ex.split_train_test()


def test_clinical_features_values_and_no_mutation():
    df = pd.DataFrame(
        {"age": [60.0], "thalach": [120.0], "ca": [2.0], "oldpeak": [0.0],
         "trestbps": [140.0], "chol": [239.0]}
    )
    before = df.copy()
    out = ex.add_clinical_features(df)
    assert out.loc[0, "hr_reserve_pct"] == pytest.approx(120 / (220 - 60))
    assert out.loc[0, "any_vessel"] == 1
    assert out.loc[0, "st_depression"] == 0
    assert out.loc[0, "high_bp"] == 1          # threshold is inclusive: >= 140
    assert out.loc[0, "high_chol"] == 0        # 239 < 240
    pd.testing.assert_frame_equal(df, before)  # input untouched


def test_clinical_features_tolerate_missing_vessel_count():
    df = pd.DataFrame(
        {"age": [50.0], "thalach": [150.0], "ca": [np.nan], "oldpeak": [1.0],
         "trestbps": [120.0], "chol": [200.0]}
    )
    assert ex.add_clinical_features(df).loc[0, "any_vessel"] == 0


@pytest.mark.parametrize("variant", ex.VARIANTS, ids=lambda v: v.name)
def test_every_variant_fits_and_predicts_probabilities(variant, split):
    X_train, X_test, y_train, _ = split
    pipe = ex.build_pipeline(variant, ex.models()["logistic_regression"]).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_feature_counts_reflect_the_variant(split):
    X_train, _, y_train, _ = split
    by_name = {v.name: v for v in ex.VARIANTS}
    n_all, _ = ex.feature_counts(by_name["baseline_all13"], X_train, y_train)
    n_dropped, _ = ex.feature_counts(by_name["drop_fbs_chol"], X_train, y_train)
    _, n_selected = ex.feature_counts(by_name["filter_kbest14"], X_train, y_train)
    assert n_dropped < n_all
    assert n_selected == ex.K_SELECTED


def test_baseline_variant_matches_production_feature_set():
    assert ex.BASELINE.drop == () and not ex.BASELINE.add_numeric and not ex.BASELINE.add_flags
    assert ex.BASELINE.selector is None
    assert len(data.FEATURES) == 13


def test_run_feature_experiments_reports_paired_deltas(split, monkeypatch):
    X_train, _, y_train, _ = split
    # baseline + one variant keeps this fast
    monkeypatch.setattr(ex, "VARIANTS", ex.VARIANTS[:2])
    res = ex.run_feature_experiments(X_train, y_train, n_repeats=2)
    assert set(res["model"]) == {"logistic_regression", "random_forest"}
    base = res[res["variant"] == "baseline_all13"]
    assert (base["delta_vs_baseline"] == 0).all()
    assert res["cv_auc"].between(0.5, 1.0).all()


def test_calibration_error_is_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.0, 1.0], 50)
    y = np.repeat([0, 1], 50)
    ece, table = ex.expected_calibration_error(y, p, bins=2)
    assert ece == pytest.approx(0.0)
    assert list(table["n"]) == [50, 50]


def test_threshold_analysis_meets_target_recall_and_is_calibrated(split):
    X_train, X_test, y_train, y_test = split
    tc = ex.threshold_and_calibration(X_train, y_train, X_test, y_test, target_recall=0.90)
    curve = tc["curve"]
    row = curve[curve["threshold"] == tc["screening_threshold"]].iloc[0]
    assert row["recall"] >= 0.90
    # a screening threshold favours recall over precision
    assert tc["screening_threshold"] <= 0.5
    # a lower threshold flags at least as many patients and never loses recall
    assert curve["recall"].is_monotonic_decreasing
    assert tc["brier"] < tc["brier_prevalence_only"]  # better than always predicting the base rate
    out = tc["test_outcomes"]["screening"]
    assert out["tn"] + out["fp"] + out["fn"] + out["tp"] == len(y_test)
