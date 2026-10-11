import numpy as np
import pandas as pd
import pytest

from src.data.generate_dataset import generate_dataset
from src.data.validation import DataValidationError, validate_dataframe, validate_inference_frame
from src.features.feature_engineering import (
    ID_COLUMNS, TARGET, WELLNESS_VARS, WORKLOAD_CHANNELS, compute_workload_summaries, model_input_columns, sequences_from_frame,
)


def test_record_count_and_prevalence(full_df):
    assert len(full_df) == 12000
    assert int(full_df[TARGET].sum()) == 1078          # round(0.0898 * 12000): documented exact-count choice
    assert abs(full_df[TARGET].mean() - 0.0898) < 0.0005


def test_all_validation_checks_pass(full_df):
    results = validate_dataframe(full_df)
    failed = [r for r in results if not r["passed"]]
    assert not failed, failed


def test_identifiers_missing_duplicates(full_df):
    assert full_df["observation_id"].is_unique
    assert full_df["player_id"].nunique() == 250
    assert full_df.isna().sum().sum() == 0
    assert not full_df.drop(columns="observation_id").duplicated().any()
    assert (full_df["data_source"] == "SYNTHETIC").all()


def test_no_negative_measurements_and_valid_scales(full_df):
    wl, wb = sequences_from_frame(full_df)
    assert (wl >= 0).all()
    for i, v in enumerate(WELLNESS_VARS):
        lo, hi = (0, 100) if v == "recovery_score" else ((3.5, 10.5) if v == "sleep_hours" else (1, 10))
        assert wb[:, :, i].min() >= lo and wb[:, :, i].max() <= hi


def test_shapes_and_derived_features(full_df):
    wl, wb = sequences_from_frame(full_df)
    assert wl.shape == (12000, 28, 8) and wb.shape == (12000, 7, 7)
    rpe, dur, load = (wl[:, :, WORKLOAD_CHANNELS.index(c)] for c in
                      ("session_rating_of_perceived_exertion", "training_duration_minutes", "session_training_load"))
    assert np.allclose(load, rpe * dur, atol=1e-6)
    summ = compute_workload_summaries(wl)
    assert np.allclose(summ["acute_chronic_workload_ratio"], full_df["acute_chronic_workload_ratio"], atol=1e-3)
    assert np.allclose(summ["acute_workload"], wl[:, -7:, -1].mean(1), atol=1e-3)


def test_feature_target_separation(full_df):
    inputs = model_input_columns()
    assert TARGET not in inputs and not set(inputs) & set(ID_COLUMNS)
    assert not any("prob" in c or "latent" in c or "hidden" in c for c in full_df.columns)   # nothing hidden is saved
    # no single input feature reproduces the label
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(full_df[TARGET], full_df[c]) for c in full_df[inputs].select_dtypes("number").columns[:120]]
    assert max(max(a, 1 - a) for a in aucs) < 0.70


def test_generation_is_reproducible():
    a = generate_dataset(n_players=10, n_weeks=6, seed=3)
    b = generate_dataset(n_players=10, n_weeks=6, seed=3)
    c = generate_dataset(n_players=10, n_weeks=6, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_history_windows_do_not_overlap_the_horizon():
    """Consecutive weekly observations shift the 28-day window by exactly 7 days (no look-ahead)."""
    df = generate_dataset(n_players=3, n_weeks=6, seed=1)
    wl, _ = sequences_from_frame(df[df.player_id == "P001"])
    assert np.allclose(wl[1, :-7], wl[0, 7:])


def test_validator_detects_defects(small_df):
    bad = small_df.copy(); bad.loc[0, "wl_total_distance_km_d3"] = -1
    assert not all(r["passed"] for r in validate_dataframe(bad, expected_records=None, expected_prevalence=None))
    bad = small_df.copy(); bad.loc[1, "observation_id"] = bad.loc[0, "observation_id"]
    assert any(not r["passed"] and "unique" in r["check"] for r in validate_dataframe(bad, None, None))
    bad = small_df.copy(); bad.loc[2, "position"] = "XX"
    assert any(not r["passed"] and "position" in r["check"] for r in validate_dataframe(bad, None, None))
    bad = small_df.copy(); bad.loc[3, TARGET] = 2
    assert any(not r["passed"] and "target" in r["check"] for r in validate_dataframe(bad, None, None))


def test_inference_input_validation(small_df):
    row = small_df.head(1).copy()
    validate_inference_frame(row)
    with pytest.raises(DataValidationError, match="non-negative"):
        r = row.copy(); r["wl_total_distance_km_d5"] = -2.0; validate_inference_frame(r)
    with pytest.raises(DataValidationError, match="position"):
        r = row.copy(); r["position"] = "QB"; validate_inference_frame(r)
    with pytest.raises(DataValidationError, match="Wellness"):
        r = row.copy(); r["wb_fatigue_score_d1"] = 15.0; validate_inference_frame(r)
    with pytest.raises(DataValidationError, match="missing"):
        validate_inference_frame(row.drop(columns=["age"]))
