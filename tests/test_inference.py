import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import risk_category
from src.data.validation import DataValidationError
from src.evaluation.metrics import compute_metrics, select_threshold
from src.explainability.shap_explainer import (
    answer_questions, explain_single, integrated_gradients, modality_from_table, permutation_modality_importance,
)
from src.inference import ArtifactError, InjuryRiskPredictor
from src.models.baselines import feature_matrix
from src.models.multimodal_model import to_tensors
from src.reporting import build_report, report_to_csv, report_to_pdf


@pytest.fixture(scope="module")
def predictor(trained):
    return InjuryRiskPredictor(trained[0])


@pytest.fixture(scope="module")
def sample(small_df):
    return small_df.iloc[[5, 100, 300]].reset_index(drop=True)


def test_every_model_predicts_valid_probabilities(predictor, sample):
    assert set(predictor.available_models) == {"logistic_regression", "random_forest", "xgboost", "mlp", "workload_only",
                                               "wellness_only", "static_only", "multimodal_dl"}
    for name in predictor.available_models:
        out = predictor.predict(sample, name)
        assert len(out) == 3 and out["probability"].between(0, 1).all()
        assert (out["model"] == name).all()


def test_thresholding_and_categories(predictor, sample, trained):
    cfg = trained[0]
    out = predictor.predict(sample)
    thr = predictor.threshold()
    assert (out["threshold"] == thr).all()
    assert (out["predicted_label"] == (out["probability"] >= thr).astype(int)).all()
    assert list(out["risk_category"]) == [risk_category(p, cfg) for p in out["probability"]]
    hi = predictor.predict(sample, threshold=0.0); lo = predictor.predict(sample, threshold=1.0)
    assert hi["predicted_label"].eq(1).all() and lo["predicted_label"].eq(0).all()


def test_inference_is_deterministic_and_real(predictor, sample):
    a, b = predictor.predict(sample), predictor.predict(sample)
    assert np.allclose(a["probability"], b["probability"])
    assert a["probability"].nunique() > 1                      # not a constant / hard-coded output
    worse = sample.copy()
    for k in range(1, 8):
        worse[f"wb_fatigue_score_d{k}"] = 9.0; worse[f"wb_sleep_hours_d{k}"] = 4.0
    assert not np.allclose(predictor.predict(worse)["probability"], a["probability"])


def test_inference_matches_saved_test_predictions(predictor, small_df, trained):
    cfg = trained[0]
    from src.config import resolve_path
    preds = pd.read_csv(resolve_path(cfg, "predictions_dir") / "multimodal_dl_test_predictions.csv")
    rows = small_df[small_df["observation_id"].isin(preds["observation_id"].head(10))]
    out = predictor.predict(rows)
    ref = preds.set_index("observation_id").loc[out["observation_id"], "prob"].to_numpy()
    assert np.allclose(out["probability"], ref, atol=1e-5)


def test_invalid_input_and_missing_models(predictor, sample, small_cfg, tmp_path):
    bad = sample.copy(); bad["position"] = "QB"
    with pytest.raises(DataValidationError):
        predictor.predict(bad)
    with pytest.raises(ValueError, match="Unknown model"):
        predictor.predict(sample, "does_not_exist")
    cfg = json.loads(json.dumps(small_cfg)); cfg["paths"]["models_dir"] = str(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="Train first"):
        InjuryRiskPredictor(cfg)


def test_corrupted_artifacts_raise_clear_errors(trained, tmp_path, sample):
    cfg = json.loads(json.dumps(trained[0]))
    src = tmp_path / "models"; shutil.copytree(cfg["paths"]["models_dir"], src)
    cfg["paths"]["models_dir"] = str(src)
    (src / "xgboost.joblib").write_bytes(b"corrupt")
    p = InjuryRiskPredictor(cfg)
    with pytest.raises(ArtifactError, match="corrupted"):
        p.predict(sample, "xgboost")
    (src / "calibrators.joblib").write_bytes(b"corrupt")
    with pytest.raises(ArtifactError):
        InjuryRiskPredictor(cfg)


def test_saved_metrics_are_computed_from_saved_predictions(trained):
    from src.config import resolve_path
    cfg, out = trained
    comp = pd.read_csv(resolve_path(cfg, "metrics_dir") / "model_comparison.csv")
    pdir = resolve_path(cfg, "predictions_dir")
    ids = None
    for r in comp.itertuples():
        t = pd.read_csv(pdir / f"{r.model}_test_predictions.csv")
        m = compute_metrics(t["y_true"], t["prob"], r.threshold)
        assert m["roc_auc"] == pytest.approx(r.roc_auc) and m["recall"] == pytest.approx(r.recall)
        ids = ids if ids is not None else list(t["observation_id"])
        assert list(t["observation_id"]) == ids                    # every model is scored on the same test records
        v = pd.read_csv(pdir / f"{r.model}_val_predictions.csv")   # threshold came from VALIDATION data only
        assert r.threshold == pytest.approx(select_threshold(v["y_true"], v["prob"], cfg["threshold"]["strategy"]))


def test_splits_do_not_leak(trained):
    from src.config import resolve_path
    a = pd.read_csv(resolve_path(trained[0], "metrics_dir") / "split_assignment.csv")
    g = a.groupby("player_id")["split"].nunique()
    assert (g == 1).all()                                          # each player lives in exactly one split


def test_integrated_gradients_completeness(predictor, small_df):
    model = predictor.get_model("multimodal_dl")
    arrays = predictor.preprocessor.transform(small_df.head(12))
    tin = to_tensors(arrays)
    attr = integrated_gradients(model, tin, steps=64)
    total = sum(attr[k].reshape(12, -1).sum(1) for k in ("static", "wl_seq", "wl_sum", "wb_seq"))
    assert np.abs(total - attr["_delta"]).max() < 0.05


def test_tree_shap_is_additive(predictor, small_df):
    import shap
    arrays = predictor.preprocessor.transform(small_df.head(10))
    X = feature_matrix(arrays, "all")
    model = predictor.get_model("xgboost")
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    margin = model.predict(X, output_margin=True)
    assert np.allclose(sv.sum(1) + np.ravel(ex.expected_value), margin, atol=1e-3)


def test_single_explanation_tables(predictor, small_df):
    row = small_df.iloc[[40]]
    for name in ("multimodal_dl", "xgboost", "workload_only"):
        t, method = explain_single(predictor, row, name)
        assert {"feature", "modality", "value", "attribution", "effect"} <= set(t.columns)
        assert t["attribution"].abs().is_monotonic_decreasing and isinstance(method, str)
        assert abs(modality_from_table(t)["share_percent"].sum() - 100) < 1e-6
    assert any("Recent fatigue" in s for s in answer_questions(explain_single(predictor, row)[0]))
    with pytest.raises(NotImplementedError):
        explain_single(predictor, row, "random_forest")


def test_modality_permutation_importance(predictor, small_df):
    sub = small_df.head(200)
    arrays = predictor.preprocessor.transform(sub)
    t = permutation_modality_importance(predictor, arrays, sub["injury_next_7_days"].to_numpy(), repeats=2, seed=0)
    assert set(t["modality"]) == {"Static & medical", "Workload", "Wellness"}


def test_report_exports(predictor, small_df):
    row = small_df.iloc[[7]]
    pred = predictor.predict(row).iloc[0]
    contrib, method = explain_single(predictor, row)
    rep = build_report(player_id=row["player_id"].iloc[0], observation_date=row["observation_date"].iloc[0], model_name=pred["model"],
                       probability=pred["probability"], threshold=pred["threshold"], risk_category=pred["risk_category"],
                       predicted_label=pred["predicted_label"], input_summary={"age": int(row["age"].iloc[0])},
                       contributions=contrib.head(5), explanation_method=method)
    csv_bytes, pdf_bytes = report_to_csv(rep), report_to_pdf(rep)
    text = csv_bytes.decode()
    assert "SYNTHETIC" in text and "estimated_injury_probability" in text and "limitation" in text
    assert pdf_bytes[:4] == b"%PDF" and len(pdf_bytes) > 1500
