import numpy as np
import pytest

from src.config import load_config, risk_category
from src.evaluation.metrics import apply_threshold, compute_metrics, curves, select_threshold
from src.models.calibration import PlattCalibrator

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05, 0.4, 0.0])


def test_metric_values_hand_computed():
    m = compute_metrics(Y, P, 0.5)          # preds: 1,1,0,1,0,0,0,0,0,0 -> tp=2 fp=1 fn=1 tn=6
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(6 / 7)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["brier"] == pytest.approx(np.mean((P - Y) ** 2))
    assert 0 < m["roc_auc"] <= 1 and 0 < m["pr_auc"] <= 1


def test_perfect_and_one_class():
    assert compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1.0
    assert np.isnan(compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])["roc_auc"])


def test_thresholding_and_invalid_probabilities():
    assert apply_threshold([0.2, 0.5, 0.51], 0.5).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        apply_threshold([1.2], 0.5)


def test_threshold_selection_uses_only_given_validation_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500); p = np.clip(0.3 * y + rng.normal(0.3, 0.15, 500), 0, 1)
    t_f1 = select_threshold(y, p, "val_f1")
    assert compute_metrics(y, p, t_f1)["f1"] >= max(compute_metrics(y, p, t)["f1"] for t in (0.2, 0.5, 0.8)) - 1e-9
    assert select_threshold(y, p, "fixed", 0.37) == 0.37
    t_rec = select_threshold(y, p, "val_recall_target", recall_target=0.8)
    assert compute_metrics(y, p, t_rec)["recall"] >= 0.8
    assert 0 < select_threshold(y, p, "val_youden") < 1
    with pytest.raises(ValueError):
        select_threshold(y, p, "bogus")


def test_curves_shapes():
    c = curves(Y, P, n_bins=3)
    assert len(c["fpr"]) == len(c["tpr"]) and len(c["precision"]) == len(c["recall"])


def test_calibrator_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.02, 0.98, 800); y = (rng.uniform(size=800) < raw * 0.3).astype(int)
    cal = PlattCalibrator().fit(raw, y)
    out = cal.transform(np.linspace(0.01, 0.99, 50))
    assert (np.diff(out) >= -1e-12).all() and (out >= 0).all() and (out <= 1).all()
    assert out.mean() < 0.5                                    # shrinks inflated raw scores
    with pytest.raises(ValueError):
        PlattCalibrator().fit(raw, np.zeros(800))


def test_risk_categories_boundaries():
    cfg = load_config()
    assert [risk_category(p, cfg) for p in (0, 0.199, 0.2, 0.399, 0.4, 0.599, 0.6, 1.0)] == \
        ["Low", "Low", "Moderate", "Moderate", "Elevated", "Elevated", "High", "High"]
    with pytest.raises(ValueError):
        risk_category(1.5, cfg)
    cfg["risk_categories"][1]["lower"] = 0.1                    # configurable in the single settings file
    assert risk_category(0.15, cfg) == "Moderate"
