"""Evaluation metrics, threshold selection and curves. All values are computed from real predictions."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)


def confusion(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def apply_threshold(prob, threshold: float) -> np.ndarray:
    prob = np.asarray(prob, float)
    if not ((prob >= 0).all() and (prob <= 1).all()):
        raise ValueError("Probabilities must lie within [0, 1]")
    return (prob >= threshold).astype(int)


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, float)
    pred = apply_threshold(prob, threshold)
    c = confusion(y_true, pred)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    safe = lambda a, b: float(a / b) if b else 0.0
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    one_class = len(np.unique(y_true)) < 2
    return {
        "threshold": float(threshold),
        "accuracy": safe(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "specificity": safe(tn, tn + fp),
        "f1": safe(2 * precision * recall, precision + recall),
        "roc_auc": float("nan") if one_class else float(roc_auc_score(y_true, prob)),
        "pr_auc": float("nan") if one_class else float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        **c,
    }


def select_threshold(y_val, prob_val, strategy: str = "fixed", fixed_value: float = 0.5, recall_target: float = 0.7) -> float:
    """Pick a decision threshold from VALIDATION data only. Never call this with test labels."""
    y_val = np.asarray(y_val).astype(int)
    prob_val = np.asarray(prob_val, float)
    if strategy == "fixed":
        return float(fixed_value)
    grid = np.unique(np.round(np.quantile(prob_val, np.linspace(0.02, 0.995, 300)), 5))
    if strategy == "val_f1":
        scores = [compute_metrics(y_val, prob_val, t)["f1"] for t in grid]
        return float(grid[int(np.argmax(scores))])
    if strategy == "val_youden":
        scores = [(m := compute_metrics(y_val, prob_val, t))["recall"] + m["specificity"] - 1 for t in grid]
        return float(grid[int(np.argmax(scores))])
    if strategy == "val_recall_target":
        ok = [t for t in grid if compute_metrics(y_val, prob_val, t)["recall"] >= recall_target]
        return float(max(ok)) if ok else float(grid[0])
    raise ValueError(f"Unknown threshold strategy '{strategy}'")


def curves(y_true, prob, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec, "calib_true": frac_pos, "calib_pred": mean_pred}
