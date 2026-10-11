"""Baseline models and the documented time-series aggregation strategy.

Classical models cannot read sequences, so each modality is summarised into fixed-length features
(see src/features/feature_engineering.py):
  workload : 7 derived summaries + per channel {mean_28d, std_28d, max_28d, mean_7d}   (39 features)
  wellness : per variable {mean_7d, std_7d, last_day, slope_7d}                         (28 features)
  static   : numeric static/medical features + one-hot position                         (16 features)
Models: logistic regression, random forest, XGBoost (gradient boosting), sklearn MLP on all features;
unimodal XGBoost models (workload-only / wellness-only / static-only) use ONLY their own modality.
Class imbalance: every model gets the same positive weight (neg/pos)**power (configurable).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

BASELINE_NAMES = ["logistic_regression", "random_forest", "xgboost", "mlp", "workload_only", "wellness_only", "static_only"]
FEATURE_GROUP = {"logistic_regression": "all", "random_forest": "all", "xgboost": "all", "mlp": "all",
                 "workload_only": "workload", "wellness_only": "wellness", "static_only": "static"}


def feature_matrix(tab: dict, group: str) -> np.ndarray:
    if group == "all":
        return np.hstack([tab["tab_static"], tab["tab_workload"], tab["tab_wellness"]])
    return tab[f"tab_{group}"]


def all_feature_names(preprocessor) -> list[str]:
    g = preprocessor.feature_groups
    return [f"[static] {n}" for n in g["static"]] + [f"[workload] {n}" for n in g["workload"]] + [f"[wellness] {n}" for n in g["wellness"]]


def _xgb(cfg, pos_weight, seed):
    p = cfg["baselines"]["xgboost"]
    if HAS_XGB:
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                             subsample=p["subsample"], colsample_bytree=p["colsample_bytree"], scale_pos_weight=pos_weight,
                             eval_metric="logloss", random_state=seed, n_jobs=2, verbosity=0)
    return HistGradientBoostingClassifier(max_depth=p["max_depth"], learning_rate=p["learning_rate"], random_state=seed)


def make_estimator(name: str, cfg: dict, pos_weight: float, seed: int):
    b = cfg["baselines"]
    if name == "logistic_regression":
        return LogisticRegression(C=b["logistic_regression"]["C"], max_iter=b["logistic_regression"]["max_iter"],
                                  class_weight={0: 1.0, 1: pos_weight})
    if name == "random_forest":
        p = b["random_forest"]
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                      class_weight={0: 1.0, 1: pos_weight}, random_state=seed, n_jobs=2)
    if name in ("xgboost", "workload_only", "wellness_only", "static_only"):
        return _xgb(cfg, pos_weight, seed)
    if name == "mlp":
        p = b["mlp"]
        return MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), alpha=p["alpha"], max_iter=p["max_iter"],
                             early_stopping=True, n_iter_no_change=15, random_state=seed)
    raise ValueError(f"Unknown baseline '{name}'")


def fit_baseline(name: str, cfg: dict, X: np.ndarray, y: np.ndarray, pos_weight: float, seed: int):
    est = make_estimator(name, cfg, pos_weight, seed)
    if name == "mlp":
        sw = np.where(y == 1, pos_weight, 1.0)
        est.fit(X, y, sample_weight=sw)
    else:
        est.fit(X, y)
    return est
