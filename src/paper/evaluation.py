"""Evaluation protocol of the paper: Youden threshold on validation, bootstrap CIs, paired bootstrap, calibration
quintiles, mean-substitution ablation, subgroup AUCs, 5-fold CV of the fusion head. All numbers come from predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.paper.features import MODALITY_FEATURES, TARGET


def grouped_split(df: pd.DataFrame, n_train: int, n_val: int, n_test: int, seed: int = 42):
    """Player-grouped split; player counts follow the requested record proportions (no player appears in two sets)."""
    rng = np.random.default_rng(seed)
    players = np.array(sorted(df.player_id.unique()))
    rng.shuffle(players)
    tot = n_train + n_val + n_test
    a = int(round(len(players) * n_train / tot)); b = int(round(len(players) * n_val / tot))
    sets = {"train": players[:a], "val": players[a:a + b], "test": players[a + b:]}
    return {k: df[df.player_id.isin(v)].reset_index(drop=True) for k, v in sets.items()}


def random_split(df, n_train, n_val, n_test, seed=42):
    from sklearn.model_selection import train_test_split
    tr, rest = train_test_split(df, train_size=n_train / (n_train + n_val + n_test), stratify=df[TARGET], random_state=seed)
    va, te = train_test_split(rest, train_size=n_val / (n_val + n_test), stratify=rest[TARGET], random_state=seed)
    return {"train": tr.reset_index(drop=True), "val": va.reset_index(drop=True), "test": te.reset_index(drop=True)}


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def point_metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "specificity": tn / (tn + fp), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p), "threshold": thr,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap_auc(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(y, p_a, p_b, n=1000, seed=42) -> dict:
    """AUC(a) - AUC(b) over resampled test sets; reports mean difference, 95 % interval and share of resamples a > b."""
    rng = np.random.default_rng(seed); y = np.asarray(y); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    d = np.array(d)
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p_a_better": float((d > 0).mean())}


def calibration_quintiles(y, p) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
    return (pd.DataFrame({"quintile": q + 1, "y": np.asarray(y), "p": p}).groupby("quintile")
            .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index())


def modality_ablation(model, df, train_ref) -> pd.DataFrame:
    """Mean-substitution ablation: replace one modality's raw inputs with training means (mode for position)."""
    base = roc_auc_score(df[TARGET], model.predict_proba(df)); rows = []
    for m in model.modalities:
        d = df.copy()
        for c in MODALITY_FEATURES[m]:
            d[c] = train_ref[c].mode().iloc[0] if c == "position" else train_ref[c].mean()
        auc = roc_auc_score(df[TARGET], model.predict_proba(d))
        rows.append({"ablated_modality": m, "auc_full": base, "auc_ablated": auc, "auc_drop": base - auc})
    return pd.DataFrame(rows)


def subgroup_auc(df, p) -> pd.DataFrame:
    groups = {f"position={g}": df.position == g for g in ["GK", "DEF", "MID", "FWD"]}
    groups.update({"age<=24": df.age <= 24, "age>=30": df.age >= 30})
    rows = []
    for name, mask in groups.items():
        mask = mask.to_numpy()
        y = df[TARGET].to_numpy()[mask]
        rows.append({"subgroup": name, "n": int(mask.sum()), "positives": int(y.sum()),
                     "roc_auc": roc_auc_score(y, p[mask]) if 0 < y.sum() < len(y) else np.nan})
    return pd.DataFrame(rows)


def cv_head(model, df, folds=5, seed=42) -> dict:
    """5-fold CV of the fusion head on frozen encoder outputs (encoders already saw these rows: optimistic; reported as such)."""
    X, y = model._design(df), df[TARGET].to_numpy()
    p = cross_val_predict(LogisticRegression(C=model.cfg["head_C"], max_iter=2000), X, y,
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return {"cv_auc": float(roc_auc_score(y, p)), "folds": folds}
