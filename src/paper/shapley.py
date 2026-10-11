"""Monte-Carlo permutation Shapley values over the WHOLE pipeline (raw inputs -> encoders -> head -> probability).

Feature set = raw model inputs, with ``position`` treated as ONE feature. Baseline distribution = a random background
sample of training rows. Values are in probability units: sum of values = f(x) - E[f(background)] (approximately, MC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.paper.features import MODALITY_FEATURES, MODALITY_LABEL


def shapley_values(predict, x_row: pd.Series, background: pd.DataFrame, features: list[str], n_perm: int, rng) -> np.ndarray:
    F = len(features)
    rows, orders, bgs = [], [], []
    for _ in range(n_perm):
        order = rng.permutation(F)
        b = background.iloc[int(rng.integers(0, len(background)))]
        cur = b.copy()
        rows.append(cur.copy())
        for j in order:
            cur[features[j]] = x_row[features[j]]
            rows.append(cur.copy())
        orders.append(order)
    pred = predict(pd.DataFrame(rows).reset_index(drop=True)).reshape(n_perm, F + 1)
    phi = np.zeros(F)
    for k, order in enumerate(orders):
        phi[order] += np.diff(pred[k])
    return phi / n_perm


def global_importance(model, background, instances, n_perm=12, seed=42, bg_size=60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = model.raw_columns
    bg = background.sample(min(bg_size, len(background)), random_state=seed).reset_index(drop=True)
    phis = np.array([shapley_values(model.predict_proba, inst, bg, feats, n_perm, rng) for _, inst in instances.iterrows()])
    imp = pd.DataFrame({"feature": feats, "mean_abs_shapley": np.abs(phis).mean(0), "mean_shapley": phis.mean(0)})
    mod_of = {c: m for m in model.modalities for c in MODALITY_FEATURES[m]}
    imp["modality"] = imp.feature.map(mod_of)
    return imp.sort_values("mean_abs_shapley", ascending=False).reset_index(drop=True), phis


def modality_share(imp: pd.DataFrame) -> pd.DataFrame:
    s = imp.groupby("modality").mean_abs_shapley.sum()
    out = (s / s.sum() * 100).rename("importance_pct").reset_index()
    out["label"] = out.modality.map(MODALITY_LABEL)
    return out.sort_values("importance_pct", ascending=False).reset_index(drop=True)


def case_study(model, background, row: pd.Series, n_perm=60, seed=42, bg_size=60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bg = background.sample(min(bg_size, len(background)), random_state=seed).reset_index(drop=True)
    phi = shapley_values(model.predict_proba, row, bg, model.raw_columns, n_perm, rng)
    out = pd.DataFrame({"feature": model.raw_columns, "value": [row[c] for c in model.raw_columns], "shapley": phi})
    out["baseline_prob"] = float(model.predict_proba(bg).mean())
    out["predicted_prob"] = float(model.predict_proba(row.to_frame().T.astype(object).infer_objects())[0])
    return out.reindex(out.shapley.abs().sort_values(ascending=False).index).reset_index(drop=True)
