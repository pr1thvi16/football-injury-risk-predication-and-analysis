"""Leakage-safe preprocessing: splitting, imputation, encoding, scaling, sequence construction.

All statistics (medians, means, std, one-hot categories) are fitted on the TRAINING split only.
The target and the prediction-horizon are never used by any transformer.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import (
    POSITIONS, STATIC_NUMERIC, SUMMARY_FEATURES, TARGET, WELLNESS_VARS, WORKLOAD_CHANNELS,
    aggregate_wellness_features, aggregate_workload_features, compute_average_rest_days,
    sequences_from_frame, static_frame,
)


# ----------------------------------------------------------------------------- splitting
def extract_target(df: pd.DataFrame) -> np.ndarray:
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    return df[TARGET].to_numpy(dtype=int)


def _split_sizes(n: int, cfg: dict) -> tuple[int, int, int]:
    s = cfg["split"]
    tot = s["train_size"] + s["val_size"] + s["test_size"]
    n_test = int(round(n * s["test_size"] / tot))
    n_val = int(round(n * s["val_size"] / tot))
    return n - n_val - n_test, n_val, n_test


def make_splits(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Return positional indices for train / val / test.

    strategy 'grouped': whole players are assigned to one split (evaluates generalisation to UNSEEN players).
        With the default 250 players x 48 weeks this reproduces exactly 8,160 / 1,440 / 2,400 records.
        Among ``grouped_candidates`` seeded random partitions the one whose split prevalences are closest to the
        overall prevalence is kept (a stratified-group split; only used to balance the split, never to train).
    strategy 'random_stratified': record-level stratified split. The same player then appears in train and
        test, which makes the evaluation optimistic - provided only for comparison.
    """
    seed = cfg["project"]["seed"]
    y = extract_target(df)
    n = len(df)
    n_tr, n_va, n_te = _split_sizes(n, cfg)
    strategy = cfg["split"]["strategy"]
    if strategy == "random_stratified":
        idx = np.arange(n)
        rest, test = train_test_split(idx, test_size=n_te, stratify=y, random_state=seed)
        train, val = train_test_split(rest, test_size=n_va, stratify=y[rest], random_state=seed)
        return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    if strategy != "grouped":
        raise ValueError(f"Unknown split strategy '{strategy}' (use 'grouped' or 'random_stratified')")

    players = df["player_id"].to_numpy()
    uniq = np.array(sorted(set(players)))
    n_p = len(uniq)
    s = cfg["split"]
    tot = s["train_size"] + s["val_size"] + s["test_size"]          # fractions 68 / 12 / 20 %
    p_te = int(round(n_p * s["test_size"] / tot))
    p_va = int(round(n_p * s["val_size"] / tot))
    overall = y.mean()
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(int(cfg["split"].get("grouped_candidates", 1))):
        perm = rng.permutation(uniq)
        groups = {"test": perm[:p_te], "val": perm[p_te:p_te + p_va], "train": perm[p_te + p_va:]}
        score = max(abs(y[np.isin(players, g)].mean() - overall) for g in groups.values())
        if score < best_score:
            best, best_score = groups, score
    return {k: np.flatnonzero(np.isin(players, best[k])) for k in ("train", "val", "test")}


def describe_splits(df: pd.DataFrame, splits: dict[str, np.ndarray], cfg: dict) -> dict:
    y = extract_target(df)
    players = df["player_id"].to_numpy()
    info = {"strategy": cfg["split"]["strategy"], "seed": cfg["project"]["seed"], "splits": {}}
    for k, idx in splits.items():
        info["splits"][k] = {"n_records": int(len(idx)), "n_players": int(len(set(players[idx]))),
                             "n_positive": int(y[idx].sum()), "prevalence": float(y[idx].mean())}
    sets = {k: set(df["observation_id"].iloc[v]) for k, v in splits.items()}
    info["record_overlap"] = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    pl = {k: set(players[v]) for k, v in splits.items()}
    info["player_overlap"] = {f"{a}&{b}": len(pl[a] & pl[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    return info


def split_assignment_frame(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame({"observation_id": df["observation_id"], "player_id": df["player_id"], "split": ""})
    for k, idx in splits.items():
        out.loc[idx, "split"] = k
    return out


def compute_pos_weight(y_train: np.ndarray, power: float) -> float:
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    return float((neg / pos) ** power)


# ----------------------------------------------------------------------------- preprocessor
class Preprocessor:
    """Fits on train, transforms any split / new input into model-ready arrays.

    transform() returns a dict:
        static (N, Ds)        imputed + scaled numeric static features and one-hot position
        wl_seq (N, 28, 8)     per-channel standardised workload sequence (NaN -> training mean)
        wl_sum (N, 7)         standardised derived workload features
        wb_seq (N, 7, 7)      per-variable standardised wellness sequence
        tab_static / tab_workload / tab_wellness   fixed-length scaled aggregates for classical models
    """

    def __init__(self):
        self.fitted = False

    # ---- helpers
    def _static_matrix(self, sdf: pd.DataFrame, wl: np.ndarray, fit: bool) -> np.ndarray:
        num = sdf[STATIC_NUMERIC].astype(float).copy()
        # average_rest_days is always derived from the supplied workload sequence (keeps inputs self-consistent)
        num["average_rest_days"] = compute_average_rest_days(wl)
        if fit:
            self.static_imputer = SimpleImputer(strategy="median").fit(num)
            self.static_scaler = StandardScaler().fit(self.static_imputer.transform(num))
            self.position_encoder = OneHotEncoder(categories=[POSITIONS], handle_unknown="ignore", sparse_output=False).fit(sdf[["position"]])
        x_num = self.static_scaler.transform(self.static_imputer.transform(num))
        x_pos = self.position_encoder.transform(sdf[["position"]])
        return np.hstack([x_num, x_pos]).astype(np.float32)

    def _fit_transform_tab(self, name: str, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            imp = SimpleImputer(strategy="median").fit(frame)
            sc = StandardScaler().fit(imp.transform(frame))
            setattr(self, f"{name}_imputer", imp)
            setattr(self, f"{name}_scaler", sc)
            setattr(self, f"{name}_names", list(frame.columns))
        imp, sc = getattr(self, f"{name}_imputer"), getattr(self, f"{name}_scaler")
        return sc.transform(imp.transform(frame[getattr(self, f"{name}_names")])).astype(np.float32)

    def _scale_seq(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        z = (x - mean) / std
        return np.nan_to_num(z, nan=0.0).astype(np.float32)   # NaN -> training mean (0 after scaling)

    def _build(self, sdf, wl, wb, fit: bool) -> dict[str, np.ndarray]:
        if fit:
            self.wl_mean = np.nanmean(wl, axis=(0, 1))
            self.wl_std = np.maximum(np.nanstd(wl, axis=(0, 1)), 1e-6)
            self.wb_mean = np.nanmean(wb, axis=(0, 1))
            self.wb_std = np.maximum(np.nanstd(wb, axis=(0, 1)), 1e-6)
        wl_agg = aggregate_workload_features(wl)
        wb_agg = aggregate_wellness_features(wb)
        static = self._static_matrix(sdf, wl, fit)
        if fit:
            self.static_names = STATIC_NUMERIC + [f"position_{p}" for p in POSITIONS]
        out = {
            "static": static,
            "wl_seq": self._scale_seq(wl, self.wl_mean, self.wl_std),
            "wl_sum": self._fit_transform_tab("wlsum", wl_agg[SUMMARY_FEATURES], fit),
            "wb_seq": self._scale_seq(wb, self.wb_mean, self.wb_std),
            "tab_static": static,
            "tab_workload": self._fit_transform_tab("tabwl", wl_agg, fit),
            "tab_wellness": self._fit_transform_tab("tabwb", wb_agg, fit),
        }
        return out

    # ---- public API
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        wl, wb = sequences_from_frame(df)
        self._build(static_frame(df), wl, wb, fit=True)
        self.fitted = True
        return self

    def transform_arrays(self, sdf: pd.DataFrame, wl: np.ndarray, wb: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform()")
        return self._build(sdf.reset_index(drop=True), wl, wb, fit=False)

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        wl, wb = sequences_from_frame(df)
        return self.transform_arrays(static_frame(df), wl, wb)

    def fit_transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(df).transform(df)

    # ---- metadata
    @property
    def feature_groups(self) -> dict[str, list[str]]:
        """Names of the tabular (classical-model) feature groups, in column order."""
        return {"static": list(self.static_names), "workload": list(self.tabwl_names), "wellness": list(self.tabwb_names)}

    @property
    def nn_feature_names(self) -> dict[str, list[str]]:
        return {"static": list(self.static_names), "wl_seq": list(WORKLOAD_CHANNELS),
                "wl_sum": list(SUMMARY_FEATURES), "wb_seq": list(WELLNESS_VARS)}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact missing: {path}. Train first: python -m src.training.train")
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Preprocessor artifact {path} is corrupted: {exc}") from exc
        if not isinstance(obj, Preprocessor):
            raise RuntimeError(f"{path} does not contain a Preprocessor")
        return obj
