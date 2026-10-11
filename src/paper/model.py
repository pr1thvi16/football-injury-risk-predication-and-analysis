"""Paper architecture: per-modality MLP encoders -> 16-d embeddings -> logistic-regression fusion head, plus the 7 baselines.

PaperModel works on the raw player-week table produced by ``real_data.build_player_weeks`` (SYNTHETIC or REAL).
Everything that is fitted (imputers, scalers, encoders, head) is fitted on the training rows only.
"""
from __future__ import annotations

import copy
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.paper.features import MODALITY_FEATURES, POSITIONS, STATIC_NUMERIC, TARGET


class Tabular:
    """Median imputation + standardisation (+ one-hot position) for a set of raw columns; fit on train only."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.num = [c for c in self.columns if c != "position"]
        self.has_pos = "position" in self.columns

    def fit(self, df: pd.DataFrame):
        x = df[self.num].astype(float)
        self.median_ = x.median().fillna(0.0)
        x = x.fillna(self.median_)
        self.mean_ = x.mean()
        self.std_ = x.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.empty_ = [c for c in self.num if df[c].notna().sum() == 0]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.num].astype(float).fillna(self.median_)
        z = ((x - self.mean_) / self.std_).to_numpy()
        if self.has_pos:
            oh = np.stack([(df["position"] == p).to_numpy(float) for p in POSITIONS], 1)
            z = np.hstack([z, oh])
        return z

    @property
    def names(self):
        return self.num + ([f"position={p}" for p in POSITIONS] if self.has_pos else [])



class EarlyStoppedMLP:
    """sklearn MLP trained epoch-by-epoch with early stopping on held-out LOG-LOSS (best weights restored).

    sklearn's built-in early_stopping monitors accuracy, which is flat for a 9 %-positive target and halts training
    after ~2 epochs, so we do our own. The inner validation split is grouped by player when ids are supplied."""

    def __init__(self, hidden=(32, 16), alpha=0.01, max_iter=400, patience=20, val_frac=0.15, seed=42):
        self.mlp = MLPClassifier(hidden_layer_sizes=tuple(hidden), alpha=alpha, random_state=seed)
        self.max_iter, self.patience, self.val_frac, self.seed = max_iter, patience, val_frac, seed

    def fit(self, X, y, groups=None):
        from sklearn.metrics import log_loss
        rng = np.random.default_rng(self.seed)
        if groups is not None:
            ug = np.unique(groups); rng.shuffle(ug)
            val = np.isin(groups, ug[:max(1, int(len(ug) * self.val_frac))])
        else:
            val = rng.random(len(y)) < self.val_frac
        if y[val].min() == y[val].max():
            val = rng.random(len(y)) < self.val_frac
        Xt, yt, Xv, yv = X[~val], y[~val], X[val], y[val]
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.max_iter):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.mlp.partial_fit(Xt, yt, classes=[0, 1])
            loss = log_loss(yv, self.mlp.predict_proba(Xv)[:, 1], labels=[0, 1])
            if loss < best - 1e-5:
                best, bad = loss, 0
                best_state = ([w.copy() for w in self.mlp.coefs_], [b.copy() for b in self.mlp.intercepts_])
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.mlp.coefs_, self.mlp.intercepts_ = best_state
        self.n_epochs_, self.best_val_loss_ = epoch + 1, best
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(X)

    def fine_tune(self, X, y, epochs=30):
        for _ in range(epochs):
            self.mlp.partial_fit(X, y, classes=[0, 1])
        return self

    @property
    def coefs_(self): return self.mlp.coefs_
    @property
    def intercepts_(self): return self.mlp.intercepts_


class ModalityEncoder:
    """Compact MLP on one modality; the 16-d embedding is the activation of the last hidden ReLU layer."""

    def __init__(self, columns, hidden=(32, 16), alpha=0.01, max_iter=400, seed=42):
        self.tab = Tabular(columns)
        self.mlp = EarlyStoppedMLP(hidden, alpha, max_iter, seed=seed)

    def fit(self, df, y):
        g = df["player_id"].to_numpy() if "player_id" in df else None
        self.mlp.fit(self.tab.fit(df).transform(df), np.asarray(y), g)
        return self

    def fine_tune(self, df, y, max_iter=30, refit_preprocessing=False):
        """Warm-start further training on new (e.g. real) data, keeping learned weights."""
        if refit_preprocessing:
            self.tab.fit(df)
        self.mlp.fine_tune(self.tab.transform(df), np.asarray(y), epochs=max_iter)
        return self

    def embed(self, df) -> np.ndarray:
        a = self.tab.transform(df)
        for W, b in zip(self.mlp.coefs_[:-2], self.mlp.intercepts_[:-2]):
            a = np.maximum(a @ W + b, 0)
        W, b = self.mlp.coefs_[-2], self.mlp.intercepts_[-2]           # last hidden layer
        return np.maximum(a @ W + b, 0)

    def predict_proba(self, df):
        return self.mlp.predict_proba(self.tab.transform(df))[:, 1]


class PaperModel:
    """Fusion model. ``modalities`` may be a subset (e.g. ["load", "wellness"] when real data has no static fields)."""

    def __init__(self, modalities=("static", "load", "wellness"), hidden=(32, 16), alpha=0.01, max_iter=400, head_C=0.05, seed=42):
        self.modalities = list(modalities)
        self.cfg = dict(hidden=tuple(hidden), alpha=alpha, max_iter=max_iter, head_C=head_C, seed=seed)

    @property
    def raw_columns(self):
        return [c for m in self.modalities for c in MODALITY_FEATURES[m]]

    def _design(self, df):
        parts = [self.encoders[m].embed(df) for m in self.modalities]
        parts.append(self.raw_tab.transform(df))
        return np.hstack(parts)

    def fit(self, df, y=None):
        y = df[TARGET].to_numpy() if y is None else y
        self.encoders = {m: ModalityEncoder(MODALITY_FEATURES[m], seed=self.cfg["seed"], hidden=self.cfg["hidden"],
                                            alpha=self.cfg["alpha"], max_iter=self.cfg["max_iter"]).fit(df, y)
                         for m in self.modalities}
        self.raw_tab = Tabular(self.raw_columns).fit(df)
        self.head = LogisticRegression(C=self.cfg["head_C"], max_iter=2000).fit(self._design(df), y)
        return self

    def predict_proba(self, df):
        return self.head.predict_proba(self._design(df))[:, 1]

    def modality_scores(self, df):
        return {m: self.encoders[m].predict_proba(df) for m in self.modalities}

    def fine_tune(self, df, y=None, max_iter=30, refit_preprocessing=True, refit_head=True):
        """Warm-start the encoders and refit the (cheap) head on new data. Returns a new model; the original is untouched."""
        y = df[TARGET].to_numpy() if y is None else y
        new = copy.deepcopy(self)
        for m in new.modalities:
            new.encoders[m].fine_tune(df, y, max_iter=max_iter, refit_preprocessing=refit_preprocessing)
        if refit_preprocessing:
            new.raw_tab = Tabular(new.raw_columns).fit(df)
        if refit_head:
            new.head = LogisticRegression(C=new.cfg["head_C"], max_iter=2000).fit(new._design(df), y)
        return new

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class SklearnBaseline:
    """Baseline on a column subset with the same train-only preprocessing."""

    def __init__(self, name, columns, estimator):
        self.name, self.tab, self.est = name, Tabular(columns), estimator

    def fit(self, df, y=None):
        y = df[TARGET].to_numpy() if y is None else y
        X = self.tab.fit(df).transform(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if isinstance(self.est, EarlyStoppedMLP):
                self.est.fit(X, np.asarray(y), df["player_id"].to_numpy() if "player_id" in df else None)
            else:
                self.est.fit(X, y)
        return self

    def predict_proba(self, df):
        return self.est.predict_proba(self.tab.transform(df))[:, 1]


def make_baselines(cfg: dict, modalities=("static", "load", "wellness"), seed=42) -> dict:
    """The paper's seven comparison models. Unimodal baselines are skipped for modalities that are unavailable."""
    b = cfg["baselines"]
    allc = [c for m in modalities for c in MODALITY_FEATURES[m]]
    mlp = lambda h: EarlyStoppedMLP(h, cfg["encoder"]["alpha"], cfg["encoder"]["max_iter"], seed=seed)
    out = {}
    if "static" in modalities:
        out["LR (static only)"] = SklearnBaseline("LR (static only)", MODALITY_FEATURES["static"], LogisticRegression(C=b["logistic_C"], max_iter=2000))
    if "load" in modalities:
        out["MLP (load only)"] = SklearnBaseline("MLP (load only)", MODALITY_FEATURES["load"], mlp(cfg["encoder"]["hidden"]))
    if "wellness" in modalities:
        out["MLP (wellness only)"] = SklearnBaseline("MLP (wellness only)", MODALITY_FEATURES["wellness"], mlp(cfg["encoder"]["hidden"]))
    out["LR (early fusion)"] = SklearnBaseline("LR (early fusion)", allc, LogisticRegression(C=b["logistic_C"], max_iter=2000))
    out["Random forest"] = SklearnBaseline("Random forest", allc, RandomForestClassifier(
        n_estimators=b["random_forest"]["n_estimators"], min_samples_leaf=b["random_forest"]["min_samples_leaf"], n_jobs=-1, random_state=seed))
    out["Gradient boosting"] = SklearnBaseline("Gradient boosting", allc, GradientBoostingClassifier(random_state=seed))
    out["MLP (early fusion)"] = SklearnBaseline("MLP (early fusion)", allc, mlp(b["mlp_early_fusion"]["hidden"]))
    return out
