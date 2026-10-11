"""Inference with the SAVED trained models (never hard-coded or random predictions)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, risk_category
from src.data.preprocessing import Preprocessor
from src.data.validation import validate_inference_frame
from src.evaluation.metrics import apply_threshold
from src.models.baselines import FEATURE_GROUP, feature_matrix
from src.models.multimodal_model import load_model, predict_proba, to_tensors


class ArtifactError(RuntimeError):
    """Raised when saved model artifacts are missing or corrupted."""


class InjuryRiskPredictor:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.models_dir = resolve_path(self.cfg, "models_dir")
        reg = self.models_dir / "model_registry.json"
        if not reg.exists():
            raise FileNotFoundError(f"No trained models found ({reg}). Train first:  python -m src.training.train")
        try:
            self.registry = json.loads(reg.read_text())
        except Exception as exc:
            raise ArtifactError(f"Model registry {reg} is corrupted: {exc}") from exc
        self.preprocessor = Preprocessor.load(self.models_dir / "preprocessor.joblib")
        cal = self.models_dir / "calibrators.joblib"
        if not cal.exists():
            raise FileNotFoundError(f"Calibrator artifact missing: {cal}")
        try:
            self.calibrators = joblib.load(cal)
        except Exception as exc:
            raise ArtifactError(f"Calibrator artifact {cal} is corrupted: {exc}") from exc
        self._models: dict = {}

    # ------------------------------------------------------------------ info
    @property
    def available_models(self) -> list[str]:
        return list(self.registry["models"])

    @property
    def selected_model(self) -> str:
        return self.registry["selected_model"]

    def threshold(self, name: str | None = None) -> float:
        return float(self.registry["thresholds"][name or self.selected_model])

    def get_model(self, name: str):
        if name not in self.available_models:
            raise ValueError(f"Unknown model '{name}'. Available: {self.available_models}")
        if name not in self._models:
            if name == "multimodal_dl":
                self._models[name] = load_model(self.models_dir / "multimodal_dl")
            else:
                p = self.models_dir / f"{name}.joblib"
                if not p.exists():
                    raise FileNotFoundError(f"Model artifact missing: {p}")
                try:
                    self._models[name] = joblib.load(p)
                except Exception as exc:
                    raise ArtifactError(f"Model artifact {p} is corrupted: {exc}") from exc
        return self._models[name]

    # ------------------------------------------------------------------ prediction
    def raw_scores(self, name: str, arrays: dict) -> np.ndarray:
        model = self.get_model(name)
        if name == "multimodal_dl":
            return predict_proba(model, to_tensors(arrays)).numpy()
        return model.predict_proba(feature_matrix(arrays, FEATURE_GROUP[name]))[:, 1]

    def predict(self, df: pd.DataFrame, model_name: str | None = None, threshold: float | None = None) -> pd.DataFrame:
        name = model_name or self.selected_model
        validate_inference_frame(df)
        arrays = self.preprocessor.transform(df.reset_index(drop=True))
        raw = self.raw_scores(name, arrays)
        prob = self.calibrators[name].transform(raw)
        thr = self.threshold(name) if threshold is None else float(threshold)
        out = pd.DataFrame({
            "observation_id": df["observation_id"].to_numpy() if "observation_id" in df else np.arange(len(df)),
            "player_id": df["player_id"].to_numpy() if "player_id" in df else "HYPOTHETICAL",
            "model": name, "probability": prob, "threshold": thr,
            "predicted_label": apply_threshold(prob, thr),
        })
        out["risk_category"] = [risk_category(float(p), self.cfg) for p in prob]
        return out
