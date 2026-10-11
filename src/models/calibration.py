"""Post-hoc Platt calibration (fitted on the VALIDATION split only)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Maps raw scores to calibrated probabilities. Needed because class weighting inflates raw probabilities."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.fitted = False

    def fit(self, raw_prob, y) -> "PlattCalibrator":
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration needs both classes in the validation set")
        self.lr.fit(_logit(raw_prob), y)
        self.fitted = True
        return self

    def transform(self, raw_prob) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator is not fitted")
        return self.lr.predict_proba(_logit(raw_prob))[:, 1]
