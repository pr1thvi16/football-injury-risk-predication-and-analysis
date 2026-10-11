"""Feature definitions and the documented formulas for derived features.

Conventions
-----------
* Workload history: 28 days, column ``wl_<channel>_d<k>``; ``d1`` = oldest day, ``d28`` = the day
  before the observation date.  Wellness history: 7 days, ``wb_<variable>_d<k>``; ``d7`` = day before.
* Rest day: ``training_duration_minutes == 0`` (all workload channels are 0).
* All windows end the day BEFORE the observation date, so no information from the prediction
  horizon (the following 7 days) is ever used.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

WORKLOAD_CHANNELS = [
    "total_distance_km",
    "high_speed_running_distance_km",
    "sprint_distance_km",
    "acceleration_count",
    "deceleration_count",
    "training_duration_minutes",
    "session_rating_of_perceived_exertion",
    "session_training_load",
]
WELLNESS_VARS = [
    "sleep_hours",
    "sleep_quality",
    "fatigue_score",
    "muscle_soreness_score",
    "mood_score",
    "recovery_score",
    "perceived_stress_score",
]
WORKLOAD_DAYS = 28
WELLNESS_DAYS = 7
POSITIONS = ["GK", "CB", "FB", "CM", "W", "ST"]

STATIC_NUMERIC = [
    "age", "height_cm", "weight_kg", "bmi", "previous_injury_count", "previous_injury_days_missed",
    "days_since_last_injury", "season_minutes_played", "strength_asymmetry_percent", "average_rest_days",
]
STATIC_CATEGORICAL = ["position"]
SUMMARY_FEATURES = [
    "acute_workload", "chronic_workload", "acute_chronic_workload_ratio", "workload_monotony",
    "workload_strain", "workload_variability", "recent_workload_change",
]
TARGET = "injury_next_7_days"
ID_COLUMNS = ["observation_id", "player_id", "week_index", "observation_date", "data_source"]
LOAD_IDX = WORKLOAD_CHANNELS.index("session_training_load")
DURATION_IDX = WORKLOAD_CHANNELS.index("training_duration_minutes")
MONOTONY_CAP = 10.0
NO_INJURY_DAYS = 730  # days_since_last_injury is capped at 730; players with no injury get 730

MODALITIES = ["Static & medical", "Workload", "Wellness"]


# --------------------------------------------------------------------------- columns / arrays
def wl_columns(n_days: int = WORKLOAD_DAYS) -> list[str]:
    return [f"wl_{c}_d{k}" for c in WORKLOAD_CHANNELS for k in range(1, n_days + 1)]


def wb_columns(n_days: int = WELLNESS_DAYS) -> list[str]:
    return [f"wb_{v}_d{k}" for v in WELLNESS_VARS for k in range(1, n_days + 1)]


def sequences_from_frame(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return workload (N, 28, 8) and wellness (N, 7, 7) arrays from the wide dataframe."""
    missing = [c for c in wl_columns() + wb_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} sequence columns, e.g. {missing[:3]}")
    wl = np.stack(
        [df[[f"wl_{c}_d{k}" for k in range(1, WORKLOAD_DAYS + 1)]].to_numpy(float) for c in WORKLOAD_CHANNELS],
        axis=-1,
    )
    wb = np.stack(
        [df[[f"wb_{v}_d{k}" for k in range(1, WELLNESS_DAYS + 1)]].to_numpy(float) for v in WELLNESS_VARS],
        axis=-1,
    )
    return wl, wb


def frame_from_sequences(wl: np.ndarray, wb: np.ndarray) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {}
    for ci, c in enumerate(WORKLOAD_CHANNELS):
        for k in range(WORKLOAD_DAYS):
            cols[f"wl_{c}_d{k + 1}"] = wl[:, k, ci]
    for vi, v in enumerate(WELLNESS_VARS):
        for k in range(WELLNESS_DAYS):
            cols[f"wb_{v}_d{k + 1}"] = wb[:, k, vi]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- derived workload features
def compute_workload_summaries(wl: np.ndarray) -> pd.DataFrame:
    """Derived workload features from ``session_training_load`` (daily, arbitrary units).

    acute_workload               = mean daily load over the last 7 days
    chronic_workload             = mean daily load over the last 28 days
    acute_chronic_workload_ratio = acute / max(chronic, 1)
    workload_monotony            = mean_7d / std_7d (ddof=1), capped at 10
    workload_strain              = sum of load over last 7 days * monotony
    workload_variability         = std_28d (ddof=1) / max(chronic, 1)   (coefficient of variation)
    recent_workload_change       = (acute - mean of days 15-21 back) / max(mean of days 15-21 back, 1)
    """
    load = wl[:, :, LOAD_IDX]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        acute = np.nanmean(load[:, -7:], axis=1)
        chronic = np.nanmean(load, axis=1)
        std7 = np.nanstd(load[:, -7:], axis=1, ddof=1)
        std28 = np.nanstd(load, axis=1, ddof=1)
        prev = np.nanmean(load[:, -14:-7], axis=1)
        wsum = np.nansum(load[:, -7:], axis=1)
    acwr = acute / np.maximum(chronic, 1.0)
    monotony = np.minimum(acute / np.maximum(std7, 1e-6), MONOTONY_CAP)
    return pd.DataFrame(
        {
            "acute_workload": acute,
            "chronic_workload": chronic,
            "acute_chronic_workload_ratio": acwr,
            "workload_monotony": monotony,
            "workload_strain": wsum * monotony,
            "workload_variability": std28 / np.maximum(chronic, 1.0),
            "recent_workload_change": (acute - prev) / np.maximum(prev, 1.0),
        }
    )


def compute_average_rest_days(wl: np.ndarray) -> np.ndarray:
    """Average number of rest days per 7 days over the 28-day window."""
    return (wl[:, :, DURATION_IDX] == 0).sum(axis=1) / (WORKLOAD_DAYS / 7.0)


def compute_bmi(height_cm, weight_kg):
    return np.asarray(weight_kg, float) / (np.asarray(height_cm, float) / 100.0) ** 2


# --------------------------------------------------------------------------- aggregation for classical models
def _slope(x: np.ndarray) -> np.ndarray:
    t = np.arange(x.shape[1], dtype=float)
    t = t - t.mean()
    xc = x - np.nanmean(x, axis=1, keepdims=True)
    return np.nansum(xc * t, axis=1) / (t ** 2).sum()


def aggregate_workload_features(wl: np.ndarray) -> pd.DataFrame:
    """Fixed-length workload features for models that cannot read sequences.

    7 derived summaries + per channel: mean_28d, std_28d, max_28d, mean_7d  (8*4 = 32)  => 39 features.
    """
    out = compute_workload_summaries(wl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for ci, c in enumerate(WORKLOAD_CHANNELS):
            x = wl[:, :, ci]
            out[f"{c}__mean_28d"] = np.nanmean(x, axis=1)
            out[f"{c}__std_28d"] = np.nanstd(x, axis=1)
            out[f"{c}__max_28d"] = np.nanmax(x, axis=1)
            out[f"{c}__mean_7d"] = np.nanmean(x[:, -7:], axis=1)
    return out


def aggregate_wellness_features(wb: np.ndarray) -> pd.DataFrame:
    """Per wellness variable: mean_7d, std_7d, last_day, slope_7d (linear trend per day) => 28 features."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for vi, v in enumerate(WELLNESS_VARS):
            x = wb[:, :, vi]
            out[f"{v}__mean_7d"] = np.nanmean(x, axis=1)
            out[f"{v}__std_7d"] = np.nanstd(x, axis=1)
            out[f"{v}__last_day"] = x[:, -1]
            out[f"{v}__slope_7d"] = _slope(x)
    return pd.DataFrame(out)


def static_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STATIC_NUMERIC + STATIC_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"Missing static columns: {missing}")
    return df[STATIC_NUMERIC + STATIC_CATEGORICAL].copy().reset_index(drop=True)


def model_input_columns() -> list[str]:
    """Every raw column a model may read. Used by the leakage check (target / ids are excluded)."""
    return STATIC_NUMERIC + STATIC_CATEGORICAL + wl_columns() + wb_columns()
