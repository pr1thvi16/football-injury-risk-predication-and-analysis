"""Feature definitions for the paper pipeline (paper Sec. VI).

Load encoder inputs   : acute, chronic, acute:chronic ratio, monotony, latest and peak daily values,
                        day-to-day variability, early-window mean                          (8 features)
Wellness encoder inputs: for each item, mean of the past week, mean of the last 3 days, and last-3 minus week (3 x 4 = 12)
Static/medical inputs  : age, position, previous injuries (2 seasons), season minutes, BMI, strength asymmetry,
                        average rest days

Window conventions: the 28-day (load) / 7-day (wellness) windows end the day BEFORE the observation date.
Details the paper does not define are implementation assumptions (marked A below).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

POSITIONS = ["GK", "DEF", "MID", "FWD"]
STATIC_NUMERIC = ["age", "prev_injuries_2seasons", "season_minutes", "bmi", "strength_asymmetry", "avg_rest_days"]
STATIC_FEATURES = STATIC_NUMERIC + ["position"]
LOAD_FEATURES = ["acute_load", "chronic_load", "acwr", "monotony", "latest_daily_load", "peak_daily_load",
                 "daily_variability", "early_window_mean"]
WELLNESS_ITEMS = ["fatigue", "sleep", "soreness", "mood"]   # fatigue/soreness: higher = worse; sleep (hours), mood: higher = better
WELLNESS_FEATURES = [f"{i}_{s}" for i in WELLNESS_ITEMS for s in ("mean_week", "mean_last3", "diff_last3_week")]
TARGET = "injury_next_7_days"
ID_COLUMNS = ["observation_id", "player_id", "week_index", "observation_date", "data_source"]
MODALITY_FEATURES = {"static": STATIC_FEATURES, "load": LOAD_FEATURES, "wellness": WELLNESS_FEATURES}
MODALITY_LABEL = {"static": "Static & medical", "load": "Workload", "wellness": "Wellness"}
MONOTONY_CAP = 10.0


def load_features(load: np.ndarray) -> pd.DataFrame:
    """load: (N, 28) daily session load, oldest -> newest, NaN = not recorded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        acute = np.nanmean(load[:, -7:], axis=1)
        chronic = np.nanmean(load, axis=1)
        std7 = np.nanstd(load[:, -7:], axis=1, ddof=1)
        last7 = load[:, -7:]
        variability = np.nanmean(np.abs(np.diff(last7, axis=1)), axis=1)        # A: mean |day-to-day change|, last 7 d
        early = np.nanmean(load[:, :14], axis=1)                                 # A: first 14 days of the window
        peak = np.nanmax(last7, axis=1) if last7.shape[1] else np.full(len(load), np.nan)   # A: peak of last 7 days
    latest = load[:, -1]
    return pd.DataFrame({
        "acute_load": acute,
        "chronic_load": chronic,
        "acwr": acute / np.maximum(chronic, 1.0),                                # A: denominator floored at 1 AU
        "monotony": np.minimum(acute / np.maximum(std7, 1e-6), MONOTONY_CAP),    # Foster: mean / SD, capped at 10
        "latest_daily_load": latest,
        "peak_daily_load": peak,
        "daily_variability": variability,
        "early_window_mean": early,
    })


def wellness_features(wb: np.ndarray) -> pd.DataFrame:
    """wb: (N, 7, 4) daily [fatigue, sleep, soreness, mood], oldest -> newest, NaN = not recorded."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j, item in enumerate(WELLNESS_ITEMS):
            week = np.nanmean(wb[:, :, j], axis=1)
            last3 = np.nanmean(wb[:, -3:, j], axis=1)
            out[f"{item}_mean_week"] = week
            out[f"{item}_mean_last3"] = last3
            out[f"{item}_diff_last3_week"] = last3 - week
    return pd.DataFrame(out)
