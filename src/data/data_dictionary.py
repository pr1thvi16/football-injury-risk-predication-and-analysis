"""Single source of truth for the data dictionary (used by docs and the dashboard)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ROOT
from src.features.feature_engineering import (
    LOAD_IDX, WELLNESS_VARS, WORKLOAD_CHANNELS, SUMMARY_FEATURES,
)

_E = lambda name, group, typ, unit, rng, desc, direction="": dict(
    feature=name, group=group, type=typ, unit=unit, valid_range=rng, description=desc, direction=direction
)

STATIC = [
    _E("observation_id", "Identifier", "string", "-", "unique", "Unique id of the player-week record."),
    _E("player_id", "Identifier", "string", "-", "P001-P250", "Stable synthetic player identifier."),
    _E("week_index", "Identifier", "int", "week", "0-47", "Index of the weekly assessment."),
    _E("observation_date", "Identifier", "date", "-", "2023-08-07 onward", "Assessment date (Monday). History windows end the day before; the target covers the next 7 days."),
    _E("data_source", "Identifier", "string", "-", "SYNTHETIC", "Always SYNTHETIC - not real club data."),
    _E("age", "Static & medical", "int", "years", "17-40", "Player age."),
    _E("position", "Static & medical", "category", "-", "GK, CB, FB, CM, W, ST", "Playing position (goalkeeper, centre-back, full-back, central midfielder, winger, striker)."),
    _E("height_cm", "Static & medical", "float", "cm", "155-210", "Height."),
    _E("weight_kg", "Static & medical", "float", "kg", "55-110", "Body mass."),
    _E("bmi", "Static & medical", "float", "kg/m^2", "17-30", "weight_kg / (height_cm/100)^2."),
    _E("previous_injury_count", "Static & medical", "int", "count", "0-10", "Number of previous time-loss injuries."),
    _E("previous_injury_days_missed", "Static & medical", "int", "days", "0-600", "Total days missed through previous injuries."),
    _E("days_since_last_injury", "Static & medical", "int", "days", "0-730", "Days since the last injury; capped at 730, players without injury = 730."),
    _E("season_minutes_played", "Static & medical", "int", "minutes", "0-8000", "Cumulative match minutes up to the observation date (simulation period)."),
    _E("strength_asymmetry_percent", "Static & medical", "float", "%", "0-40", "Inter-limb strength asymmetry. Higher = more asymmetric.", "higher = worse"),
    _E("average_rest_days", "Static & medical", "float", "days/week", "0-7", "Mean rest days per week over the 28-day window (days with zero training)."),
]

WORKLOAD_DESC = {
    "total_distance_km": ("km", "0-20", "Total distance covered per day (training + match)."),
    "high_speed_running_distance_km": ("km", "0-5", "Distance above the high-speed threshold per day."),
    "sprint_distance_km": ("km", "0-2", "Sprint distance per day."),
    "acceleration_count": ("count", "0-140", "Number of accelerations per day."),
    "deceleration_count": ("count", "0-150", "Number of decelerations per day."),
    "training_duration_minutes": ("min", "0-130", "Session duration per day (0 = rest day)."),
    "session_rating_of_perceived_exertion": ("CR-10 scale", "0-10", "Session RPE; 0 only on rest days."),
    "session_training_load": ("AU", "0-1300", "session_rpe x duration_minutes."),
}
WELLNESS_DESC = {
    "sleep_hours": ("hours", "3.5-10.5", "Night sleep duration.", "higher = better"),
    "sleep_quality": ("1-10", "1-10", "Self-rated sleep quality.", "higher = better"),
    "fatigue_score": ("1-10", "1-10", "Self-rated fatigue.", "higher = worse"),
    "muscle_soreness_score": ("1-10", "1-10", "Self-rated muscle soreness.", "higher = worse"),
    "mood_score": ("1-10", "1-10", "Self-rated mood.", "higher = better"),
    "recovery_score": ("0-100", "0-100", "Self-rated overall recovery.", "higher = better"),
    "perceived_stress_score": ("1-10", "1-10", "Self-rated stress.", "higher = worse"),
}
SUMMARY_DESC = {
    "acute_workload": ("AU/day", "mean session load, last 7 days"),
    "chronic_workload": ("AU/day", "mean session load, last 28 days"),
    "acute_chronic_workload_ratio": ("ratio", "acute / max(chronic, 1). A candidate feature only, not a definitive predictor."),
    "workload_monotony": ("ratio", "mean_7d / std_7d, capped at 10"),
    "workload_strain": ("AU", "sum of load over 7 days x monotony"),
    "workload_variability": ("ratio", "coefficient of variation of 28-day load: std_28d / max(chronic, 1)"),
    "recent_workload_change": ("ratio", "(acute - mean of days 15-21 back) / max(that mean, 1)"),
}


def build_dictionary() -> pd.DataFrame:
    rows = list(STATIC)
    for f in SUMMARY_FEATURES:
        unit, desc = SUMMARY_DESC[f]
        rows.append(_E(f, "Workload summary (derived)", "float", unit, ">= 0", desc))
    for c in WORKLOAD_CHANNELS:
        unit, rng, desc = WORKLOAD_DESC[c]
        rows.append(_E(f"wl_{c}_d1 ... d28", "Workload series", "float", unit, rng, desc + " Columns d1 (oldest) to d28 (day before observation)."))
    for v in WELLNESS_VARS:
        unit, rng, desc, direction = WELLNESS_DESC[v]
        rows.append(_E(f"wb_{v}_d1 ... d7", "Wellness series", "float", unit, rng, desc + " Columns d1 (oldest) to d7 (day before observation).", direction))
    rows.append(_E("injury_next_7_days", "Target", "int", "0/1", "{0, 1}",
                   "1 = a simulated time-loss injury occurs in the 7 days starting on the observation date."))
    return pd.DataFrame(rows)


def write_markdown(path: Path | None = None) -> Path:
    path = path or ROOT / "docs" / "data_dictionary.md"
    df = build_dictionary()
    lines = ["# Data dictionary", "",
             "All values are **SYNTHETIC**. See `docs/methodology.md` for how they are simulated.", "",
             "| Feature | Group | Type | Unit | Valid range | Direction | Meaning |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| `{r.feature}` | {r.group} | {r.type} | {r.unit} | {r.valid_range} | {r.direction or '-'} | {r.description} |")
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    print(write_markdown())
