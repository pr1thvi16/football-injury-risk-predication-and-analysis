"""Dataset validation, data-quality report and input validation for inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features.feature_engineering import (
    ID_COLUMNS, POSITIONS, STATIC_CATEGORICAL, STATIC_NUMERIC, SUMMARY_FEATURES, TARGET, WELLNESS_VARS,
    WORKLOAD_CHANNELS, model_input_columns, sequences_from_frame, wb_columns, wl_columns,
    compute_workload_summaries,
)

WELLNESS_RANGES = {"sleep_hours": (0, 14), "sleep_quality": (1, 10), "fatigue_score": (1, 10),
                   "muscle_soreness_score": (1, 10), "mood_score": (1, 10), "recovery_score": (0, 100),
                   "perceived_stress_score": (1, 10)}
FORBIDDEN_FEATURE_HINTS = ("prob", "latent", "hidden", "proneness", "logit", "noise", "future", "next_7")


class DataValidationError(ValueError):
    """Raised when the dataset/input fails validation."""


def _check(name, passed, detail=""):
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def validate_dataframe(df: pd.DataFrame, expected_records: int | None = 12000, expected_prevalence: float | None = 0.0898,
                       prevalence_tolerance: float = 0.002) -> list[dict]:
    """Run all dataset checks. Returns a list of {check, passed, detail}."""
    res = []
    required = ID_COLUMNS + STATIC_NUMERIC + STATIC_CATEGORICAL + SUMMARY_FEATURES + wl_columns() + wb_columns() + [TARGET]
    missing = [c for c in required if c not in df.columns]
    res.append(_check("schema: all required columns present", not missing, f"missing={missing[:5]}"))
    if missing:
        return res
    if expected_records is not None:
        res.append(_check(f"record count == {expected_records}", len(df) == expected_records, f"n={len(df)}"))
    res.append(_check("observation_id unique", df["observation_id"].is_unique))
    res.append(_check("player_id valid (P###)", df["player_id"].astype(str).str.fullmatch(r"P\d{3}").all()))
    n_missing = int(df[required].isna().sum().sum())
    res.append(_check("no missing values", n_missing == 0, f"missing cells={n_missing}"))
    res.append(_check("no duplicate records", not df.drop(columns=["observation_id"]).duplicated().any(),
                      f"duplicates={int(df.drop(columns=['observation_id']).duplicated().sum())}"))
    res.append(_check("no duplicate (player, week) pairs", not df.duplicated(["player_id", "week_index"]).any()))
    neg_cols = [c for c in STATIC_NUMERIC + SUMMARY_FEATURES if c != "recent_workload_change" and (df[c] < 0).any()]
    seq_wl_neg = (df[wl_columns()].to_numpy() < 0).any()
    res.append(_check("no impossible negative measurements", not neg_cols and not seq_wl_neg, f"cols={neg_cols}"))
    res.append(_check("position values valid", df["position"].isin(POSITIONS).all(),
                      f"unique={sorted(df['position'].unique())}"))
    bad_well = [v for v in WELLNESS_VARS
                if ((df[[f"wb_{v}_d{k}" for k in range(1, 8)]] < WELLNESS_RANGES[v][0]).any().any()
                    or (df[[f"wb_{v}_d{k}" for k in range(1, 8)]] > WELLNESS_RANGES[v][1]).any().any())]
    res.append(_check("wellness scores within documented scales", not bad_well, f"bad={bad_well}"))
    res.append(_check("target encoded as {0,1}", set(df[TARGET].unique()) <= {0, 1}, f"values={sorted(int(v) for v in df[TARGET].unique())}"))
    prev = float(df[TARGET].mean())
    if expected_prevalence is not None:
        res.append(_check(f"target prevalence ~ {expected_prevalence:.2%}", abs(prev - expected_prevalence) <= prevalence_tolerance,
                          f"prevalence={prev:.4%}, positives={int(df[TARGET].sum())}"))
    dates = pd.to_datetime(df["observation_date"], errors="coerce")
    res.append(_check("observation dates valid", dates.notna().all() and dates.min() >= pd.Timestamp("2000-01-01")
                      and dates.max() <= pd.Timestamp("2100-01-01"), f"{dates.min().date()} -> {dates.max().date()}"))
    res.append(_check("data_source == SYNTHETIC for every row", (df["data_source"] == "SYNTHETIC").all()))
    wl, _ = sequences_from_frame(df)
    summ = compute_workload_summaries(wl)
    res.append(_check("derived workload features match their documented formulas",
                      np.allclose(summ.values, df[SUMMARY_FEATURES].to_numpy(), atol=1e-2, rtol=1e-3)))
    load_ok = np.allclose(wl[:, :, WORKLOAD_CHANNELS.index("session_training_load")],
                          wl[:, :, WORKLOAD_CHANNELS.index("session_rating_of_perceived_exertion")]
                          * wl[:, :, WORKLOAD_CHANNELS.index("training_duration_minutes")], atol=1e-2, rtol=1e-3)
    res.append(_check("session_training_load == session_rpe * duration", load_ok))
    inputs = model_input_columns()
    leak = [c for c in inputs if TARGET in c or any(h in c.lower() for h in FORBIDDEN_FEATURE_HINTS)]
    res.append(_check("feature/target separation (no target-like column in model inputs)",
                      TARGET not in inputs and not leak and not set(inputs) & set(ID_COLUMNS), f"suspicious={leak}"))
    return res


def build_report(df: pd.DataFrame, results: list[dict]) -> dict:
    return {
        "data_label": "SYNTHETIC - not real club data",
        "n_records": int(len(df)),
        "n_players": int(df["player_id"].nunique()),
        "n_positive": int(df[TARGET].sum()),
        "prevalence": float(df[TARGET].mean()),
        "date_range": [str(df["observation_date"].min()), str(df["observation_date"].max())],
        "all_passed": all(r["passed"] for r in results),
        "checks": results,
    }


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "data_quality_report.json", out_dir / "data_quality_report.md"
    jp.write_text(json.dumps(report, indent=2))
    lines = ["# Data-quality report (SYNTHETIC dataset)", "",
             f"- Records: **{report['n_records']:,}**  |  Players: **{report['n_players']}**",
             f"- Positives: **{report['n_positive']:,}** ({report['prevalence']:.2%})",
             f"- Date range: {report['date_range'][0]} -> {report['date_range'][1]}",
             f"- Overall: **{'PASS' if report['all_passed'] else 'FAIL'}**", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in report["checks"]]
    mp.write_text("\n".join(lines) + "\n")
    return jp, mp


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Generate it first with:  python -m src.data.generate_dataset")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # corrupted / unreadable CSV
        raise DataValidationError(f"Could not read dataset {path}: {exc}") from exc


# ----------------------------------------------------------------------------- inference-time input checks
def validate_inference_frame(df: pd.DataFrame) -> None:
    """Raise DataValidationError with a clear message if rows cannot be scored."""
    if df is None or len(df) == 0:
        raise DataValidationError("No input rows supplied.")
    required = STATIC_NUMERIC + STATIC_CATEGORICAL + wl_columns() + wb_columns()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Input is missing {len(missing)} required columns, e.g. {missing[:4]}")
    num = df[STATIC_NUMERIC + wl_columns() + wb_columns()]
    if not all(pd.api.types.is_numeric_dtype(t) for t in num.dtypes):
        raise DataValidationError("Non-numeric values found in numeric input columns.")
    if not np.isfinite(num.to_numpy(float)).all():
        raise DataValidationError("Input contains NaN/inf values; supply complete 28-day workload and 7-day wellness histories.")
    if (df[wl_columns()].to_numpy() < 0).any():
        raise DataValidationError("Workload measurements must be non-negative.")
    bad_pos = sorted(set(df["position"].astype(str)) - set(POSITIONS))
    if bad_pos:
        raise DataValidationError(f"Unknown position(s) {bad_pos}; valid: {POSITIONS}")
    for v, (lo, hi) in WELLNESS_RANGES.items():
        vals = df[[f"wb_{v}_d{k}" for k in range(1, 8)]].to_numpy(float)
        if (vals < lo).any() or (vals > hi).any():
            raise DataValidationError(f"Wellness variable '{v}' must be within [{lo}, {hi}].")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the synthetic dataset and write a data-quality report.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    path = Path(args.input) if args.input else resolve_path(cfg, "dataset")
    df = load_dataset(path)
    results = validate_dataframe(df, expected_records=cfg["data"]["n_players"] * cfg["data"]["n_weeks"],
                                 expected_prevalence=cfg["data"]["prevalence"])
    report = build_report(df, results)
    jp, mp = write_report(report, resolve_path(cfg, "metrics_dir"))
    for r in results:
        print(("PASS " if r["passed"] else "FAIL ") + r["check"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    print(f"\nReport written to {jp} and {mp}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
