"""Train real-data baselines on SoccerMon subjective reports.

Downloads only the small public subjective archive from Zenodo automatically. This is
separate from the synthetic-data experiment in src/training/train.py. It creates
player-day examples using reports available up to the prediction date and labels a
positive when a player-reported injury is recorded in the next 7 days.

Run: python scripts/train_soccermon.py
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

ZENODO_RECORD = "https://zenodo.org/api/records/10033832"
PLAYER_KEYS = ("player_id", "playerid", "athlete_id", "athleteid", "player")
DATE_KEYS = ("date", "datetime", "timestamp", "report_date", "created_at", "start_time")
INJURY_HINTS = ("injury",)
IGNORE_HINTS = ("illness", "game-performance", "game_performance")


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if path.suffix.lower() == ".json":
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                return pd.read_json(path)
    except Exception as exc:
        print(f"Skipping unreadable file {path.name}: {exc}")
    return None


def choose_col(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    for col in columns:
        if any(alias in col for alias in aliases):
            return col
    return None


def download_subjective_archive() -> bytes:
    response = requests.get(ZENODO_RECORD, timeout=30)
    response.raise_for_status()
    record = response.json()
    files = record.get("files", [])
    item = next((f for f in files if "subjective.zip" in f.get("key", "").lower()), None)
    if item is None:
        # Some Zenodo record versions expose the archive in a nested file list.
        item = next((f for f in files if "subjective" in f.get("key", "").lower() and
                     f.get("key", "").lower().endswith(".zip")), None)
    if item is None:
        raise RuntimeError("Could not find the subjective ZIP in the Zenodo record. "
                           "Inspect https://zenodo.org/records/10033832 and update the archive selector.")
    url = item.get("links", {}).get("self") or item.get("links", {}).get("download")
    if not url:
        raise RuntimeError("Zenodo did not return a download URL for the subjective archive.")
    print(f"Fetching {item.get('key')} ({item.get('size', 'unknown')} bytes) from Zenodo…")
    data = requests.get(url, timeout=120)
    data.raise_for_status()
    return data.content


def load_reports(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily numeric feature table, injury event table) from available reports."""
    daily_parts: list[pd.DataFrame] = []
    injury_parts: list[pd.DataFrame] = []
    found = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        table = read_table(path)
        if table is None or table.empty:
            continue
        found += 1
        table.columns = [norm(c) for c in table.columns]
        cols = list(table.columns)
        player_col = choose_col(cols, PLAYER_KEYS)
        date_col = choose_col(cols, DATE_KEYS)
        if player_col is None or date_col is None:
            print(f"Schema review needed: {path.relative_to(root)} columns={cols[:15]}")
            continue

        table[date_col] = pd.to_datetime(table[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")
        table = table.dropna(subset=[player_col, date_col]).copy()
        if table.empty:
            continue
        file_hint = norm(str(path.relative_to(root)))
        is_injury = any(h in file_hint for h in INJURY_HINTS)

        if is_injury:
            injury_parts.append(table[[player_col, date_col]].rename(
                columns={player_col: "player_id", date_col: "date"}).drop_duplicates())
            continue
        if any(h in file_hint for h in IGNORE_HINTS):
            continue

        # Wide report files: each numeric column becomes a feature.
        numeric_cols = [c for c in cols if c not in (player_col, date_col) and
                        pd.api.types.is_numeric_dtype(table[c])]
        # If reports use a single generic numeric 'value' field, identify the metric by filename.
        if numeric_cols == ["value"] or (len(numeric_cols) == 1 and numeric_cols[0] in ("value", "score")):
            table = table.rename(columns={numeric_cols[0]: norm(path.stem)})
            numeric_cols = [norm(path.stem)]
        if not numeric_cols:
            print(f"No numeric features found in {path.relative_to(root)}")
            continue
        part = table[[player_col, date_col] + numeric_cols].rename(
            columns={player_col: "player_id", date_col: "date"})
        part["player_id"] = part["player_id"].astype(str)
        daily_parts.append(part.groupby(["player_id", "date"], as_index=False)[numeric_cols].mean())

    print(f"Read {found} tabular files; feature tables={len(daily_parts)}, injury tables={len(injury_parts)}")
    if not daily_parts:
        raise RuntimeError("No usable feature reports were detected. Review the printed file schemas; "
                           "the archive schema may need a dataset-specific adapter.")
    if not injury_parts:
        raise RuntimeError("No injury event table was detected. Refusing to train without real injury labels.")
    features = pd.concat(daily_parts, ignore_index=True, sort=False)
    features = features.groupby(["player_id", "date"], as_index=False).mean(numeric_only=True)
    injuries = pd.concat(injury_parts, ignore_index=True).drop_duplicates()
    injuries["player_id"] = injuries["player_id"].astype(str)
    return features, injuries


def make_player_days(features: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in features.columns if c not in ("player_id", "date")]
    features = features.sort_values(["player_id", "date"]).copy()
    # Rolling inputs only use observations at or before the current date.
    engineered = []
    for player_id, group in features.groupby("player_id", sort=False):
        group = group.sort_values("date").set_index("date")
        daily = group[feature_cols].resample("D").mean()
        daily["player_id"] = player_id
        for col in feature_cols:
            daily[f"{col}_mean_7d"] = daily[col].rolling(7, min_periods=1).mean()
            daily[f"{col}_mean_28d"] = daily[col].rolling(28, min_periods=1).mean()
            daily[f"{col}_sum_7d"] = daily[col].rolling(7, min_periods=1).sum()
            daily[f"{col}_sum_28d"] = daily[col].rolling(28, min_periods=1).sum()
        engineered.append(daily.reset_index(names="date"))
    panel = pd.concat(engineered, ignore_index=True)
    events = {pid: np.sort(pd.to_datetime(g["date"]).values.astype("datetime64[ns]"))
              for pid, g in injuries.groupby("player_id")}
    labels = []
    for row in panel[["player_id", "date"]].itertuples(index=False):
        dates = events.get(str(row.player_id), np.array([], dtype="datetime64[ns]"))
        start = np.datetime64(pd.Timestamp(row.date).to_datetime64()) + np.timedelta64(1, "D")
        end = np.datetime64(pd.Timestamp(row.date).to_datetime64()) + np.timedelta64(8, "D")
        labels.append(int(((dates >= start) & (dates < end)).any()))
    panel["injury_next_7_days"] = labels
    panel = panel.replace([np.inf, -np.inf], np.nan)
    # Remove the first 28 days for each player to ensure a full history window.
    first_dates = features.groupby("player_id")["date"].min().add(pd.Timedelta(days=27))
    panel = panel[panel.apply(lambda r: r["date"] >= first_dates.get(r["player_id"], r["date"]), axis=1)]
    return panel.dropna(subset=["injury_next_7_days"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["date"].unique())
    if len(dates) < 10:
        raise RuntimeError("Not enough distinct dates for temporal evaluation.")
    train_end, val_end = dates[int(len(dates) * 0.70)], dates[int(len(dates) * 0.85)]
    return (df[df.date < train_end].copy(),
            df[(df.date >= train_end) & (df.date < val_end)].copy(),
            df[df.date >= val_end].copy())


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "42")
    archive = download_subjective_archive()
    with tempfile.TemporaryDirectory(prefix="soccermon-") as temp:
        root = Path(temp)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            zf.extractall(root)
        features, injuries = load_reports(root)
    dataset = make_player_days(features, injuries)
    if dataset["injury_next_7_days"].nunique() < 2:
        raise RuntimeError("The constructed label has only one class. Check injury date parsing and report schema.")
    train, val, test = temporal_split(dataset)
    feature_cols = [c for c in dataset.columns if c not in ("player_id", "date", "injury_next_7_days")]
    y_train = train["injury_next_7_days"].astype(int)
    if y_train.nunique() < 2:
        raise RuntimeError("Training period has only one target class; adjust temporal split after inspecting event dates.")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    out_dir = Path("results/real_soccermon")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for name, model in [
        ("logistic_regression", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                               LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("random_forest", make_pipeline(SimpleImputer(strategy="median"),
                                        RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                               min_samples_leaf=3, random_state=42, n_jobs=-1))),
    ]:
        model.fit(train[feature_cols], y_train)
        prob = model.predict_proba(test[feature_cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        row = {
            "model": name, "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "test_positive_rate": float(test["injury_next_7_days"].mean()),
            "precision_at_0_5": float(precision_score(test["injury_next_7_days"], pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(test["injury_next_7_days"], pred, zero_division=0)),
            "f1_at_0_5": float(f1_score(test["injury_next_7_days"], pred, zero_division=0)),
            "pr_auc": float(average_precision_score(test["injury_next_7_days"], prob)),
            "roc_auc": float(roc_auc_score(test["injury_next_7_days"], prob)) if test["injury_next_7_days"].nunique() == 2 else None,
        }
        report.append(row)
        import joblib
        joblib.dump(model, out_dir / f"{name}.joblib")
        print(row)

    pd.DataFrame(report).to_csv(out_dir / "metrics.csv", index=False)
    dataset.to_csv(out_dir / "player_day_dataset.csv", index=False)
    print(f"Saved results to {out_dir.resolve()}")
    print("IMPORTANT: labels are player-reported injury events, not medically verified diagnoses.")
    print("Review the printed schema warnings and verify event-date alignment before interpreting results.")


if __name__ == "__main__":
    main()
