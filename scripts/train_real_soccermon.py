"""Automatically fetch SoccerMon subjective data and train real-data baselines.

Usage:
  python scripts/train_real_soccermon.py

Only the small public subjective.zip (~706 kB) is fetched. The 99 GB GPS archive is
not downloaded. This script intentionally refuses to invent missing modalities.
"""
from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

RECORD_API = "https://zenodo.org/api/records/10033832"
PLAYER_ALIASES = ("player_id", "playerid", "player_uuid", "playeruuid", "athlete_id", "athleteid", "uuid", "player")
DATE_ALIASES = ("timestamp", "datetime", "report_date", "date", "created_at", "start_time", "time")
IGNORE_PATHS = ("illness", "game_performance", "game-performance")


def normalize(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    # Prefer exact aliases, then stable substring matches.
    for alias in aliases:
        if alias in columns:
            return alias
    for col in columns:
        if any(alias in col for alias in aliases):
            return col
    return None


def json_to_frames(path: Path) -> list[pd.DataFrame]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    frames = []
    def visit(obj):
        if isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                frames.append(pd.json_normalize(obj, sep="_"))
            else:
                for item in obj:
                    visit(item)
        elif isinstance(obj, dict):
            # Extract record-like lists recursively; also try the object itself.
            if any(k in {normalize(x) for x in obj.keys()} for x in PLAYER_ALIASES):
                frames.append(pd.json_normalize(obj, sep="_"))
            for value in obj.values():
                if isinstance(value, (list, dict)):
                    visit(value)
    visit(raw)
    return frames


def read_frames(path: Path) -> list[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return [pd.read_csv(path)]
        if path.suffix.lower() in (".json", ".jsonl"):
            return json_to_frames(path)
        if path.suffix.lower() in (".parquet", ".pq"):
            return [pd.read_parquet(path)]
    except Exception as exc:
        print(f"Skip {path.name}: {exc}")
    return []


def download_archive() -> bytes:
    meta = requests.get(RECORD_API, timeout=30)
    meta.raise_for_status()
    record = meta.json()
    item = next((f for f in record.get("files", []) if f.get("key", "").lower() == "subjective.zip"), None)
    if not item:
        raise RuntimeError("Zenodo record did not expose subjective.zip.")
    url = item.get("links", {}).get("self") or item.get("links", {}).get("download")
    if not url:
        raise RuntimeError("Zenodo did not provide an archive URL.")
    size = item.get("size")
    print(f"Downloading public SoccerMon subjective.zip ({size:,} bytes) to temporary memory…")
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content


def prepare_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[pd.DataFrame] = []
    injury_rows: list[pd.DataFrame] = []
    schema = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = normalize(str(path.relative_to(root)))
        if any(x in rel for x in IGNORE_PATHS):
            continue
        for frame in read_frames(path):
            if frame.empty:
                continue
            frame.columns = [normalize(c) for c in frame.columns]
            cols = list(frame.columns)
            player_col = find_column(cols, PLAYER_ALIASES)
            date_col = find_column(cols, DATE_ALIASES)
            schema.append({"file": str(path.relative_to(root)), "columns": cols[:30], "rows": len(frame)})
            if not player_col or not date_col:
                continue
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")
            frame = frame.dropna(subset=[player_col, date_col]).copy()
            frame["player_id"] = frame[player_col].astype(str)
            frame["date"] = frame[date_col]
            if "injury" in rel:
                injury_rows.append(frame[["player_id", "date"]].drop_duplicates())
                continue
            # Metric files are often long format (one value column per file) or wide format.
            candidate_cols = [c for c in frame.columns if c not in {player_col, date_col, "player_id", "date"}
                              and not any(s in c for s in ("uuid", "player", "timestamp", "date", "time", "id"))]
            numeric_cols = []
            for c in candidate_cols:
                converted = pd.to_numeric(frame[c], errors="coerce")
                if converted.notna().sum() > 0:
                    frame[c] = converted
                    numeric_cols.append(c)
            if not numeric_cols:
                continue
            # Use the metric name in the path if the file has only a generic value column.
            generic = [c for c in numeric_cols if c in ("value", "score", "metric_value")]
            if generic and len(numeric_cols) == 1:
                metric_name = normalize(path.parent.name + "_" + path.stem)
                frame = frame.rename(columns={generic[0]: metric_name})
                numeric_cols = [metric_name]
            part = frame[["player_id", "date"] + numeric_cols].copy()
            feature_rows.append(part.groupby(["player_id", "date"], as_index=False)[numeric_cols].mean())

    Path("results/real_soccermon").mkdir(parents=True, exist_ok=True)
    (Path("results/real_soccermon") / "source_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"Inspected {len(schema)} tabular report tables.")
    if not feature_rows:
        raise RuntimeError("Could not map feature tables to player/date/numeric columns. "
                           "Inspect results/real_soccermon/source_schema.json and update the adapter.")
    if not injury_rows:
        raise RuntimeError("No injury report table with player/date fields was found. "
                           "Inspect results/real_soccermon/source_schema.json; refusing to fabricate labels.")
    features = pd.concat(feature_rows, ignore_index=True, sort=False)
    features = features.groupby(["player_id", "date"], as_index=False).mean(numeric_only=True)
    injuries = pd.concat(injury_rows, ignore_index=True).drop_duplicates()
    return features, injuries


def build_examples(features: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in features.columns if c not in ("player_id", "date")]
    if not numeric:
        raise RuntimeError("No numeric features available after parsing.")
    panels = []
    for player, grp in features.groupby("player_id"):
        grp = grp.sort_values("date").set_index("date")
        # Calendar-day panel; features are rolling summaries of observed reports only.
        daily = grp[numeric].resample("D").mean()
        for col in numeric:
            daily[f"{col}_mean_7d"] = daily[col].rolling(7, min_periods=1).mean()
            daily[f"{col}_mean_28d"] = daily[col].rolling(28, min_periods=1).mean()
            daily[f"{col}_sum_7d"] = daily[col].rolling(7, min_periods=1).sum()
        daily["player_id"] = str(player)
        daily = daily.reset_index(names="date")
        first_observation = pd.to_datetime(grp.index.min()) + pd.Timedelta(days=27)
        daily = daily[daily["date"] >= first_observation]
        panels.append(daily)
    panel = pd.concat(panels, ignore_index=True)
    events = {str(p): np.sort(pd.to_datetime(g["date"]).values.astype("datetime64[ns]"))
              for p, g in injuries.groupby("player_id")}
    y = []
    for row in panel[["player_id", "date"]].itertuples(index=False):
        dates = events.get(str(row.player_id), np.array([], dtype="datetime64[ns]"))
        day = np.datetime64(pd.Timestamp(row.date).to_datetime64())
        # Prediction after day t: any reported injury in the next seven days (t+1 to t+7).
        y.append(int(((dates > day) & (dates <= day + np.timedelta64(7, "D"))).any()))
    panel["injury_next_7_days"] = y
    return panel.replace([np.inf, -np.inf], np.nan)


def temporal_split(df: pd.DataFrame):
    dates = np.sort(df["date"].unique())
    if len(dates) < 10:
        raise RuntimeError("Too few unique dates for a chronological train/validation/test split.")
    a, b = dates[int(len(dates) * .70)], dates[int(len(dates) * .85)]
    return df[df.date < a].copy(), df[(df.date >= a) & (df.date < b)].copy(), df[df.date >= b].copy()


def main():
    archive = download_archive()
    with tempfile.TemporaryDirectory(prefix="soccermon-") as tmp:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            zf.extractall(tmp)
        features, injuries = prepare_tables(Path(tmp))
    data = build_examples(features, injuries)
    data.to_csv("results/real_soccermon/player_day_dataset.csv", index=False)
    if data["injury_next_7_days"].nunique() < 2:
        raise RuntimeError("Only one class in the constructed labels. Inspect source_schema.json and injury-date mapping.")
    train, val, test = temporal_split(data)
    excluded = {"player_id", "date", "injury_next_7_days"}
    cols = [c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]
    y_train = train["injury_next_7_days"].astype(int)
    if y_train.nunique() < 2 or test["injury_next_7_days"].nunique() < 2:
        raise RuntimeError("Train/test split does not contain both classes. Inspect label counts and consider a justified split adjustment.")
    output = []
    for name, model in [
        ("logistic_regression", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                               LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("random_forest", make_pipeline(SimpleImputer(strategy="median"),
                                        RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                               class_weight="balanced", random_state=42, n_jobs=-1))),
    ]:
        model.fit(train[cols], y_train)
        prob = model.predict_proba(test[cols])[:, 1]
        pred = (prob >= .5).astype(int)
        y_test = test["injury_next_7_days"].astype(int)
        row = {"model": name, "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
               "test_prevalence": float(y_test.mean()), "roc_auc": float(roc_auc_score(y_test, prob)),
               "pr_auc": float(average_precision_score(y_test, prob)),
               "precision_at_0_5": float(precision_score(y_test, pred, zero_division=0)),
               "recall_at_0_5": float(recall_score(y_test, pred, zero_division=0)),
               "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
               "brier": float(brier_score_loss(y_test, prob))}
        output.append(row)
        joblib.dump(model, f"results/real_soccermon/{name}.joblib")
        print(row)
    pd.DataFrame(output).to_csv("results/real_soccermon/metrics.csv", index=False)
    print("Saved metrics, models, parsed source schema and player-day table in results/real_soccermon/.")
    print("Scientific caution: labels are player-reported injuries, not clinically verified time-loss injuries.")
    print("Check results/real_soccermon/source_schema.json and confirm event-date mapping before reporting results.")


if __name__ == "__main__":
    main()
