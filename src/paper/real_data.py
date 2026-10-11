"""Shared player-week builder: daily tables (SYNTHETIC or REAL) -> model-ready player-week records.

The synthetic generator and any real dataset go through exactly this code, so a model that works here can
be re-trained on real club data by supplying a mapping file (see config/real_data_mapping.example.yaml).

Input tables (any column names; mapped through ``mapping``):
  daily     one row per player-day: player id, date, session load (or rpe x duration), wellness items,
            optional minutes played
  injuries  one row per time-loss injury: player id, date          (label source; no other use)
  static    optional, one row per player: age, position, height, weight, strength asymmetry, previous injuries
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.paper.features import (
    POSITIONS, STATIC_NUMERIC, TARGET, WELLNESS_ITEMS, load_features, wellness_features,
)

DEFAULT_OPTIONS = {"history_days": 28, "wellness_days": 7, "horizon_days": 7, "stride_days": 7,
                   "missing_load": "nan",           # "nan": unrecorded days are unknown; "zero": treat as rest
                   "min_load_days": 14, "min_wellness_days": 4, "data_source": "REAL",
                   "derive_previous_injuries": False, "exclude_if_injured_at_obs": False}
CANON_WELLNESS_RANGE = {"fatigue": (1, 10), "sleep": (0, 14), "soreness": (1, 10), "mood": (1, 10)}


class MappingError(ValueError):
    pass


def _col(df: pd.DataFrame, name: str | None, table: str, required: bool = True):
    if name is None:
        if required:
            raise MappingError(f"Mapping for table '{table}' is missing a required column name")
        return None
    if name not in df.columns:
        raise MappingError(f"Column '{name}' (table '{table}') not found. Available: {list(df.columns)[:12]}")
    return df[name]


def standardise_daily(daily: pd.DataFrame, m: dict) -> pd.DataFrame:
    d = m["daily"]
    out = pd.DataFrame({"player_id": _col(daily, d["player_id"], "daily").astype(str),
                        "date": pd.to_datetime(_col(daily, d["date"], "daily"), errors="coerce")})
    if out["date"].isna().any():
        raise MappingError("Some daily dates could not be parsed")
    if d.get("load"):
        out["load"] = pd.to_numeric(_col(daily, d["load"], "daily"), errors="coerce")
    elif d.get("rpe") and d.get("duration_minutes"):
        out["load"] = (pd.to_numeric(_col(daily, d["rpe"], "daily"), errors="coerce")
                       * pd.to_numeric(_col(daily, d["duration_minutes"], "daily"), errors="coerce"))
    else:
        raise MappingError("daily mapping needs 'load' or both 'rpe' and 'duration_minutes'")
    scales = m.get("wellness_scales", {})
    invert = set(m.get("invert", []))
    for item in WELLNESS_ITEMS:
        src = d.get("wellness", {}).get(item)
        if not src:
            out[item] = np.nan
            continue
        v = pd.to_numeric(_col(daily, src, "daily"), errors="coerce")
        if item in scales:                                  # rescale source range -> canonical range
            lo, hi = scales[item]
            clo, chi = CANON_WELLNESS_RANGE[item]
            v = clo + (v - lo) * (chi - clo) / (hi - lo)
        if item in invert:
            clo, chi = CANON_WELLNESS_RANGE[item]
            v = clo + chi - v
        out[item] = v
    out["minutes_played"] = (pd.to_numeric(_col(daily, d["minutes_played"], "daily"), errors="coerce")
                             if d.get("minutes_played") else np.nan)
    if out.duplicated(["player_id", "date"]).any():
        out = out.groupby(["player_id", "date"], as_index=False).agg(
            {"load": "sum", **{i: "mean" for i in WELLNESS_ITEMS}, "minutes_played": "sum"})
    return out.sort_values(["player_id", "date"]).reset_index(drop=True)


def standardise_static(static: pd.DataFrame | None, m: dict) -> pd.DataFrame:
    if static is None:
        return pd.DataFrame({"player_id": []})
    s = m.get("static", {})
    out = pd.DataFrame({"player_id": _col(static, s["player_id"], "static").astype(str)})
    for canon in ("age", "height_cm", "weight_kg", "strength_asymmetry", "prev_injuries_2seasons"):
        out[canon] = pd.to_numeric(_col(static, s.get(canon), "static", required=False), errors="coerce") if s.get(canon) else np.nan
    if s.get("position"):
        pm = {str(k): v for k, v in m.get("position_map", {}).items()}
        raw = _col(static, s["position"], "static").astype(str)
        out["position"] = raw.map(pm) if pm else raw
        bad = sorted(set(out["position"].dropna()) - set(POSITIONS))
        if bad:
            raise MappingError(f"Unmapped position values {bad}; add them to position_map (targets: {POSITIONS})")
    else:
        out["position"] = np.nan
    return out.drop_duplicates("player_id").reset_index(drop=True)


def build_player_weeks(daily: pd.DataFrame, injuries: pd.DataFrame, static: pd.DataFrame | None, mapping: dict) -> pd.DataFrame:
    """Build one record per (player, weekly observation date). Features use ONLY days before the observation date."""
    opt = {**DEFAULT_OPTIONS, **mapping.get("options", {})}
    H, W, HZ, STRIDE = opt["history_days"], opt["wellness_days"], opt["horizon_days"], opt["stride_days"]
    if H != 28 or W != 7:
        raise MappingError("The paper pipeline uses a 28-day workload and 7-day wellness window")
    dd = standardise_daily(daily, mapping)
    st = standardise_static(static, mapping)
    ij = mapping["injuries"]
    inj = pd.DataFrame({"player_id": _col(injuries, ij["player_id"], "injuries").astype(str),
                        "date": pd.to_datetime(_col(injuries, ij["date"], "injuries"), errors="coerce")}).dropna()
    inj_by_player = {p: g["date"].sort_values().to_numpy() for p, g in inj.groupby("player_id")}
    st_idx = st.set_index("player_id") if len(st) else pd.DataFrame()
    rows = []
    for pid, g in dd.groupby("player_id", sort=True):
        g = g.set_index("date")
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)
        if opt["missing_load"] == "zero":
            g["load"] = g["load"].fillna(0.0)
        load = g["load"].to_numpy(float)
        wb = g[WELLNESS_ITEMS].to_numpy(float)
        mins = g["minutes_played"].to_numpy(float)
        n = len(full)
        inj_dates = inj_by_player.get(pid, np.array([], dtype="datetime64[ns]"))
        first = H                                              # need a full 28-day history before the first observation
        for k, end in enumerate(range(first, n - HZ + 1, STRIDE)):    # `end` = index of observation day; horizon fully inside data
            lw, ww = load[end - H:end], wb[end - W:end]
            if np.isfinite(lw).sum() < opt["min_load_days"] or np.isfinite(ww[:, 0]).sum() < min(opt["min_wellness_days"], W):
                continue
            obs = full[end]
            horizon_end = obs + pd.Timedelta(days=HZ - 1)
            label = int(((inj_dates >= np.datetime64(obs)) & (inj_dates <= np.datetime64(horizon_end))).any())
            if opt["exclude_if_injured_at_obs"] and ((inj_dates >= np.datetime64(obs - pd.Timedelta(days=7))) & (inj_dates < np.datetime64(obs))).any():
                continue
            prev_inj = int((inj_dates < np.datetime64(obs)).sum()) if opt["derive_previous_injuries"] else np.nan
            rows.append((pid, k, obs, lw, ww, float(np.nansum(mins[:end])) if np.isfinite(mins[:end]).any() else np.nan, prev_inj, label))
    if not rows:
        raise MappingError("No player-weeks could be built; check dates, history length and the missing-data thresholds")
    pids = [r[0] for r in rows]
    LW = np.stack([r[3] for r in rows]); WB = np.stack([r[4] for r in rows])
    df = pd.DataFrame({"player_id": pids, "week_index": [r[1] for r in rows],
                       "observation_date": [r[2].strftime("%Y-%m-%d") for r in rows],
                       "season_minutes": [r[5] for r in rows], "_prev_derived": [r[6] for r in rows]})
    df.insert(0, "observation_id", [f"OBS{i:06d}" for i in range(len(df))])
    df["data_source"] = opt["data_source"]
    if len(st_idx):
        for c in ("age", "height_cm", "weight_kg", "strength_asymmetry", "prev_injuries_2seasons", "position"):
            df[c] = df["player_id"].map(st_idx[c]) if c in st_idx.columns else np.nan
    else:
        for c in ("age", "height_cm", "weight_kg", "strength_asymmetry", "prev_injuries_2seasons", "position"):
            df[c] = np.nan
    df["prev_injuries_2seasons"] = df["prev_injuries_2seasons"].fillna(df["_prev_derived"])
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
    df["avg_rest_days"] = np.where(np.isfinite(LW).sum(1) > 0, (np.nan_to_num(LW, nan=1.0) == 0).sum(1) / 4.0, np.nan)
    feats = pd.concat([load_features(LW), wellness_features(WB)], axis=1)
    out = pd.concat([df.drop(columns=["_prev_derived", "height_cm", "weight_kg"]), feats], axis=1)
    out[TARGET] = [r[7] for r in rows]
    cols = ["observation_id", "player_id", "week_index", "observation_date", "data_source"] + STATIC_NUMERIC + ["position"] + list(feats.columns) + [TARGET]
    out = out[cols]
    out.attrs["available_static"] = [c for c in STATIC_NUMERIC + ["position"] if out[c].notna().any()]
    return out
