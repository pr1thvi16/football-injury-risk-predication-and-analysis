"""Paper-faithful SYNTHETIC data generator (paper Sec. VII-A), emitting the same three tables a club would supply.

Outputs (all SYNTHETIC): daily table (load + wellness + minutes per player-day), injuries table (time-loss dates)
and a static table.  Player-weeks are then built by ``real_data.build_player_weeks`` - the same code real data uses.

Mechanism (assumptions, the paper gives only the outline): player-level static risk factors dominate the label
(consistent with the paper's finding that static features carry most signal), workload spikes/drops (ACWR) and poor
sleep/high soreness add risk, plus a hidden player trait and noise.  The positive count is enforced exactly.
Injuries do not feed back into later load (simplification).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.paper.features import POSITIONS, WELLNESS_ITEMS
from src.paper.real_data import build_player_weeks

DOW = np.array([0.0, 0.8, 1.0, 0.85, 0.5, 0.3, 1.0])     # per-weekday load template (day 0 = rest); day 6 = match
POS_COUNTS = np.array([572, 590, 605, 633], float)          # paper position mix (GK, DEF, MID, FWD)

MAPPING = {   # identity mapping for the generator's own tables
    "daily": {"player_id": "player_id", "date": "date", "load": "session_load", "minutes_played": "minutes_played",
              "wellness": {i: i for i in WELLNESS_ITEMS}},
    "injuries": {"player_id": "player_id", "date": "injury_date"},
    "static": {"player_id": "player_id", "age": "age", "height_cm": "height_cm", "weight_kg": "weight_kg",
               "strength_asymmetry": "strength_asymmetry", "prev_injuries_2seasons": "prev_injuries_2seasons",
               "position": "position"},
    "options": {"data_source": "SYNTHETIC"},
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_tables(n_players=250, n_weeks=48, prevalence=0.0898, start_date="2023-08-07", signal_strength=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n_days = 28 + 7 * n_weeks
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    # --- static -----------------------------------------------------------------------------------------------
    pos = rng.choice(len(POSITIONS), n_players, p=POS_COUNTS / POS_COUNTS.sum())
    age = np.clip(rng.normal(26.5, 4.2, n_players), 17, 38).round(1)
    height = np.clip(rng.normal(180, 6.5, n_players) + (pos == 0) * 4, 160, 202).round(1)
    weight = np.clip((height - 100) * rng.normal(0.9, 0.06, n_players), 55, 100).round(1)
    prev_inj = rng.poisson(0.9, n_players)
    asym = np.clip(rng.gamma(4.0, 2.2, n_players) + 0.4 * prev_inj, 0, 35).round(1)
    base_load = np.clip(rng.normal(430, 70, n_players), 250, 650)
    resil = rng.normal(0, 1, n_players)                                  # hidden trait (never exported)
    static = pd.DataFrame({"player_id": [f"P{i:03d}" for i in range(n_players)], "age": age, "height_cm": height,
                           "weight_kg": weight, "strength_asymmetry": asym, "prev_injuries_2seasons": prev_inj,
                           "position": [POSITIONS[p] for p in pos]})
    # --- daily series -----------------------------------------------------------------------------------------
    rows = []
    for i in range(n_players):
        n_blocks = int(np.ceil(n_days / 7))
        block = np.where(rng.random(n_blocks) < 0.5, rng.uniform(1.15, 1.8, n_blocks), rng.uniform(0.6, 1.0, n_blocks))
        block = np.repeat(block, 7)[:n_days]
        dow = np.roll(DOW, rng.integers(0, 7))[np.arange(n_days) % 7]
        extra_rest = rng.random(n_days) < 0.04
        load = base_load[i] * dow * block * rng.lognormal(0, 0.12, n_days)
        load[extra_rest] = 0.0
        load = np.round(load, 0)
        minutes = np.where(dow >= 1.0, np.clip(rng.normal(70, 22, n_days), 0, 100), 0).round(0)
        minutes[load == 0] = 0
        rel = pd.Series(load / base_load[i]).ewm(alpha=0.25).mean().to_numpy()      # recent relative load
        fatigue = np.clip(3.2 + 2.6 * (rel - 0.6) + rng.normal(0, 0.9, n_days) - 0.4 * resil[i] * 0.3, 1, 10)
        sleep = np.clip(7.8 - 0.7 * (rel - 0.6) + rng.normal(0, 0.8, n_days), 3, 11)
        soreness = np.clip(3.0 + 2.4 * (rel - 0.6) + rng.normal(0, 1.0, n_days), 1, 10)
        mood = np.clip(7.0 - 0.9 * (rel - 0.6) + rng.normal(0, 0.9, n_days), 1, 10)
        # sporadic missing wellness entries (~3 %)
        miss = rng.random(n_days) < 0.03
        w = {k: np.where(miss, np.nan, v.round(1)) for k, v in zip(WELLNESS_ITEMS, (fatigue, sleep, soreness, mood))}
        rows.append(pd.DataFrame({"player_id": static.player_id[i], "date": dates, "session_load": load,
                                  "minutes_played": minutes, **w}))
    daily = pd.concat(rows, ignore_index=True)
    # --- label mechanism on player-weeks ----------------------------------------------------------------------
    weeks = build_player_weeks(daily, pd.DataFrame({"player_id": [], "injury_date": []}), static, MAPPING)
    idx = static.set_index("player_id")
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    logit = (0.55 * z(weeks.age) + 0.65 * z(weeks.prev_injuries_2seasons) + 0.45 * z(weeks.strength_asymmetry)
             + 0.30 * z(weeks.season_minutes) + 0.25 * z(weeks.avg_rest_days.clip(upper=3)) * -1
             + 0.50 * (weeks.acwr > 1.3) + 0.30 * (weeks.acwr < 0.8)
             + 0.25 * (weeks.acwr > 1.3) * (z(weeks.fatigue_mean_last3) > 0.5)
             + 0.30 * z(-weeks.sleep_mean_last3) + 0.30 * z(weeks.soreness_mean_last3.fillna(weeks.soreness_mean_last3.mean())))
    logit = signal_strength * logit + 0.8 * weeks.player_id.map(dict(zip(static.player_id, resil))).to_numpy()
    n_pos = int(round(prevalence * len(weeks)))
    noisy = logit.to_numpy() + rng.logistic(0, 1.6, len(weeks))
    pos_idx = np.argsort(-noisy)[:n_pos]
    obs = pd.to_datetime(weeks.observation_date).to_numpy()
    inj_rows = [(weeks.player_id.iloc[j], pd.Timestamp(obs[j]) + pd.Timedelta(days=int(rng.integers(0, 7)))) for j in pos_idx]
    injuries = pd.DataFrame(inj_rows, columns=["player_id", "injury_date"]).sort_values(["player_id", "injury_date"])
    injuries["injury_date"] = injuries["injury_date"].dt.strftime("%Y-%m-%d")
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily, injuries, static


def main():
    ap = argparse.ArgumentParser(description="Generate the paper-style SYNTHETIC dataset")
    ap.add_argument("--config", default="config/paper.yaml")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = a.seed if a.seed is not None else cfg["seed"]
    d = cfg["data"]
    daily, inj, static = generate_tables(d["n_players"], d["n_weeks"], d["prevalence"], d["start_date"], d["signal_strength"], seed)
    p = cfg["paths"]
    Path(p["daily"]).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(p["daily"], index=False); inj.to_csv(p["injuries"], index=False); static.to_csv(p["static"], index=False)
    weeks = build_player_weeks(daily, inj, static, MAPPING)
    weeks.to_csv(p["weeks"], index=False)
    print(f"SYNTHETIC: {len(weeks)} player-weeks, {int(weeks.injury_next_7_days.sum())} positives "
          f"({weeks.injury_next_7_days.mean():.2%}), {weeks.player_id.nunique()} players -> {p['weeks']}")


if __name__ == "__main__":
    main()
