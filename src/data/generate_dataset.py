"""Reproducible SYNTHETIC player-week dataset generator.

Everything produced here is simulated.  Nothing is, or should ever be described as, real club data.

Simulation outline (all documented assumptions - the paper does not specify them)
--------------------------------------------------------------------------------
1. Each synthetic player has latent traits (fitness, injury proneness, stress propensity) and
   static attributes.  Proneness is a *hidden* player-level trait that is never written to the CSV.
2. Daily workload is simulated for 28 pre-study days + n_weeks*7 days with a weekly microcycle
   (rest day, build-up days, pre-match taper, match day), an AR(1) block-periodisation multiplier and noise.
3. Daily wellness responds to recent training load with lag (fatigue/soreness accumulate, sleep/mood/recovery
   degrade) plus player-level baselines and noise.
4. Weekly observation w uses ONLY the 28 days (workload) / 7 days (wellness) before the observation date.
5. Label: logit = signal_strength * f(observed features) + hidden player proneness + week-level noise.
   A Logistic(0,1) noise term is added and exactly round(prevalence * N) records with the largest noisy
   score become positives.  This equals sampling Bernoulli(sigmoid(logit - c)) conditioned on the total
   count (an explicit "exact positive count" choice: 0.0898 * 12,000 -> 1,078).
   The hidden proneness, noise and probabilities are discarded and NOT saved, so features cannot leak them.
Associations learned from these synthetic labels are not evidence of real-world causation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features.feature_engineering import (
    NO_INJURY_DAYS, POSITIONS, TARGET, WELLNESS_VARS, WORKLOAD_CHANNELS, WELLNESS_DAYS, WORKLOAD_DAYS,
    compute_average_rest_days, compute_bmi, compute_workload_summaries, frame_from_sequences,
)

POS_PROFILE = {  # distance factor, high-speed factor, accel factor, height mean, weight mean, starter prob mean, base risk
    "GK": dict(dist=0.40, hsr=0.30, acc=0.45, h=190, w=84, start=0.8, risk=-0.45),
    "CB": dict(dist=0.85, hsr=0.70, acc=0.85, h=186, w=80, start=0.7, risk=0.00),
    "FB": dict(dist=1.00, hsr=1.20, acc=1.05, h=177, w=73, start=0.7, risk=0.10),
    "CM": dict(dist=1.12, hsr=0.95, acc=1.10, h=179, w=75, start=0.7, risk=0.05),
    "W": dict(dist=1.02, hsr=1.35, acc=1.15, h=177, w=72, start=0.65, risk=0.15),
    "ST": dict(dist=0.95, hsr=1.10, acc=1.00, h=183, w=78, start=0.65, risk=0.10),
}
POS_PROBS = [0.09, 0.19, 0.17, 0.22, 0.15, 0.18]  # GK, CB, FB, CM, W, ST
DOW_TEMPLATE = np.array([0.0, 0.80, 1.00, 0.85, 0.50, 0.30, 1.00])   # Mon..Sun intensity (Sun = match day)
DOW_DURATION = np.array([0.0, 75, 90, 80, 60, 40, 0.0])
DOW_REST_PROB = np.array([0.95, 0.10, 0.05, 0.10, 0.15, 0.10, 0.03])


def _clip_round(x, lo, hi, decimals=0):
    return np.round(np.clip(x, lo, hi), decimals)


def _simulate_player(rng: np.random.Generator, n_weeks: int, pos: str) -> dict:
    prof = POS_PROFILE[pos]
    n_days = WORKLOAD_DAYS + n_weeks * 7
    dow = np.arange(n_days) % 7

    fitness = rng.normal(0, 1)
    proneness = rng.normal(0, 1)               # HIDDEN
    stress_base = rng.normal(0, 1)
    rest_bias = rng.uniform(-0.05, 0.20)
    starter = float(np.clip(rng.normal(prof["start"], 0.15), 0.15, 0.97))

    # --- periodisation multiplier (AR(1) around 1, with occasional ramp-ups)
    m = np.empty(n_days)
    m_prev = 1.0
    for t in range(n_days):
        m_prev = 1.0 + 0.85 * (m_prev - 1.0) + rng.normal(0, 0.07)
        if rng.random() < 0.015:               # random load spike / return from break
            m_prev += rng.choice([-0.35, 0.40])
        m[t] = np.clip(m_prev, 0.45, 1.8)

    rest = rng.random(n_days) < np.clip(DOW_REST_PROB[dow] + rest_bias, 0, 1)
    played = (dow == 6) & (rng.random(n_days) < starter)
    sub_minutes = np.where((dow == 6) & ~played & (rng.random(n_days) < 0.5), rng.integers(10, 35, n_days), 0)

    intensity = np.clip(DOW_TEMPLATE[dow] * m * (0.92 + 0.05 * fitness), 0.1, 1.6)
    dur = DOW_DURATION[dow] * np.where(dow == 6, 0, m) * rng.normal(1, 0.08, n_days)
    dur = np.where(dow == 6, np.where(played, 90 + rng.integers(0, 6, n_days), np.where(sub_minutes > 0, sub_minutes, 40)), dur)
    intensity = np.where((dow == 6) & ~played & (sub_minutes == 0), 0.3, intensity)
    dur = np.round(np.where(rest, 0.0, np.clip(dur, 0, 130)))   # whole minutes, so load == rpe * duration exactly
    active = dur > 0

    rpe = np.where(active, _clip_round(2.5 + 4.0 * intensity + rng.normal(0, 0.8, n_days), 1, 10), 0.0)
    speed = 0.095 * prof["dist"] * (0.75 + 0.35 * intensity)
    dist = np.where(active, dur * speed * rng.normal(1, 0.06, n_days), 0.0)
    hsr = np.where(active, dist * np.clip((0.06 + 0.07 * intensity) * prof["hsr"], 0.01, 0.25) * rng.normal(1, 0.12, n_days), 0.0)
    sprint = np.where(active, hsr * np.clip(0.20 + 0.10 * intensity, 0.05, 0.5) * rng.normal(1, 0.15, n_days), 0.0)
    acc = np.where(active, np.round(dur * (0.35 + 0.30 * intensity) * prof["acc"] * rng.normal(1, 0.10, n_days)), 0.0)
    dec = np.where(active, np.round(dur * (0.38 + 0.30 * intensity) * prof["acc"] * rng.normal(1, 0.10, n_days)), 0.0)
    load = rpe * dur

    workload = np.stack([dist, hsr, sprint, acc, dec, dur, rpe, load], axis=-1)  # (days, 8)
    workload[:, :3] = np.round(workload[:, :3], 2)

    # --- wellness: lagged response to load
    z = (load - load.mean()) / (load.std() + 1e-9)
    F = S = T = 0.0
    wb = np.empty((n_days, len(WELLNESS_VARS)))
    asym_z = 0.0
    sleep_pref = rng.normal(0, 0.3)
    for t in range(n_days):
        F = 0.55 * F + 0.45 * z[t] + rng.normal(0, 0.35)
        S = 0.65 * S + 0.35 * z[max(t - 1, 0)] + rng.normal(0, 0.40)
        T = 0.80 * T + rng.normal(0, 0.45)
        stress = _clip_round(4 + 1.0 * (T + stress_base * 0.6), 1, 10)
        fatigue = _clip_round(4 + 1.4 * F + 0.2 * (stress - 4) + rng.normal(0, 0.6), 1, 10)
        sore = _clip_round(3.5 + 1.3 * S + rng.normal(0, 0.6), 1, 10)
        sleep = np.round(np.clip(7.6 + sleep_pref - 0.25 * F - 0.2 * (stress - 4) / 1.0 + rng.normal(0, 0.6), 3.5, 10.5), 1)
        sq = _clip_round(7 - 0.5 * F - 0.3 * (stress - 4) + 0.8 * (sleep - 7.5) + rng.normal(0, 0.9), 1, 10)
        mood = _clip_round(7 - 0.4 * F - 0.3 * (stress - 4) + rng.normal(0, 1.0), 1, 10)
        rec = _clip_round(65 - 8 * F - 3 * (stress - 4) + 4 * (sleep - 7.5) + rng.normal(0, 6), 0, 100)
        wb[t] = [sleep, sq, fatigue, sore, mood, rec, stress]

    return dict(workload=workload, wellness=wb, proneness=proneness, fitness=fitness,
                match_minutes=np.where(dow == 6, dur, 0.0))


def _static_player(rng, pos, proneness):
    prof = POS_PROFILE[pos]
    age = int(np.clip(round(rng.normal(26, 4.2)), 18, 38))
    height = float(np.clip(rng.normal(prof["h"], 5.0), 160, 205))
    bmi_target = rng.normal(23.3, 1.1)
    weight = float(np.clip(bmi_target * (height / 100) ** 2, 58, 105))
    lam = float(np.clip(0.9 + 0.04 * (age - 26) + 0.35 * proneness, 0.1, 4.0))
    n_prev = int(min(rng.poisson(lam), 10))
    days_missed = int(min(rng.gamma(2.0, 14.0, size=n_prev).sum(), 600)) if n_prev else 0
    since0 = int(rng.integers(30, 700)) if n_prev else NO_INJURY_DAYS
    asym = float(np.clip(abs(rng.normal(7, 3.5)) + 1.5 * max(proneness, 0), 0.5, 35))
    return dict(age=age, height=round(height, 1), weight=round(weight, 1), n_prev=n_prev,
                days_missed=days_missed, since0=since0, asym=asym)


def _label_logit(df: pd.DataFrame, hidden: np.ndarray, rng, signal_strength: float) -> np.ndarray:
    """Observable part of the label logit (centered so the intercept is irrelevant: the exact positive count
    is enforced afterwards).  Weights are illustrative assumptions - not findings."""
    acwr = df["acute_chronic_workload_ratio"].to_numpy()
    f7 = df["wb_fatigue_score_mean7"].to_numpy()
    sore7 = df["wb_soreness_mean7"].to_numpy()
    sleep7 = df["wb_sleep_mean7"].to_numpy()
    rec7 = df["wb_recovery_mean7"].to_numpy()
    str7 = df["wb_stress_mean7"].to_numpy()
    pos_risk = df["position"].map({k: v["risk"] for k, v in POS_PROFILE.items()}).to_numpy()
    obs = (
        0.55 * np.clip((acwr - 1.0) / 0.3, -1.5, 3.0) ** 2 * np.sign(np.abs(acwr - 1.0) - 0.1).clip(0)  # U-shaped ACWR
        + 0.30 * np.clip((df["workload_strain"].to_numpy() - 2400) / 1100, -1.5, 3)
        + 0.30 * np.clip((df["workload_monotony"].to_numpy() - 1.2) / 0.3, -1.5, 3)
        + 0.30 * np.clip(df["previous_injury_count"].to_numpy(), 0, 5)
        + 0.60 * np.exp(-df["days_since_last_injury"].to_numpy() / 60.0)
        + 0.05 * (df["strength_asymmetry_percent"].to_numpy() - 8)
        + 0.04 * (df["age"].to_numpy() - 26)
        + 0.28 * (f7 - 4.0)
        + 0.12 * (sore7 - 3.5)
        + 0.30 * np.clip(7.5 - sleep7, 0, 3)
        - 0.02 * (rec7 - 60.0)
        + 0.10 * (str7 - 4.0)
        + 0.35 * ((acwr > 1.3) & (rec7 < 55)).astype(float)       # interaction: load spike + poor recovery
        + pos_risk
    )
    obs = obs - obs.mean()
    return signal_strength * obs + 0.65 * hidden + rng.normal(0, 0.8, len(df))


def generate_dataset(n_players: int = 250, n_weeks: int = 48, seed: int = 42, prevalence: float = 0.0898,
                     start_date: str = "2023-08-07", signal_strength: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    records, hidden = [], []
    positions = rng.choice(POSITIONS, size=n_players, p=POS_PROBS)

    for pid in range(n_players):
        pos = str(positions[pid])
        sim = _simulate_player(rng, n_weeks, pos)
        st = _static_player(rng, pos, sim["proneness"])
        wl_all, wb_all = sim["workload"], sim["wellness"]
        cum_minutes = np.cumsum(sim["match_minutes"])
        weeks = np.arange(n_weeks)
        end = WORKLOAD_DAYS + 7 * weeks            # first day index of the prediction horizon
        wl = np.stack([wl_all[e - WORKLOAD_DAYS:e] for e in end])            # (W, 28, 8)  strictly before obs date
        wb = np.stack([wb_all[e - WELLNESS_DAYS:e] for e in end])            # (W, 7, 7)
        summ = compute_workload_summaries(wl)
        seq = frame_from_sequences(wl, wb)
        rest_days = np.round(compute_average_rest_days(wl), 2)
        asym_obs = np.clip(st["asym"] + rng.normal(0, 0.8, n_weeks), 0, 40)
        base = pd.DataFrame({
            "observation_id": [f"OBS{pid * n_weeks + w:06d}" for w in weeks],
            "player_id": f"P{pid + 1:03d}",
            "week_index": weeks,
            "observation_date": (start + pd.to_timedelta(7 * weeks, unit="D")).strftime("%Y-%m-%d"),
            "data_source": "SYNTHETIC",
            "age": st["age"], "position": pos, "height_cm": st["height"], "weight_kg": st["weight"],
            "bmi": np.round(compute_bmi(st["height"], st["weight"]), 2),
            "previous_injury_count": st["n_prev"], "previous_injury_days_missed": st["days_missed"],
            "days_since_last_injury": np.minimum(st["since0"] + 7 * weeks, NO_INJURY_DAYS) if st["n_prev"] else NO_INJURY_DAYS,
            "season_minutes_played": np.round(cum_minutes[end - 1]).astype(int),
            "strength_asymmetry_percent": np.round(asym_obs, 2),
            "average_rest_days": rest_days,
        })
        records.append(pd.concat([base, summ.round(4), seq.round(3)], axis=1))
        hidden.append(np.full(n_weeks, sim["proneness"]))

    df = pd.concat(records, ignore_index=True)
    hidden = np.concatenate(hidden)

    # helper aggregates for the label mechanism only (dropped afterwards)
    tmp = df.copy()
    for new, col in [("wb_fatigue_score_mean7", "fatigue_score"), ("wb_soreness_mean7", "muscle_soreness_score"),
                     ("wb_sleep_mean7", "sleep_hours"), ("wb_recovery_mean7", "recovery_score"),
                     ("wb_stress_mean7", "perceived_stress_score")]:
        tmp[new] = tmp[[f"wb_{col}_d{k}" for k in range(1, 8)]].mean(axis=1)
    logit = _label_logit(tmp, hidden, rng, signal_strength)
    noisy = logit + rng.logistic(0, 1, len(df))
    n_pos = int(round(prevalence * len(df)))
    pos_idx = np.argsort(-noisy)[:n_pos]
    y = np.zeros(len(df), dtype=int)
    y[pos_idx] = 1
    df[TARGET] = y
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the SYNTHETIC football injury dataset.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default=None, help="CSV output path (default from config)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-players", type=int, default=None)
    ap.add_argument("--n-weeks", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    d = cfg["data"]
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    df = generate_dataset(args.n_players or d["n_players"], args.n_weeks or d["n_weeks"], seed,
                          d["prevalence"], d["start_date"], d["signal_strength"])
    out = Path(args.output) if args.output else resolve_path(cfg, "dataset")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sample = resolve_path(cfg, "sample_dataset")
    sample.parent.mkdir(parents=True, exist_ok=True)
    df[df["player_id"].isin(sorted(df["player_id"].unique())[:5])].to_csv(sample, index=False)
    print(f"[SYNTHETIC] wrote {len(df):,} records, {df['player_id'].nunique()} players, "
          f"{int(df[TARGET].sum()):,} positives ({df[TARGET].mean():.4%}) -> {out}")
    return df


if __name__ == "__main__":
    main()
