"""Convert the SoccerMon *subjective* release (wide per-metric CSVs + session.json + injury.csv) into the three
long tables the pipeline's real-data path expects (daily, injuries). Dataset: Midoglu et al., CC BY 4.0 (Zenodo).

Injury definition (an ASSUMPTION - SoccerMon has no time-loss field): injury.csv holds daily self-reported problems
(body area + minor/major). An *episode onset* is a report with no report by that player in the previous ``gap_days``.
mode="any": every episode onset; mode="major": onsets of episodes that contain at least one 'major' report.
"""
import argparse
from pathlib import Path

import pandas as pd


def _long(raw, name, col):
    df = pd.read_csv(raw / name)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    return df.melt("date", var_name="player_id", value_name=col)


def episodes(inj: pd.DataFrame, gap_days=14, mode="any") -> pd.DataFrame:
    rows = []
    for pid, g in inj.sort_values("date").groupby("player_id"):
        dates = g["date"].tolist(); major = g["major"].tolist()
        start = 0
        for i in range(1, len(dates) + 1):
            if i == len(dates) or (dates[i] - dates[i - 1]).days >= gap_days:
                if mode == "any" or any(major[start:i]):
                    rows.append((pid, dates[start]))
                start = i
    return pd.DataFrame(rows, columns=["player_id", "injury_date"])


def convert(raw_dir="data/real/raw", out_dir="data/real", gap_days=14, mode="any"):
    raw, out = Path(raw_dir), Path(out_dir)
    d = _long(raw, "daily_load.csv", "load")
    for n, c in [("fatigue", "fatigue"), ("soreness", "soreness"), ("mood", "mood"), ("sleep_duration", "sleep")]:
        d = d.merge(_long(raw, f"{n}.csv", c), on=["date", "player_id"], how="outer")
    d = d.dropna(subset=["load", "fatigue", "soreness", "mood", "sleep"], how="all").sort_values(["player_id", "date"])
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    inj = pd.read_csv(raw / "injury.csv").rename(columns={"player_name": "player_id"})
    inj["date"] = pd.to_datetime(inj["timestamp"], format="%d.%m.%Y"); inj["major"] = inj["type"].str.contains("major")
    ep = episodes(inj, gap_days, mode); ep["injury_date"] = ep["injury_date"].dt.strftime("%Y-%m-%d")
    d.to_csv(out / "daily.csv", index=False); ep.to_csv(out / "injuries.csv", index=False)
    print(f"daily rows {len(d)}, players {d.player_id.nunique()}; injury episodes ({mode}, gap {gap_days}d): {len(ep)} "
          f"in {ep.player_id.nunique()} players")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="any", choices=["any", "major"]); ap.add_argument("--gap", type=int, default=14)
    a = ap.parse_args(); convert(gap_days=a.gap, mode=a.mode)
