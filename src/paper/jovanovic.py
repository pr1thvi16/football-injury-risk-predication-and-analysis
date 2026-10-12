"""Python port of Jovanovic (2018) 'Data Preparation for Injury Prediction' (SPSR 34) + adapter to the paper pipeline.

Source code/data: https://github.com/mladenjovanovic/Data-Preparation-for-Injury-Prediction (sample data: 3 players, 307 days,
8 injuries; no wellness, no static fields). Features: EMA acute (7 d) / chronic (42 d, lagged 7 d), ACWR, 7-day rolling ACWR peak,
lags 0/7/14/21, and injury tags with a lead of 1/7/14/21 days (non-contact = 'Soft Tissue' or 'Overuse').
"""
import numpy as np
import pandas as pd

METRICS = ["RPE.Load", "Total.Distance", "HS.Distance"]


def prepare(raw: pd.DataFrame, pad_days=70) -> pd.DataFrame:
    """Reproduces Data Preparation.R: returns player-day rows with <metric>.{Acute,Chronic,ACWR,ACWRrollPeak,Day.Value}."""
    out = []
    for p, g in raw.sort_values(["Player.Name", "Date"]).groupby("Player.Name"):
        pad = pd.DataFrame({"Date": np.arange(-pad_days, 1), "Player.Name": p, **{m: 0.0 for m in METRICS}})
        g = pd.concat([pad, g], ignore_index=True)
        for m in METRICS:
            v = g[m].astype(float)
            acute = v.ewm(alpha=2 / 8, adjust=False).mean().round(2)
            chronic = v.ewm(alpha=2 / 43, adjust=False).mean().round(2).shift(7)
            acwr = (acute / chronic).where(chronic != 0, 0.0).round(2)
            g[f"{m}.Day.Value"], g[f"{m}.Acute"], g[f"{m}.Chronic"], g[f"{m}.ACWR"] = v, acute, chronic, acwr
            g[f"{m}.ACWRrollPeak"] = acwr.rolling(7).max()
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    d["noncontact"] = d["Injury.Type"].isin(["Soft Tissue", "Overuse"])
    return d


def to_pipeline_tables(raw: pd.DataFrame):
    """daily + injuries tables for src.paper.real_data (load = RPE load; non-contact injuries only)."""
    base = pd.Timestamp("2000-01-01")
    daily = pd.DataFrame({"player_id": raw["Player.Name"], "date": (base + pd.to_timedelta(raw["Date"], unit="D")).dt.strftime("%Y-%m-%d"),
                          "load": raw["RPE.Load"]})
    i = raw[raw["Injury.Type"].isin(["Soft Tissue", "Overuse"])]
    inj = pd.DataFrame({"player_id": i["Player.Name"], "injury_date": (base + pd.to_timedelta(i["Date"], unit="D")).dt.strftime("%Y-%m-%d")})
    return daily, inj
