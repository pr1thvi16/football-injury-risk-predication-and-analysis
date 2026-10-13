"""Repeat the whole synthetic experiment over several seeds (new dataset + new split + new init each time).
Reports mean / SD of test ROC-AUC and PR-AUC per model, and the per-seed paired difference fusion - baseline."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.paper import evaluation as ev
from src.paper.features import TARGET
from src.paper.generator import MAPPING, generate_tables
from src.paper.model import PaperModel, make_baselines
from src.paper.real_data import build_player_weeks
from src.paper.run import load_cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=None); ap.add_argument("--hidden", default=None); ap.add_argument("--tag", default="default")
    a = ap.parse_args(); cfg = load_cfg(); e = dict(cfg["encoder"]); d = cfg["data"]
    if a.alpha is not None: e["alpha"] = a.alpha
    if a.hidden: e["hidden"] = [int(x) for x in a.hidden.split(",")]
    cfg["encoder"] = e; rows = []
    for s in range(a.start, a.start + a.seeds):
        w = build_player_weeks(*generate_tables(d["n_players"], d["n_weeks"], d["prevalence"], d["start_date"], d["signal_strength"], s), MAPPING)
        sp = ev.grouped_split(w, cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"], s)
        models = {"Fusion model": PaperModel(("static", "load", "wellness"), e["hidden"], e["alpha"], e["max_iter"], cfg["fusion_head"]["C"], s).fit(sp["train"]),
                  **{k: v.fit(sp["train"]) for k, v in make_baselines(cfg, ("static", "load", "wellness"), s).items()}}
        yt = sp["test"][TARGET].to_numpy()
        for k, m in models.items():
            p = m.predict_proba(sp["test"])
            rows.append({"seed": s, "model": k, "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p), "n_pos_test": int(yt.sum())})
        print("seed", s, "done", flush=True)
    r = pd.DataFrame(rows); out = Path(cfg["paths"]["results_dir"]); out.mkdir(parents=True, exist_ok=True); r.to_csv(out / f"multiseed_{a.tag}_raw.csv", index=False)
    summ = r.groupby("model").agg(auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"), pr_auc_mean=("pr_auc", "mean")).sort_values("auc_mean", ascending=False)
    piv = r.pivot(index="seed", columns="model", values="roc_auc")
    summ["fusion_minus_this_mean"] = (piv["Fusion model"].to_numpy()[:, None] - piv).mean().reindex(summ.index)
    summ["fusion_wins_seeds"] = (piv["Fusion model"].to_numpy()[:, None] > piv).sum().reindex(summ.index)
    summ.to_csv(out / f"multiseed_{a.tag}.csv"); print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
