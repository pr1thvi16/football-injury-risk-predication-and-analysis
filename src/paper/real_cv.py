"""Leave-players-out cross-validation for SMALL real datasets (few injuries): every injury ends up in exactly one
held-out fold, predictions are pooled, and the AUC CI comes from a player-cluster bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.paper.features import TARGET
from src.paper.model import PaperModel, make_baselines
from src.paper.real_data import build_player_weeks
from src.paper.run import load_cfg


def cluster_bootstrap_auc(y, p, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; v = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[ii].sum() < len(ii):
            v.append(roc_auc_score(y[ii], p[ii]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mapping", default="config/real_data_mapping.yaml")
    ap.add_argument("--mods", default="load,wellness"); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="soccermon")
    a = ap.parse_args(); cfg = load_cfg(); spec = yaml.safe_load(open(a.mapping)); mods = a.mods.split(",")
    spec.setdefault("options", {})["data_source"] = "REAL"
    rd = lambda k: pd.read_csv(spec["files"][k]) if spec["files"].get(k) else None
    w = build_player_weeks(rd("daily"), rd("injuries"), rd("static"), spec)
    y = w[TARGET].to_numpy(); g = w.player_id.to_numpy()
    print(f"REAL: {len(w)} player-weeks, {w.player_id.nunique()} players, {y.sum()} positive weeks ({y.mean():.1%}), "
          f"players with an injury week: {w[w[TARGET] == 1].player_id.nunique()}")
    syn = PaperModel.load(Path(cfg["paths"]["models_dir"]) / f"fusion_{'_'.join(mods)}.joblib")
    e = cfg["encoder"]; names = ["Synthetic-trained zero-shot", "Synthetic-trained fine-tuned", "Retrained on real folds", "LR (early fusion)",
                                  "Random forest", "Gradient boosting", "ACWR only (heuristic)"]
    oof = {n: np.zeros(len(w)) for n in names}; runs = []
    for r in range(a.repeats):
        o = {n: np.zeros(len(w)) for n in names}
        for tr, te in StratifiedGroupKFold(a.folds, shuffle=True, random_state=r).split(w, y, g):
            dtr, dte = w.iloc[tr].reset_index(drop=True), w.iloc[te]
            o["Synthetic-trained zero-shot"][te] = syn.predict_proba(dte)
            o["Synthetic-trained fine-tuned"][te] = syn.fine_tune(dtr).predict_proba(dte)
            o["Retrained on real folds"][te] = PaperModel(mods, e["hidden"], e["alpha"], e["max_iter"], cfg["fusion_head"]["C"], r).fit(dtr).predict_proba(dte)
            bl = make_baselines(cfg, mods, r)
            for k in ("LR (early fusion)", "Random forest", "Gradient boosting"):
                o[k][te] = bl[k].fit(dtr).predict_proba(dte)
            o["ACWR only (heuristic)"][te] = dte["acwr"].fillna(1).to_numpy()
        runs.append(o)
    rows = []
    for n in names:
        aucs = [roc_auc_score(y, o[n]) for o in runs]; pooled = np.mean([o[n] for o in runs], 0)
        lo, hi = cluster_bootstrap_auc(y, pooled, g)
        rows.append({"model": n, "roc_auc_mean_over_repeats": np.mean(aucs), "auc_sd_over_repeats": np.std(aucs), "ci_low": lo, "ci_high": hi,
                     "pr_auc": average_precision_score(y, pooled), "prevalence": y.mean()})
    t = pd.DataFrame(rows); out = Path(spec.get("output_dir", "results/real")); out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / f"cv_{a.tag}.csv", index=False); print(t.round(3).to_string(index=False))
    json.dump({"players": int(w.player_id.nunique()), "weeks": len(w), "positives": int(y.sum()), "mods": mods}, open(out / f"cv_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
