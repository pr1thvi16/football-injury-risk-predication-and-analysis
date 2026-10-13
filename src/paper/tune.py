"""Hyper-parameter search for the fusion model using VALIDATION AUC only (seeds disjoint from the test-evaluation seeds)."""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.paper import evaluation as ev
from src.paper.features import TARGET
from src.paper.generator import MAPPING, generate_tables
from src.paper.model import PaperModel
from src.paper.real_data import build_player_weeks
from src.paper.run import load_cfg

cfg = load_cfg(); d = cfg["data"]; SEEDS = range(200, 205)
data = []
for s in SEEDS:
    w = build_player_weeks(*generate_tables(d["n_players"], d["n_weeks"], d["prevalence"], d["start_date"], d["signal_strength"], s), MAPPING)
    data.append((s, ev.grouped_split(w, 8160, 1440, 2400, s)))
rows = []
for alpha, hidden, C in itertools.product([0.01, 0.1, 1.0], [(32, 16), (16, 16)], [0.05, 0.5]):
    aucs = [roc_auc_score(sp["val"][TARGET], PaperModel(("static", "load", "wellness"), hidden, alpha, 400, C, s).fit(sp["train"]).predict_proba(sp["val"])) for s, sp in data]
    rows.append({"alpha": alpha, "hidden": str(hidden), "head_C": C, "val_auc_mean": np.mean(aucs), "val_auc_sd": np.std(aucs)})
    print(rows[-1], flush=True)
t = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False); t.to_csv("results/paper/tuning_validation.csv", index=False); print(t.round(4).to_string(index=False))
