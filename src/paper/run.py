"""CLI for the paper pipeline.

  python -m src.paper.run generate                      # SYNTHETIC tables + player-weeks
  python -m src.paper.run train [--modalities static,load,wellness]   # train + evaluate on synthetic data
  python -m src.paper.run explain                       # permutation-Shapley (global + case study)
  python -m src.paper.run real --mapping config/real_data_mapping.yaml [--modes zero_shot,retrain,fine_tune]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.paper import evaluation as ev
from src.paper.features import MODALITY_LABEL, TARGET
from src.paper.model import PaperModel, make_baselines
from src.paper.real_data import build_player_weeks

PAPER_REPORTED_AUC = {"Fusion model": 0.706, "LR (static only)": 0.693, "LR (early fusion)": 0.719}   # IEEE paper, as quoted


def load_cfg(path="config/paper.yaml"):
    return yaml.safe_load(open(path))


def _tag(mods):
    return "_".join(mods)


def evaluate_models(models: dict, splits: dict, n_boot: int, seed: int):
    """models: name -> fitted object with predict_proba. Threshold from validation; metrics + CIs on test."""
    rows, preds = [], {}
    yv, yt = splits["val"][TARGET].to_numpy(), splits["test"][TARGET].to_numpy()
    for name, m in models.items():
        pv, pt = m.predict_proba(splits["val"]), m.predict_proba(splits["test"])
        thr = ev.youden_threshold(yv, pv)
        r = {"model": name, **ev.point_metrics(yt, pt, thr)}
        r["auc_ci_low"], r["auc_ci_high"] = ev.bootstrap_auc(yt, pt, n_boot, seed)
        rows.append(r); preds[name] = pt
    return pd.DataFrame(rows), preds


def cmd_train(cfg, mods, quick=False):
    p = cfg["paths"]
    weeks = pd.read_csv(p["weeks"])
    sp = cfg["split"]
    splits = (ev.grouped_split if sp["strategy"] == "grouped" else ev.random_split)(weeks, sp["train"], sp["val"], sp["test"], cfg["seed"])
    print({k: (len(v), int(v[TARGET].sum())) for k, v in splits.items()})
    e = cfg["encoder"]
    fusion = PaperModel(mods, e["hidden"], e["alpha"], e["max_iter"], cfg["fusion_head"]["C"], cfg["seed"]).fit(splits["train"])
    models = {"Fusion model": fusion, **{k: v.fit(splits["train"]) for k, v in make_baselines(cfg, mods, cfg["seed"]).items()}}
    n_boot = 100 if quick else cfg["bootstrap"]["n"]
    table, preds = evaluate_models(models, splits, n_boot, cfg["seed"])
    tag = _tag(mods); out = Path(cfg["paths"]["results_dir"]); out.mkdir(parents=True, exist_ok=True)
    table["paper_reported_auc"] = table.model.map(PAPER_REPORTED_AUC)
    table.to_csv(out / f"model_comparison_{tag}.csv", index=False)
    yt = splits["test"][TARGET].to_numpy()
    paired = {k: ev.paired_bootstrap(yt, preds["Fusion model"], preds[k], n_boot, cfg["seed"]) for k in preds if k != "Fusion model"}
    ev.calibration_quintiles(yt, preds["Fusion model"]).to_csv(out / f"calibration_{tag}.csv", index=False)
    ev.modality_ablation(fusion, splits["test"], splits["train"]).to_csv(out / f"ablation_{tag}.csv", index=False)
    ev.subgroup_auc(splits["test"], preds["Fusion model"]).to_csv(out / f"subgroups_{tag}.csv", index=False)
    json.dump({"paired_bootstrap_fusion_minus_other": paired, "cv_head": ev.cv_head(fusion, splits["train"], 5, cfg["seed"]),
               "n_test": len(yt), "n_test_positives": int(yt.sum()), "data_source": str(weeks.data_source.iloc[0])},
              open(out / f"extra_{tag}.json", "w"), indent=2)
    Path(cfg["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    fusion.save(Path(cfg["paths"]["models_dir"]) / f"fusion_{tag}.joblib")
    for k, s in splits.items():
        s.to_csv(out / f"split_{k}.csv", index=False) if tag == "static_load_wellness" else None
    print(table[["model", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "precision", "recall", "f1", "brier"]].round(3).to_string(index=False))
    return fusion, splits


def cmd_explain(cfg, mods):
    from src.paper import shapley
    out = Path(cfg["paths"]["results_dir"]); tag = _tag(mods)
    model = PaperModel.load(Path(cfg["paths"]["models_dir"]) / f"fusion_{tag}.joblib")
    train, test = pd.read_csv(out / "split_train.csv"), pd.read_csv(out / "split_test.csv")
    s = cfg["shapley"]
    inst = test.sample(s["global_instances"], random_state=cfg["seed"])
    imp, _ = shapley.global_importance(model, train, inst, s["global_permutations"], cfg["seed"], s["background"])
    imp.to_csv(out / "shapley_global.csv", index=False)
    share = shapley.modality_share(imp); share.to_csv(out / "shapley_modality_share.csv", index=False)
    pt = model.predict_proba(test); top = test.iloc[int(np.argmax(pt))]
    shapley.case_study(model, train, top, s["case_permutations"], cfg["seed"], s["background"]).to_csv(out / "shapley_case_study.csv", index=False)
    print(share.round(1).to_string(index=False)); print(imp.head(8).round(4).to_string(index=False))


def cmd_real(cfg, mapping_path, modes, quick=False):
    """Apply the pipeline to REAL data. Same builder, same split protocol, three ways to use the synthetic-trained model."""
    spec = yaml.safe_load(open(mapping_path))
    spec.setdefault("options", {})["data_source"] = "REAL"
    rd = lambda k: pd.read_csv(spec["files"][k]) if spec["files"].get(k) else None
    weeks = build_player_weeks(rd("daily"), rd("injuries"), rd("static"), spec)
    avail = weeks.attrs["available_static"]
    mods = ["load", "wellness"] + (["static"] if len(avail) >= 3 else [])
    mods = [m for m in ["static", "load", "wellness"] if m in mods]
    print(f"REAL data: {len(weeks)} player-weeks, {weeks.player_id.nunique()} players, {int(weeks[TARGET].sum())} positives; "
          f"modalities used: {mods}; static fields present: {avail}")
    if weeks[TARGET].sum() < 20:
        print("WARNING: <20 positives - any metric will be extremely uncertain.")
    out = Path(spec.get("output_dir", "results/real")); out.mkdir(parents=True, exist_ok=True)
    weeks.to_csv(out / "real_player_weeks.csv", index=False)
    sp = cfg["split"]; tot = sp["train"] + sp["val"] + sp["test"]
    splits = ev.grouped_split(weeks, sp["train"], sp["val"], sp["test"], cfg["seed"])
    tag = _tag(mods); mpath = Path(cfg["paths"]["models_dir"]) / f"fusion_{tag}.joblib"
    if not mpath.exists():
        print(f"No synthetic-trained model for modalities {mods}; training one on synthetic data first")
        cmd_train(cfg, mods, quick)
    syn = PaperModel.load(mpath)
    e = cfg["encoder"]; models = {}
    if "zero_shot" in modes: models["Synthetic-trained, zero-shot"] = syn
    if "fine_tune" in modes: models["Synthetic-trained, fine-tuned on real train"] = syn.fine_tune(splits["train"])
    if "retrain" in modes: models["Retrained from scratch on real train"] = PaperModel(mods, e["hidden"], e["alpha"], e["max_iter"], cfg["fusion_head"]["C"], cfg["seed"]).fit(splits["train"])
    models.update({k: v.fit(splits["train"]) for k, v in make_baselines(cfg, mods, cfg["seed"]).items()})
    table, _ = evaluate_models(models, splits, 100 if quick else cfg["bootstrap"]["n"], cfg["seed"])
    table.to_csv(out / "real_model_comparison.csv", index=False)
    print(table[["model", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "recall", "precision"]].round(3).to_string(index=False))
    print(f"Test set: {len(splits['test'])} rows, {int(splits['test'][TARGET].sum())} positives. Saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "train", "explain", "real"])
    ap.add_argument("--config", default="config/paper.yaml"); ap.add_argument("--modalities", default="static,load,wellness")
    ap.add_argument("--mapping"); ap.add_argument("--modes", default="zero_shot,fine_tune,retrain"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); cfg = load_cfg(a.config); mods = a.modalities.split(",")
    if a.command == "generate":
        from src.paper.generator import main as g; import sys; sys.argv = ["gen", "--config", a.config]; g()
    elif a.command == "train": cmd_train(cfg, mods, a.quick)
    elif a.command == "explain": cmd_explain(cfg, mods)
    else: cmd_real(cfg, a.mapping, a.modes.split(","), a.quick)


if __name__ == "__main__":
    main()
