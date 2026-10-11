"""End-to-end training pipeline:  python -m src.training.train [--smoke]

load -> validate -> split -> fit preprocessing (train only) -> baselines -> multimodal NN -> calibrate (val only)
-> choose threshold (val only) -> evaluate once on the held-out test set -> save models, predictions, metrics, figures.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from src.config import load_config, resolve_path
from src.data.preprocessing import Preprocessor, compute_pos_weight, describe_splits, extract_target, make_splits, split_assignment_frame
from src.data.validation import DataValidationError, build_report, load_dataset, validate_dataframe, write_report
from src.evaluation.metrics import apply_threshold, compute_metrics, select_threshold
from src.evaluation import plots
from src.models.baselines import BASELINE_NAMES, FEATURE_GROUP, feature_matrix, fit_baseline
from src.models.calibration import PlattCalibrator
from src.models.multimodal_model import ModelConfig, predict_proba, save_model, to_tensors
from src.training.nn_trainer import train_multimodal
from src.utils.reproducibility import set_seed

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "mlp", "workload_only", "wellness_only", "static_only", "multimodal_dl"]
MODALITIES_USED = {"logistic_regression": "all (aggregated)", "random_forest": "all (aggregated)", "xgboost": "all (aggregated)",
                   "mlp": "all (aggregated)", "workload_only": "workload", "wellness_only": "wellness",
                   "static_only": "static/medical", "multimodal_dl": "all (sequences + static)"}


def _log(verbose, msg):
    if verbose:
        print(msg, flush=True)


def run_training(cfg: dict, smoke: bool = False, dataset_path: str | Path | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    cfg = copy.deepcopy(cfg)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    models_dir, metrics_dir = resolve_path(cfg, "models_dir"), resolve_path(cfg, "metrics_dir")
    pred_dir, fig_dir = resolve_path(cfg, "predictions_dir"), resolve_path(cfg, "figures_dir")
    for d in (models_dir, metrics_dir, pred_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    if smoke:
        cfg["baselines"]["random_forest"]["n_estimators"] = 40
        cfg["baselines"]["xgboost"]["n_estimators"] = 40
        cfg["baselines"]["mlp"]["max_iter"] = 40
        cfg["explainability"]["n_explain"] = 60

    # 1-2. load + validate
    df = load_dataset(dataset_path or resolve_path(cfg, "dataset"))
    results = validate_dataframe(df, expected_records=cfg["data"]["n_players"] * cfg["data"]["n_weeks"],
                                 expected_prevalence=cfg["data"]["prevalence"], prevalence_tolerance=0.005)
    report = build_report(df, results)
    write_report(report, metrics_dir)
    if not report["all_passed"]:
        failed = [r["check"] for r in results if not r["passed"]]
        raise DataValidationError(f"Dataset failed validation: {failed}")
    _log(verbose, f"[1/7] dataset OK: {len(df):,} SYNTHETIC records, prevalence {df['injury_next_7_days'].mean():.2%}")

    # 3. split (grouped by player by default) + leakage-safe preprocessing
    splits = make_splits(df, cfg)
    info = describe_splits(df, splits, cfg)
    assert all(v == 0 for v in info["record_overlap"].values()), "record leakage between splits"
    if cfg["split"]["strategy"] == "grouped":
        assert all(v == 0 for v in info["player_overlap"].values()), "player leakage between splits"
    (metrics_dir / "split_summary.json").write_text(json.dumps(info, indent=2))
    split_assignment_frame(df, splits).to_csv(metrics_dir / "split_assignment.csv", index=False)
    _log(verbose, "[2/7] split (" + info["strategy"] + "): " + ", ".join(f"{k}={v['n_records']} ({v['prevalence']:.2%} pos)" for k, v in info["splits"].items()))
    y = extract_target(df)
    sub = {k: df.iloc[v].reset_index(drop=True) for k, v in splits.items()}
    yy = {k: y[v] for k, v in splits.items()}
    pre = Preprocessor().fit(sub["train"])                       # fitted on TRAIN only
    data = {k: pre.transform(sub[k]) for k in sub}
    pre.save(models_dir / "preprocessor.joblib")
    pos_weight = compute_pos_weight(yy["train"], cfg["imbalance"]["pos_weight_power"])
    _log(verbose, f"[3/7] preprocessing fitted on train only; positive class weight = {pos_weight:.2f}")

    tcfg = cfg["threshold"]
    calibrators, thresholds, rows, test_preds = {}, {}, [], {}

    def finalize(name, raw_val, raw_test):
        cal = PlattCalibrator().fit(raw_val, yy["val"])
        pv, pt = cal.transform(raw_val), cal.transform(raw_test)
        thr = select_threshold(yy["val"], pv, tcfg["strategy"], tcfg["fixed_value"], tcfg["recall_target"])   # validation only
        calibrators[name], thresholds[name] = cal, thr
        m = compute_metrics(yy["test"], pt, thr)
        mv = compute_metrics(yy["val"], pv, thr)
        rows.append({"model": name, "modalities": MODALITIES_USED[name], "threshold_strategy": tcfg["strategy"], **m,
                     "val_roc_auc": mv["roc_auc"], "val_pr_auc": mv["pr_auc"], "val_f1": mv["f1"],
                     "n_test": int(len(yy["test"])), "n_test_positive": int(yy["test"].sum())})
        for split, raw, p in (("val", raw_val, pv), ("test", raw_test, pt)):
            pd.DataFrame({"observation_id": sub[split]["observation_id"], "player_id": sub[split]["player_id"],
                          "y_true": yy[split], "prob_raw": raw, "prob": p, "pred": apply_threshold(p, thr)}
                         ).to_csv(pred_dir / f"{name}_{split}_predictions.csv", index=False)
        test_preds[name] = pd.read_csv(pred_dir / f"{name}_test_predictions.csv")
        _log(verbose, f"      {name:<20s} test ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | recall {m['recall']:.3f} | precision {m['precision']:.3f} | thr {thr:.3f}")

    # 4. baselines
    _log(verbose, "[4/7] training baselines")
    for name in BASELINE_NAMES:
        g = FEATURE_GROUP[name]
        est = fit_baseline(name, cfg, feature_matrix(data["train"], g), yy["train"], pos_weight, seed)
        joblib.dump(est, models_dir / f"{name}.joblib")
        rv = est.predict_proba(feature_matrix(data["val"], g))[:, 1]
        rt = est.predict_proba(feature_matrix(data["test"], g))[:, 1]
        finalize(name, rv, rt)

    # 5. multimodal network
    _log(verbose, "[5/7] training multimodal deep-learning model (CPU)")
    mcfg = ModelConfig.from_config(cfg, static_dim=data["train"]["static"].shape[1])
    tin = {k: to_tensors(v) for k, v in data.items()}
    epochs = 3 if smoke else None
    model, hist = train_multimodal(mcfg, tin["train"], yy["train"], tin["val"], yy["val"], cfg, pos_weight, seed,
                                   checkpoint_path=models_dir / "multimodal_dl_best_checkpoint.pt", max_epochs=epochs, verbose=verbose)
    save_model(model, models_dir / "multimodal_dl")
    hist.to_csv(metrics_dir / "nn_training_history.csv", index=False)
    finalize("multimodal_dl", predict_proba(model, tin["val"]).numpy(), predict_proba(model, tin["test"]).numpy())

    # 6-7. outputs
    comp = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER].reset_index()
    comp.to_csv(metrics_dir / "model_comparison.csv", index=False)
    joblib.dump(calibrators, models_dir / "calibrators.joblib")
    plots.plot_roc_pr_calibration({k: test_preds[k] for k in MODEL_ORDER}, fig_dir)
    for r in rows:
        plots.plot_confusion(r["model"], {k: r[k] for k in ("tn", "fp", "fn", "tp")}, r["threshold"], fig_dir)
    plots.plot_training_history(hist, fig_dir)
    plots.plot_model_comparison(comp, fig_dir)
    registry = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "smoke_run": smoke, "data_label": "SYNTHETIC", "seed": seed,
        "selected_model": cfg["selected_model"], "models": MODEL_ORDER, "thresholds": thresholds,
        "threshold_strategy": tcfg["strategy"], "pos_weight": pos_weight, "split": info,
        "nn_best_epoch": int(hist.loc[hist["val_loss"].idxmin(), "epoch"]), "nn_epochs_run": int(len(hist)),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__},
        "risk_categories": cfg["risk_categories"], "runtime_seconds": round(time.time() - t0, 1),
    }
    (models_dir / "model_registry.json").write_text(json.dumps(registry, indent=2))
    _log(verbose, f"[6/7] metrics -> {metrics_dir/'model_comparison.csv'}")
    _log(verbose, f"[7/7] artifacts saved ({registry['runtime_seconds']}s). Next: python -m src.explainability.shap_explainer")
    return {"comparison": comp, "registry": registry, "history": hist}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train all models on the SYNTHETIC dataset and evaluate on the held-out test set.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="quick smoke test (3 NN epochs, small baselines)")
    ap.add_argument("--split-strategy", choices=["grouped", "random_stratified"], default=None)
    ap.add_argument("--threshold-strategy", choices=["fixed", "val_f1", "val_youden", "val_recall_target"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    ov: dict = {}
    if args.split_strategy: ov.setdefault("split", {})["strategy"] = args.split_strategy
    if args.threshold_strategy: ov.setdefault("threshold", {})["strategy"] = args.threshold_strategy
    if args.seed is not None: ov.setdefault("project", {})["seed"] = args.seed
    cfg = load_config(args.config, ov)
    out = run_training(cfg, smoke=args.smoke)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out["comparison"][["model", "threshold", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
