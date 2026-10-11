"""Explainability:  python -m src.explainability.shap_explainer

Methods (kept clearly separate - they are NOT interchangeable):
  * SHAP TreeExplainer  -> tree models (XGBoost on all aggregated features). Additive log-odds contributions.
  * Integrated Gradients (own implementation, zero baseline = training mean after scaling)
                        -> the multimodal neural network. Gradient-based attribution of the raw logit.
  * Permutation importance -> model-agnostic fallback/sanity check (drop in score when a feature/modality is shuffled).
Importances describe MODEL behaviour on SYNTHETIC data; they do not prove causation or explain real injuries.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config, resolve_path
from src.data.validation import load_dataset
from src.features.feature_engineering import (
    POSITIONS, STATIC_NUMERIC, aggregate_wellness_features, aggregate_workload_features, compute_average_rest_days,
    sequences_from_frame,
)
from src.inference import InjuryRiskPredictor
from src.models.baselines import FEATURE_GROUP, all_feature_names, feature_matrix
from src.models.multimodal_model import to_tensors
from src.utils.reproducibility import set_seed

MOD_STATIC, MOD_WORK, MOD_WELL = "Static & medical", "Workload", "Wellness"
TREE_MODELS = ("xgboost", "workload_only", "wellness_only", "static_only")
KEYS = ("static", "wl_seq", "wl_sum", "wb_seq")
MODALITY_OF_INPUT = {"static": MOD_STATIC, "wl_seq": MOD_WORK, "wl_sum": MOD_WORK, "wb_seq": MOD_WELL}
METHOD_NAMES = {"tree_shap": "SHAP TreeExplainer (XGBoost, log-odds)", "ig": "Integrated Gradients (multimodal DL, logit)"}


# ------------------------------------------------------------------------------------ Integrated Gradients
def integrated_gradients(model, inputs: dict, steps: int = 32, baselines: dict | None = None) -> dict:
    """IG_i = (x_i - x'_i) * mean_over_alpha dF(x' + alpha(x - x'))/dx_i, with F = raw logit.
    Baseline x' = 0 (the training mean, because inputs are standardised). Completeness: sum(IG) ~ F(x) - F(x')."""
    model.eval()
    x = {k: inputs[k].detach().float() for k in KEYS}
    base = {k: torch.zeros_like(v) for k, v in x.items()} if baselines is None else baselines
    grads = {k: torch.zeros_like(v) for k, v in x.items()}
    for a in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:      # midpoint rule
        pt = {k: (base[k] + a * (x[k] - base[k])).requires_grad_(True) for k in KEYS}
        out = model(**pt).sum()
        g = torch.autograd.grad(out, [pt[k] for k in KEYS])
        for k, gk in zip(KEYS, g):
            grads[k] += gk / steps
    attr = {k: ((x[k] - base[k]) * grads[k]).detach().numpy() for k in KEYS}
    with torch.no_grad():
        attr["_delta"] = (model(**x) - model(**base)).numpy()               # F(x) - F(x') for the completeness check
    return attr


def ig_feature_matrix(attr: dict, names: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Per-sample feature-level attributions: static as-is, sequences summed over time (per channel)."""
    cols, feats, mods = [], [], []
    for key in KEYS:
        a = attr[key]
        if a.ndim == 3:
            a = a.sum(axis=1)                                   # sum over days -> per channel
        suffix = {"wl_seq": " (28-day series)", "wb_seq": " (7-day series)", "wl_sum": " (derived)", "static": ""}[key]
        for j, n in enumerate(names[key]):
            feats.append(n + suffix); mods.append(MODALITY_OF_INPUT[key]); cols.append(a[:, j])
    return np.stack(cols, axis=1), feats, mods


def global_table(values: np.ndarray, feats: list[str], mods: list[str]) -> pd.DataFrame:
    t = pd.DataFrame({"feature": feats, "modality": mods, "mean_abs_attribution": np.abs(values).mean(0),
                      "mean_signed_attribution": values.mean(0)})
    return t.sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)


def modality_table(values: np.ndarray, mods: list[str]) -> pd.DataFrame:
    mods = np.array(mods)
    per = {m: np.abs(values[:, mods == m]).sum(1).mean() for m in (MOD_STATIC, MOD_WORK, MOD_WELL)}
    tot = sum(per.values()) or 1.0
    return pd.DataFrame({"modality": list(per), "mean_abs_attribution_sum": list(per.values()),
                         "share_percent": [100 * v / tot for v in per.values()]}).sort_values("share_percent", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------ raw display values
def raw_tab_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Un-scaled values of the aggregated features, columns aligned with all_feature_names()."""
    wl, wb = sequences_from_frame(df)
    s = df[STATIC_NUMERIC].reset_index(drop=True).astype(float).copy()
    s["average_rest_days"] = compute_average_rest_days(wl)
    for p in POSITIONS:
        s[f"position_{p}"] = (df["position"].to_numpy() == p).astype(float)
    return pd.concat([s, aggregate_workload_features(wl), aggregate_wellness_features(wb)], axis=1)


def _effect(v: float, tol: float = 1e-9) -> str:
    return "raises model output" if v > tol else ("lowers model output" if v < -tol else "no effect")


def explain_single(predictor: InjuryRiskPredictor, df_row: pd.DataFrame, model_name: str | None = None,
                   top_k: int = 12, steps: int = 32) -> tuple[pd.DataFrame, str]:
    """Feature contributions for ONE record. Returns (table, method description)."""
    name = model_name or predictor.selected_model
    arrays = predictor.preprocessor.transform(df_row.reset_index(drop=True))
    if name == "multimodal_dl":
        tin = to_tensors(arrays)
        attr = ig_feature_matrix(integrated_gradients(predictor.get_model(name), tin, steps), predictor.preprocessor.nn_feature_names)
        vals, feats, mods = attr[0][0], attr[1], attr[2]
        wl, wb = sequences_from_frame(df_row)
        raw = {}
        for i, c in enumerate(predictor.preprocessor.nn_feature_names["wl_seq"]):
            raw[c + " (28-day series)"] = f"{wl[0, -7:, i].mean():.2f} (7-day mean)"
        for i, c in enumerate(predictor.preprocessor.nn_feature_names["wb_seq"]):
            raw[c + " (7-day series)"] = f"{wb[0, :, i].mean():.2f} (7-day mean)"
        rt = raw_tab_frame(df_row).iloc[0]
        for n in predictor.preprocessor.nn_feature_names["static"]:
            raw[n] = f"{rt[n]:.2f}"
        for n in predictor.preprocessor.nn_feature_names["wl_sum"]:
            raw[n + " (derived)"] = f"{rt[n]:.2f}"
        method = METHOD_NAMES["ig"]
        values = [raw.get(f, "") for f in feats]
    elif name in TREE_MODELS:
        import shap
        g = FEATURE_GROUP[name]
        X = feature_matrix(arrays, g)
        names_all = all_feature_names(predictor.preprocessor)
        grp = predictor.preprocessor.feature_groups
        names = names_all if g == "all" else [f"[{g}] {n}" for n in grp[g]]
        vals = shap.TreeExplainer(predictor.get_model(name)).shap_values(X)[0]
        feats = [n.split("] ", 1)[1] for n in names]
        mods = [{"static": MOD_STATIC, "workload": MOD_WORK, "wellness": MOD_WELL}[n[1:n.index("]")]] for n in names]
        rt = raw_tab_frame(df_row).iloc[0]
        values = [f"{rt[f]:.2f}" for f in feats]
        method = METHOD_NAMES["tree_shap"].replace("XGBoost", name)
    else:
        raise NotImplementedError(f"No attribution method is implemented for '{name}'. Choose the multimodal model or an XGBoost-based model.")
    t = pd.DataFrame({"feature": feats, "modality": mods, "value": values, "attribution": vals})
    t["effect"] = t["attribution"].map(_effect)
    t = t.reindex(t["attribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return t, method


def modality_from_table(t: pd.DataFrame) -> pd.DataFrame:
    m = t.groupby("modality")["attribution"].agg(net="sum", abs_sum=lambda s: s.abs().sum()).reindex([MOD_STATIC, MOD_WORK, MOD_WELL]).fillna(0)
    m["share_percent"] = 100 * m["abs_sum"] / (m["abs_sum"].sum() or 1)
    return m.reset_index()


def answer_questions(t: pd.DataFrame) -> list[str]:
    """Plain-language answers derived from the attribution table (associations, not causes)."""
    out = []
    rank = {f: i + 1 for i, f in enumerate(t["feature"])}

    def block(label, mask):
        sub = t[mask]
        if sub.empty:
            return f"{label}: no matching features."
        net = sub["attribution"].sum()
        best = sub.iloc[sub["attribution"].abs().argmax()]
        return (f"{label}: net contribution {net:+.3f} ({'pushes the estimate up' if net > 0 else 'pushes the estimate down'}); "
                f"largest single feature `{best['feature']}` ({best['attribution']:+.3f}, rank {rank[best['feature']]}).")

    out.append(block("Recent fatigue (fatigue score)", t["feature"].str.contains("fatigue_score")))
    out.append(block("Previous injuries", t["feature"].str.contains("previous_injury|days_since_last_injury")))
    out.append(block("Workload features", t["modality"] == MOD_WORK))
    m = modality_from_table(t).sort_values("share_percent", ascending=False).iloc[0]
    out.append(f"Largest modality for this record: {m['modality']} ({m['share_percent']:.0f}% of total absolute attribution).")
    out.append("These are model-behaviour attributions on synthetic data, not causal explanations of real injuries.")
    return out


# ------------------------------------------------------------------------------------ permutation importance
def permutation_modality_importance(predictor, arrays: dict, y: np.ndarray, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    model = predictor.get_model("multimodal_dl")
    tin = to_tensors(arrays)
    from src.models.multimodal_model import predict_logits
    base = predict_logits(model, tin).numpy()
    base_auc, base_pr = roc_auc_score(y, base), average_precision_score(y, base)
    groups = {MOD_STATIC: ["static"], MOD_WORK: ["wl_seq", "wl_sum"], MOD_WELL: ["wb_seq"]}
    rows = []
    for mod, keys in groups.items():
        d_auc, d_pr = [], []
        for _ in range(repeats):
            perm = torch.as_tensor(rng.permutation(len(y)))
            pin = {k: (v[perm] if k in keys else v) for k, v in tin.items()}
            s = predict_logits(model, pin).numpy()
            d_auc.append(base_auc - roc_auc_score(y, s)); d_pr.append(base_pr - average_precision_score(y, s))
        rows.append({"modality": mod, "roc_auc_drop_mean": np.mean(d_auc), "roc_auc_drop_std": np.std(d_auc),
                     "pr_auc_drop_mean": np.mean(d_pr), "pr_auc_drop_std": np.std(d_pr)})
    return pd.DataFrame(rows)


def permutation_feature_importance_xgb(predictor, arrays: dict, y: np.ndarray, repeats: int, seed: int) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    X = feature_matrix(arrays, "all")
    r = permutation_importance(predictor.get_model("xgboost"), X, y, scoring="average_precision", n_repeats=repeats, random_state=seed, n_jobs=1)
    names = all_feature_names(predictor.preprocessor)
    t = pd.DataFrame({"feature": [n.split("] ", 1)[1] for n in names], "modality": [{"static": MOD_STATIC, "workload": MOD_WORK, "wellness": MOD_WELL}[n[1:n.index("]")]] for n in names],
                      "pr_auc_drop_mean": r.importances_mean, "pr_auc_drop_std": r.importances_std})
    return t.sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------ plotting
GREEN, BLUE, AMBER = "#1f9d6b", "#3b7dd8", "#d98e2b"


def _bar(df: pd.DataFrame, value_col: str, title: str, path: Path, top: int = 15, xlabel: str = "mean |attribution|"):
    d = df.head(top).iloc[::-1]
    colours = d["modality"].map({MOD_STATIC: BLUE, MOD_WORK: GREEN, MOD_WELL: AMBER})
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    ax.barh(d["feature"], d[value_col], color=colours)
    ax.set(title=title, xlabel=xlabel)
    for m, c in ((MOD_STATIC, BLUE), (MOD_WORK, GREEN), (MOD_WELL, AMBER)):
        ax.barh([], [], color=c, label=m)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _modality_bar(df: pd.DataFrame, col: str, title: str, path: Path, xlabel: str):
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.barh(df["modality"][::-1], df[col][::-1], color=[{MOD_STATIC: BLUE, MOD_WORK: GREEN, MOD_WELL: AMBER}[m] for m in df["modality"][::-1]])
    ax.set(title=title, xlabel=xlabel); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------ CLI
def run_explainability(cfg: dict | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    ex = cfg["explainability"]
    out = resolve_path(cfg, "explain_dir")
    out.mkdir(parents=True, exist_ok=True)
    predictor = InjuryRiskPredictor(cfg)
    df = load_dataset(resolve_path(cfg, "dataset"))
    assign = pd.read_csv(resolve_path(cfg, "metrics_dir") / "split_assignment.csv")
    test_df = df[df["observation_id"].isin(assign.loc[assign["split"] == "test", "observation_id"])].reset_index(drop=True)
    y_test = test_df["injury_next_7_days"].to_numpy()
    arrays = predictor.preprocessor.transform(test_df)
    rng = np.random.default_rng(cfg["project"]["seed"])
    sel = np.sort(rng.choice(len(test_df), size=min(ex["n_explain"], len(test_df)), replace=False))
    summary: dict = {"data_label": "SYNTHETIC", "n_explained_rows": int(len(sel)), "methods": {}}

    # ---- 1. SHAP TreeExplainer on XGBoost (all aggregated features)
    import shap
    X = feature_matrix(arrays, "all")
    names_all = all_feature_names(predictor.preprocessor)
    feats = [n.split("] ", 1)[1] for n in names_all]
    mods = [{"static": MOD_STATIC, "workload": MOD_WORK, "wellness": MOD_WELL}[n[1:n.index("]")]] for n in names_all]
    sv = shap.TreeExplainer(predictor.get_model("xgboost")).shap_values(X[sel])
    g = global_table(sv, feats, mods); g.to_csv(out / "shap_global_importance.csv", index=False)
    m = modality_table(sv, mods); m.to_csv(out / "shap_modality_importance.csv", index=False)
    _bar(g, "mean_abs_attribution", "Global feature importance - SHAP (XGBoost, synthetic data)", out / "shap_global_importance.png", xlabel="mean |SHAP value| (log-odds)")
    plt.figure(figsize=(8, 6.5))
    shap.summary_plot(sv, X[sel], feature_names=feats, max_display=15, show=False)
    plt.title("SHAP summary plot (XGBoost, synthetic data)"); plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=130); plt.close()
    _modality_bar(m, "share_percent", "Modality importance - SHAP", out / "shap_modality_importance.png", "% of total |SHAP|")
    summary["methods"]["shap_tree_xgboost"] = {"status": "ok", "top_feature": g.iloc[0]["feature"]}

    # ---- 2. Integrated Gradients on the multimodal network
    tin = to_tensors({k: v[sel] for k, v in arrays.items() if k in KEYS})
    attr = integrated_gradients(predictor.get_model("multimodal_dl"), tin, ex["ig_steps"])
    total = sum((attr[k].reshape(len(sel), -1)).sum(1) for k in KEYS)
    completeness = float(np.mean(np.abs(total - attr["_delta"])))
    vals, f2, m2 = ig_feature_matrix(attr, predictor.preprocessor.nn_feature_names)
    g2 = global_table(vals, f2, m2); g2.to_csv(out / "ig_global_importance.csv", index=False)
    mm = modality_table(vals, m2); mm.to_csv(out / "ig_modality_importance.csv", index=False)
    _bar(g2, "mean_abs_attribution", "Global feature importance - Integrated Gradients (multimodal DL)", out / "ig_global_importance.png", xlabel="mean |attribution| (logit)")
    _modality_bar(mm, "share_percent", "Modality importance - Integrated Gradients", out / "ig_modality_importance.png", "% of total |attribution|")
    day_attr = {"workload_by_day": np.abs(attr["wl_seq"]).sum(2).mean(0), "wellness_by_day": np.abs(attr["wb_seq"]).sum(2).mean(0)}
    pd.DataFrame({"day_index_oldest_to_newest": range(1, 29), "mean_abs_attribution": day_attr["workload_by_day"]}).to_csv(out / "ig_workload_by_day.csv", index=False)
    pd.DataFrame({"day_index_oldest_to_newest": range(1, 8), "mean_abs_attribution": day_attr["wellness_by_day"]}).to_csv(out / "ig_wellness_by_day.csv", index=False)
    summary["methods"]["integrated_gradients_multimodal_dl"] = {"status": "ok", "steps": ex["ig_steps"], "completeness_mean_abs_error": completeness,
                                                                   "top_feature": g2.iloc[0]["feature"]}

    # ---- 3. permutation importance (model-agnostic fallback / sanity check)
    pm = permutation_modality_importance(predictor, arrays, y_test, ex["permutation_repeats"], cfg["project"]["seed"])
    pm.to_csv(out / "permutation_modality_importance_multimodal_dl.csv", index=False)
    pf = permutation_feature_importance_xgb(predictor, arrays, y_test, max(2, ex["permutation_repeats"] // 2), cfg["project"]["seed"])
    pf.to_csv(out / "permutation_feature_importance_xgboost.csv", index=False)
    _bar(pf, "pr_auc_drop_mean", "Permutation importance - XGBoost (drop in PR-AUC)", out / "permutation_feature_importance_xgboost.png", xlabel="PR-AUC drop")
    summary["methods"]["permutation"] = {"status": "ok", "note": "computed on the full held-out test set"}

    # ---- 4. example individual explanation (highest-estimated-risk test record)
    probs = pd.read_csv(resolve_path(cfg, "predictions_dir") / "multimodal_dl_test_predictions.csv")
    top = probs.sort_values("prob", ascending=False).iloc[0]["observation_id"]
    row = test_df[test_df["observation_id"] == top]
    for name in ("multimodal_dl", "xgboost"):
        t, method = explain_single(predictor, row, name)
        t.to_csv(out / f"example_individual_{name}.csv", index=False)
        d = t.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.barh(d["feature"], d["attribution"], color=[GREEN if v > 0 else BLUE for v in d["attribution"]])
        ax.axvline(0, color="k", lw=0.6); ax.set_title(f"Individual explanation ({top}, SYNTHETIC)\n{method}", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"example_individual_{name}.png", dpi=130); plt.close(fig)
    summary["example_observation_id"] = top

    # ---- 5. optional SHAP GradientExplainer for the NN (compatibility check; IG above is the tested attribution)
    summary["methods"]["shap_gradient_explainer_multimodal_dl"] = _try_shap_gradient(predictor, arrays, sel, ex, out)

    summary["caveat"] = "Attributions describe model behaviour on synthetic data and are not evidence of causation."
    (out / "explainability_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if verbose:
        print(json.dumps(summary, indent=2, default=float))
        print(f"Artifacts written to {out}")
    return summary


def _try_shap_gradient(predictor, arrays, sel, ex, out) -> dict:
    """SHAP GradientExplainer on the multi-input network; reported honestly as ok/unavailable."""
    try:
        import shap

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
            def forward(self, a, b, c, d):
                return self.m(a, b, c, d).unsqueeze(-1)

        n_bg = min(ex["n_background"], len(sel))
        bg = [torch.as_tensor(arrays[k][sel[:n_bg]]) for k in KEYS]
        xs = [torch.as_tensor(arrays[k][sel[:min(60, len(sel))]]) for k in KEYS]
        wrapped = Wrap(predictor.get_model("multimodal_dl")).eval()
        sv = shap.GradientExplainer(wrapped, bg).shap_values(xs)
        sv = [np.asarray(s).reshape(len(xs[0]), *xs[i].shape[1:], -1)[..., 0] for i, s in enumerate(sv)]
        rows = []
        for k, a in zip(KEYS, sv):
            rows.append({"modality": MODALITY_OF_INPUT[k], "input": k, "mean_abs_shap": float(np.abs(a).reshape(len(a), -1).sum(1).mean())})
        t = pd.DataFrame(rows).groupby("modality", as_index=False)["mean_abs_shap"].sum()
        t["share_percent"] = 100 * t["mean_abs_shap"] / t["mean_abs_shap"].sum()
        t.to_csv(out / "shap_gradient_modality_importance_multimodal_dl.csv", index=False)
        return {"status": "ok", "n_background": n_bg, "n_explained": len(xs[0])}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:200]}", "fallback": "Integrated Gradients + permutation importance"}


if __name__ == "__main__":
    run_explainability()
