"""Streamlit dashboard:  streamlit run app.py

Research prototype on SYNTHETIC data. Every prediction shown here is produced by the saved, trained models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import RESEARCH_DISCLAIMER, RISK_COLOURS, ROOT, load_config, resolve_path, risk_category
from src.data.data_dictionary import build_dictionary
from src.data.validation import DataValidationError, load_dataset
from src.evaluation.metrics import compute_metrics, curves
from src.explainability.shap_explainer import answer_questions, explain_single, modality_from_table
from src.features.feature_engineering import (
    POSITIONS, SUMMARY_FEATURES, WELLNESS_VARS, WORKLOAD_CHANNELS, compute_average_rest_days, compute_bmi,
    compute_workload_summaries, frame_from_sequences, sequences_from_frame,
)
from src.inference import ArtifactError, InjuryRiskPredictor
from src.reporting import build_report, report_to_csv, report_to_pdf

st.set_page_config(page_title="Football Injury Risk | Research Prototype", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

BG, CARD, BORDER, TEXT, MUTED = "#0B1220", "#111B2E", "#1F2D47", "#E6EDF7", "#8FA1BD"
GREEN, BLUE, AMBER, RED = "#2EE59D", "#5B9CF5", "#F5B84B", "#F0616D"
MODALITY_COLOURS = {"Static & medical": BLUE, "Workload": GREEN, "Wellness": AMBER}
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
                "mlp": "MLP", "workload_only": "Workload-only", "wellness_only": "Wellness-only",
                "static_only": "Static-only", "multimodal_dl": "Multimodal DL"}
cfg = load_config()

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; height:100%; }}
.kpi .l {{ color:{MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em; }}
.kpi .v {{ color:{TEXT}; font-size:1.65rem; font-weight:650; line-height:1.25; }}
.kpi .s {{ color:{MUTED}; font-size:0.8rem; }}
.notice {{ background:rgba(245,184,75,.08); border:1px solid rgba(245,184,75,.35); color:{TEXT}; border-radius:10px;
          padding:10px 14px; font-size:.88rem; }}
.badge {{ display:inline-block; padding:4px 12px; border-radius:999px; font-weight:650; color:#06101f; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid {GREEN}; color:{GREEN}; font-size:.72rem; letter-spacing:.08em; }}
section[data-testid="stSidebar"] {{ border-right:1px solid {BORDER}; }}
div[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:10px 14px; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------- helpers
def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def notice(text: str) -> None:
    st.markdown(f'<div class="notice">{text}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(color=TEXT, size=12),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig


def show(fig: go.Figure, height: int = 330, key: str | None = None) -> None:
    st.plotly_chart(style(fig, height), width="stretch", key=key)


def table(df: pd.DataFrame, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading synthetic dataset…")
def load_df(_mt: float) -> pd.DataFrame:
    return load_dataset(resolve_path(cfg, "dataset"))


@st.cache_resource(show_spinner="Loading trained models…")
def get_predictor(_mt: float) -> InjuryRiskPredictor:
    return InjuryRiskPredictor(cfg)


@st.cache_data
def read_csv(path: str, _mt: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def read_json(path: str, _mt: float) -> dict:
    return json.loads(Path(path).read_text())


def rcsv(path: Path) -> pd.DataFrame | None:
    return read_csv(str(path), mtime(path)) if path.exists() else None


DATASET_PATH = resolve_path(cfg, "dataset")
MODELS_DIR, METRICS_DIR = resolve_path(cfg, "models_dir"), resolve_path(cfg, "metrics_dir")
PRED_DIR, FIG_DIR, EXPL_DIR = resolve_path(cfg, "predictions_dir"), resolve_path(cfg, "figures_dir"), resolve_path(cfg, "explain_dir")

FIRST_RUN = """```bash
python -m src.data.generate_dataset
python -m src.data.validation
python -m src.training.train
python -m src.explainability.shap_explainer
streamlit run app.py
```"""

if not DATASET_PATH.exists():
    st.title("⚽ Football injury-risk research prototype")
    st.error(f"Dataset not found at `{DATASET_PATH.relative_to(ROOT)}`. Run the first-run workflow:")
    st.markdown(FIRST_RUN)
    st.stop()
try:
    df = load_df(mtime(DATASET_PATH))
except DataValidationError as exc:
    st.error(str(exc)); st.stop()

predictor: InjuryRiskPredictor | None = None
predictor_error = None
if (MODELS_DIR / "model_registry.json").exists():
    try:
        predictor = get_predictor(mtime(MODELS_DIR / "model_registry.json"))
    except (FileNotFoundError, ArtifactError, RuntimeError) as exc:
        predictor_error = str(exc)
else:
    predictor_error = "No trained models found. Train them with `python -m src.training.train`."

assign = rcsv(METRICS_DIR / "split_assignment.csv")
split_of = dict(zip(assign["observation_id"], assign["split"])) if assign is not None else {}
comparison = rcsv(METRICS_DIR / "model_comparison.csv")


def need_models() -> InjuryRiskPredictor:
    if predictor is None:
        st.error(predictor_error or "Models unavailable.")
        st.markdown("Run the first-run workflow:"); st.markdown(FIRST_RUN)
        st.stop()
    return predictor


def pick_record(prefix: str, default_last: bool = True) -> pd.DataFrame:
    c1, c2 = st.columns([1, 1])
    players = sorted(df["player_id"].unique())
    pid = c1.selectbox("Synthetic player", players, key=f"{prefix}_player")
    weeks = df[df["player_id"] == pid].sort_values("week_index")
    dates = list(weeks["observation_date"])
    d = c2.selectbox("Observation date (weekly assessment)", dates, index=len(dates) - 1 if default_last else 0, key=f"{prefix}_date")
    return weeks[weeks["observation_date"] == d].iloc[[0]].reset_index(drop=True)


def input_summary(row: pd.DataFrame) -> dict:
    wl, wb = sequences_from_frame(row)
    s = compute_workload_summaries(wl).iloc[0]
    r = row.iloc[0]
    return {"age": int(r["age"]), "position": r["position"], "previous_injury_count": int(r["previous_injury_count"]),
            "days_since_last_injury": int(r["days_since_last_injury"]),
            "strength_asymmetry_percent": round(float(r["strength_asymmetry_percent"]), 1),
            "acute_workload (AU/day)": round(float(s["acute_workload"]), 1), "chronic_workload (AU/day)": round(float(s["chronic_workload"]), 1),
            "acute_chronic_workload_ratio": round(float(s["acute_chronic_workload_ratio"]), 2),
            "mean_fatigue_7d (1-10, higher=worse)": round(float(wb[0, :, 2].mean()), 2),
            "mean_sleep_hours_7d": round(float(wb[0, :, 0].mean()), 2),
            "mean_recovery_7d (0-100, higher=better)": round(float(wb[0, :, 5].mean()), 1)}


def make_hypothetical(row: pd.DataFrame, load_scale: float, recent_scale: float, wellness: dict, static: dict) -> pd.DataFrame:
    wl, wb = sequences_from_frame(row)
    wl, wb = wl.copy(), wb.copy()
    f = np.full(28, load_scale); f[-7:] *= recent_scale
    for c in range(0, 6):                                  # distances, counts and duration scale; RPE stays
        wl[0, :, c] = wl[0, :, c] * f
    wl[0, :, 3:5] = np.round(wl[0, :, 3:5]); wl[0, :, 5] = np.round(wl[0, :, 5])
    wl[0, :, 7] = wl[0, :, 6] * wl[0, :, 5]                # load = rpe * duration
    for name, target in wellness.items():
        i = WELLNESS_VARS.index(name)
        lo, hi = {"sleep_hours": (3.5, 10.5), "recovery_score": (0, 100)}.get(name, (1, 10))
        wb[0, :, i] = np.clip(wb[0, :, i] + (target - wb[0, :, i].mean()), lo, hi)
    out = row.copy()
    for k, v in static.items():
        out[k] = v
    out["bmi"] = compute_bmi(out["height_cm"], out["weight_kg"])
    seq = frame_from_sequences(wl, wb)
    out[seq.columns] = seq.to_numpy()
    out[SUMMARY_FEATURES] = compute_workload_summaries(wl).to_numpy()
    out["average_rest_days"] = compute_average_rest_days(wl)
    out["player_id"], out["observation_id"] = "HYPOTHETICAL", "HYPO-0001"
    return out


# --------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚽ Injury-Risk Lab")
    st.markdown('<span class="pill">SYNTHETIC DATA</span>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Player Risk Prediction", "Workload Analysis", "Wellness Analysis", "Model Performance",
                                 "Explainable AI", "Dataset Explorer", "Research Methodology"], key="page", label_visibility="collapsed")
    st.divider()
    if predictor is not None:
        st.caption(f"Selected model: **{MODEL_LABELS.get(predictor.selected_model, predictor.selected_model)}**")
        st.caption(f"Decision threshold: **{predictor.threshold():.3f}** ({predictor.registry['threshold_strategy']})")
    st.caption("Research prototype — not a clinically validated tool.")

st.markdown(f"<h2 style='margin-bottom:0'>{page}</h2>", unsafe_allow_html=True)
st.caption("Multimodal deep learning for explainable 7-day injury-risk prediction · synthetic player-week data")

# =============================================================================================== OVERVIEW
if page == "Overview":
    notice("<b>All default player records are SYNTHETIC</b> (simulated). They are not from any football club. "
           "Predictions, probabilities and risk categories are research outputs — not medical diagnoses or return-to-play decisions.")
    st.write("")
    n_pos = int(df["injury_next_7_days"].sum())
    cols = st.columns(5)
    for c, (l, v, s) in zip(cols, [("Player-week records", f"{len(df):,}", "synthetic"), ("Synthetic players", f"{df['player_id'].nunique()}", f"{df['week_index'].nunique()} weeks each"),
                                   ("Positive (injury)", f"{n_pos:,}", "target = 1"), ("Negative", f"{len(df) - n_pos:,}", "target = 0"),
                                   ("Injury-label prevalence", pct(n_pos / len(df), 2), "next 7 days")]):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Model status")
        if predictor is None:
            st.error(predictor_error)
        else:
            reg = predictor.registry
            sel = reg["selected_model"]
            st.markdown(f"**Selected model:** {MODEL_LABELS.get(sel, sel)} &nbsp;·&nbsp; **Trained:** {reg['trained_at'][:19].replace('T', ' ')} UTC "
                        f"&nbsp;·&nbsp; **Threshold:** {reg['thresholds'][sel]:.3f} ({reg['threshold_strategy']}, set on validation data)")
            if reg.get("smoke_run"):
                st.warning("These artifacts come from a quick smoke run. Run `python -m src.training.train` for the full training.")
            else:
                st.success(f"Training complete — NN best epoch {reg['nn_best_epoch']} of {reg['nn_epochs_run']} (early stopping on validation loss).")
            sp = reg["split"]
            st.caption(f"Split strategy: **{sp['strategy']}** — " + ", ".join(f"{k}: {v['n_records']:,} ({v['n_players']} players)" for k, v in sp["splits"].items()))
            st.caption("The default model is the proposed multimodal model, chosen in advance — not because it scored highest on the test set.")
    with right:
        st.subheader("Latest held-out test metrics")
        if comparison is not None and predictor is not None:
            r = comparison[comparison["model"] == predictor.selected_model].iloc[0]
            a, b, c = st.columns(3)
            a.metric("ROC-AUC", f"{r['roc_auc']:.3f}"); b.metric("PR-AUC", f"{r['pr_auc']:.3f}"); c.metric("Brier", f"{r['brier']:.3f}")
            a, b, c = st.columns(3)
            a.metric("Recall", f"{r['recall']:.3f}"); b.metric("Precision", f"{r['precision']:.3f}"); c.metric("F1", f"{r['f1']:.3f}")
            st.caption(f"At threshold {r['threshold']:.3f}. Test set: {int(r['n_test']):,} synthetic records, {int(r['n_test_positive'])} positives. "
                       "Accuracy is not emphasised because injury is rare.")
        else:
            st.info("Metrics will appear after training.")
    st.subheader("Target prevalence by position (synthetic labels)")
    g = df.groupby("position")["injury_next_7_days"].agg(["mean", "size"]).reset_index()
    fig = go.Figure(go.Bar(x=g["position"], y=g["mean"] * 100, marker_color=GREEN, text=[f"{v:.1f}%" for v in g["mean"] * 100], textposition="outside"))
    fig.update_layout(yaxis_title="Injury label %")
    show(fig, 280)

# =============================================================================================== PLAYER RISK
elif page == "Player Risk Prediction":
    pred = need_models()
    notice("Risk estimates are produced by the saved trained model on <b>synthetic</b> inputs. They are research outputs, not diagnoses.")
    st.write("")
    mode = st.radio("Input", ["Synthetic dataset record", "Hypothetical player (edit inputs)"], horizontal=True)
    base = pick_record("risk")
    hypothetical = mode.startswith("Hypothetical")
    row = base
    if hypothetical:
        with st.expander("Edit the hypothetical player (starts from the selected record)", expanded=True):
            a, b = st.columns(2)
            with a:
                st.markdown("**Static & medical**")
                r0 = base.iloc[0]
                s_age = st.slider("Age", 18, 38, int(r0["age"]))
                s_pos = st.selectbox("Position", POSITIONS, index=POSITIONS.index(r0["position"]))
                s_prev = st.slider("Previous injuries", 0, 8, int(r0["previous_injury_count"]))
                s_since = st.slider("Days since last injury", 0, 730, int(r0["days_since_last_injury"]))
                s_asym = st.slider("Strength asymmetry (%)", 0.0, 35.0, float(r0["strength_asymmetry_percent"]), 0.5)
                st.markdown("**Workload (28 days)**")
                l_scale = st.slider("Overall load multiplier", 0.5, 1.6, 1.0, 0.05)
                r_scale = st.slider("Last-7-day load multiplier (spike)", 0.5, 2.0, 1.0, 0.05)
            with b:
                st.markdown("**Wellness (7-day mean)**")
                _, wb0 = sequences_from_frame(base)
                m0 = wb0[0].mean(axis=0)
                well = {
                    "sleep_hours": st.slider("Sleep hours", 4.0, 10.0, float(round(m0[0], 1)), 0.1),
                    "sleep_quality": st.slider("Sleep quality (1-10, higher=better)", 1.0, 10.0, float(round(m0[1], 1)), 0.1),
                    "fatigue_score": st.slider("Fatigue (1-10, higher=worse)", 1.0, 10.0, float(round(m0[2], 1)), 0.1),
                    "muscle_soreness_score": st.slider("Muscle soreness (1-10, higher=worse)", 1.0, 10.0, float(round(m0[3], 1)), 0.1),
                    "mood_score": st.slider("Mood (1-10, higher=better)", 1.0, 10.0, float(round(m0[4], 1)), 0.1),
                    "recovery_score": st.slider("Recovery (0-100, higher=better)", 0.0, 100.0, float(round(m0[5], 0)), 1.0),
                    "perceived_stress_score": st.slider("Perceived stress (1-10, higher=worse)", 1.0, 10.0, float(round(m0[6], 1)), 0.1),
                }
        row = make_hypothetical(base, l_scale, r_scale, well, {"age": s_age, "position": s_pos, "previous_injury_count": s_prev,
                                                              "days_since_last_injury": s_since, "strength_asymmetry_percent": s_asym})
    r = row.iloc[0]
    obs_id = base.iloc[0]["observation_id"]
    split = split_of.get(obs_id, "n/a")
    if not hypothetical and split == "train":
        st.caption("ℹ️ This record was in the **training** split, so its prediction is in-sample. Pick a player from the test split for an honest check.")
    elif not hypothetical and split in ("val", "test"):
        st.caption(f"ℹ️ This record belongs to the **{split}** split (the model never trained on this player).")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Static & medical features")
        table(pd.DataFrame({"feature": ["age", "position", "height_cm", "weight_kg", "bmi", "previous_injury_count", "previous_injury_days_missed",
                                        "days_since_last_injury", "season_minutes_played", "strength_asymmetry_percent", "average_rest_days"],
                            "value": [r["age"], r["position"], r["height_cm"], r["weight_kg"], round(float(r["bmi"]), 1), r["previous_injury_count"],
                                      r["previous_injury_days_missed"], r["days_since_last_injury"], r["season_minutes_played"],
                                      round(float(r["strength_asymmetry_percent"]), 1), round(float(r["average_rest_days"]), 2)]}).astype({"value": str}))
    wl, wb = sequences_from_frame(row)
    with c2:
        st.subheader("Previous 28 days of workload")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        days = list(range(-28, 0))
        fig.add_trace(go.Bar(x=days, y=wl[0, :, 7], name="Session load (AU)", marker_color=GREEN, opacity=0.8))
        fig.add_trace(go.Scatter(x=days, y=wl[0, :, 0], name="Total distance (km)", line=dict(color=BLUE, width=2)), secondary_y=True)
        fig.update_xaxes(title="Days before observation date")
        show(fig, 260)
        st.subheader("Previous 7 days of wellness")
        fig = go.Figure()
        for i, v in enumerate(["fatigue_score", "muscle_soreness_score", "mood_score", "sleep_quality", "perceived_stress_score"]):
            idx = WELLNESS_VARS.index(v)
            fig.add_trace(go.Scatter(x=list(range(-7, 0)), y=wb[0, :, idx], name=v.replace("_", " "), mode="lines+markers"))
        show(fig, 250)

    st.divider()
    model_name = st.selectbox("Model", pred.available_models, index=pred.available_models.index(pred.selected_model),
                              format_func=lambda m: MODEL_LABELS.get(m, m))
    key = (hypothetical, obs_id, model_name, json.dumps(row.iloc[0][SUMMARY_FEATURES + ["age", "previous_injury_count", "days_since_last_injury"]].astype(float).round(4).tolist()))
    if st.button("Generate injury-risk estimate", type="primary"):
        try:
            res = pred.predict(row, model_name).iloc[0]
            try:
                contrib, method = explain_single(pred, row, model_name)
            except NotImplementedError as exc:
                contrib, method = None, str(exc)
            st.session_state["risk_result"] = dict(key=key, res=res, contrib=contrib, method=method)
        except (DataValidationError, FileNotFoundError, ArtifactError, ValueError, RuntimeError) as exc:
            st.error(f"Prediction failed: {exc}")
            st.session_state.pop("risk_result", None)
    out = st.session_state.get("risk_result")
    if out and out["key"] == key:
        res, contrib, method = out["res"], out["contrib"], out["method"]
        p, thr, cat = float(res["probability"]), float(res["threshold"]), res["risk_category"]
        g1, g2 = st.columns([1, 1.2])
        with g1:
            steps = []
            cats = sorted(cfg["risk_categories"], key=lambda c: c["lower"])
            for i, c in enumerate(cats):
                hi = cats[i + 1]["lower"] if i + 1 < len(cats) else 1.0
                steps.append(dict(range=[100 * c["lower"], 100 * hi], color=RISK_COLOURS.get(c["name"], MUTED)))
            fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number=dict(suffix="%", valueformat=".1f"),
                                         gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#E6EDF7", thickness=0.25), steps=steps,
                                                    threshold=dict(line=dict(color="white", width=3), value=100 * thr))))
            show(fig, 260)
            st.caption("White line = model decision threshold. Colour bands = illustrative risk categories.")
        with g2:
            st.markdown(f"**Estimated 7-day injury probability:** {pct(p)}")
            st.markdown(f'**Risk category:** <span class="badge" style="background:{RISK_COLOURS.get(cat, MUTED)}">{cat}</span>', unsafe_allow_html=True)
            st.markdown(f"**Decision threshold:** {thr:.3f} → predicted label **{'AT RISK' if res['predicted_label'] == 1 else 'below threshold'}**")
            st.caption(f"Model: {MODEL_LABELS.get(model_name, model_name)} · probability is Platt-calibrated on validation data · "
                       "category cut-offs are illustrative (config/config.yaml), not clinically validated.")
            if not hypothetical:
                st.caption(f"Simulated outcome label for this record: {int(base.iloc[0]['injury_next_7_days'])} (synthetic).")
        if contrib is not None:
            st.subheader("Important contributing features")
            st.caption(f"Method: {method}. Contributions describe model behaviour (association), not causes of injury.")
            top = contrib.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["attribution"], y=top["feature"], orientation="h",
                                   marker_color=[MODALITY_COLOURS[m] for m in top["modality"]],
                                   customdata=np.stack([top["modality"], top["value"].astype(str)], axis=1),
                                   hovertemplate="%{y}<br>contribution %{x:+.3f}<br>%{customdata[0]} · value %{customdata[1]}<extra></extra>"))
            fig.update_layout(xaxis_title="Contribution to model output (logit / log-odds)")
            show(fig, 380)
            m = modality_from_table(contrib)
            st.caption("Modality shares of total |contribution|: " + " · ".join(f"{a}: {b:.0f}%" for a, b in zip(m["modality"], m["share_percent"])))
            for line in answer_questions(contrib):
                st.markdown(f"- {line}")
        else:
            st.info(method)
        rep = build_report(player_id=str(r["player_id"]), observation_date=str(base.iloc[0]["observation_date"]), model_name=model_name,
                           probability=p, threshold=thr, risk_category=cat, predicted_label=int(res["predicted_label"]),
                           input_summary=input_summary(row), contributions=None if contrib is None else contrib.head(10), explanation_method=method if contrib is not None else None,
                           hypothetical=hypothetical)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇ Report (CSV)", report_to_csv(rep), f"injury_risk_{r['player_id']}.csv", "text/csv")
        d2.download_button("⬇ Report (PDF)", report_to_pdf(rep), f"injury_risk_{r['player_id']}.pdf", "application/pdf")
    else:
        st.info("Press **Generate injury-risk estimate** to run the saved model on the record above.")

# =============================================================================================== WORKLOAD
elif page == "Workload Analysis":
    row = pick_record("wl")
    wl, _ = sequences_from_frame(row)
    summ = compute_workload_summaries(wl).iloc[0]
    days = list(range(-28, 0))
    cols = st.columns(6)
    for c, (l, v, s) in zip(cols, [("Acute (7d)", f"{summ['acute_workload']:.0f}", "AU/day"), ("Chronic (28d)", f"{summ['chronic_workload']:.0f}", "AU/day"),
                                   ("ACWR", f"{summ['acute_chronic_workload_ratio']:.2f}", "candidate feature only"), ("Monotony", f"{summ['workload_monotony']:.2f}", "mean/std 7d"),
                                   ("Strain", f"{summ['workload_strain']:.0f}", "7d load × monotony"), ("Variability", f"{summ['workload_variability']:.2f}", "CV, 28d")]):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)
    st.caption("ACWR is shown as one candidate workload feature — it is not a scientifically definitive injury predictor.")
    a, b = st.columns(2)
    with a:
        st.subheader("Daily training load")
        fig = go.Figure(go.Bar(x=days, y=wl[0, :, 7], marker_color=GREEN, name="Session load (AU)"))
        roll = pd.Series(wl[0, :, 7]).rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=days, y=roll, name="7-day rolling mean", line=dict(color=AMBER, width=2)))
        show(fig, 300)
    with b:
        st.subheader("Total & high-speed running distance")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=days, y=wl[0, :, 0], name="Total distance (km)", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=days, y=wl[0, :, 1], name="High-speed running (km)", line=dict(color=GREEN)), secondary_y=True)
        fig.add_trace(go.Scatter(x=days, y=wl[0, :, 2], name="Sprint (km)", line=dict(color=RED, dash="dot")), secondary_y=True)
        show(fig, 300)
    a, b = st.columns(2)
    with a:
        st.subheader("Accelerations & decelerations")
        fig = go.Figure([go.Scatter(x=days, y=wl[0, :, 3], name="Accelerations", line=dict(color=GREEN)),
                         go.Scatter(x=days, y=wl[0, :, 4], name="Decelerations", line=dict(color=AMBER))])
        show(fig, 280)
    with b:
        st.subheader("Rest-day analysis")
        dur = wl[0, :, 5]
        wk = [f"Week -{4 - i}" for i in range(4)]
        rest = [(dur[i * 7:(i + 1) * 7] == 0).sum() for i in range(4)]
        fig = go.Figure([go.Bar(x=wk, y=rest, name="Rest days", marker_color=BLUE), go.Bar(x=wk, y=[7 - r for r in rest], name="Training/match days", marker_color=GREEN)])
        fig.update_layout(barmode="stack")
        show(fig, 280)
        st.caption(f"Average rest days per week over 28 days: **{compute_average_rest_days(wl)[0]:.2f}**")
    st.subheader("Weekly ACWR trend for this player (all assessments)")
    pw = df[df["player_id"] == row.iloc[0]["player_id"]].sort_values("week_index")
    fig = go.Figure(go.Scatter(x=pw["observation_date"], y=pw["acute_chronic_workload_ratio"], mode="lines+markers", line=dict(color=GREEN)))
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor=GREEN, opacity=0.08, line_width=0, annotation_text="0.8–1.3 (illustrative band)", annotation_position="top left")
    show(fig, 280)
    st.subheader("Population context: ACWR distribution (all synthetic records)")
    fig = go.Figure(go.Histogram(x=df["acute_chronic_workload_ratio"], nbinsx=60, marker_color=BLUE))
    fig.add_vline(x=float(summ["acute_chronic_workload_ratio"]), line_color=AMBER, annotation_text="this record")
    show(fig, 240)

# =============================================================================================== WELLNESS
elif page == "Wellness Analysis":
    row = pick_record("wb")
    _, wb = sequences_from_frame(row)
    labels = {"sleep_hours": ("Sleep hours", "higher = better"), "sleep_quality": ("Sleep quality (1–10)", "higher = better"),
              "fatigue_score": ("Fatigue (1–10)", "higher = worse"), "muscle_soreness_score": ("Muscle soreness (1–10)", "higher = worse"),
              "mood_score": ("Mood (1–10)", "higher = better"), "recovery_score": ("Recovery (0–100)", "higher = better"),
              "perceived_stress_score": ("Perceived stress (1–10)", "higher = worse")}
    cols = st.columns(2)
    for i, v in enumerate(["sleep_hours", "fatigue_score", "muscle_soreness_score", "mood_score", "recovery_score", "perceived_stress_score"]):
        idx = WELLNESS_VARS.index(v)
        fig = go.Figure(go.Scatter(x=list(range(-7, 0)), y=wb[0, :, idx], mode="lines+markers", line=dict(color=[GREEN, AMBER, RED, BLUE, GREEN, AMBER][i], width=3)))
        fig.update_layout(title=f"{labels[v][0]} — {labels[v][1]}", xaxis_title="Days before observation")
        with cols[i % 2]:
            show(fig, 240, key=f"well_{v}")
    st.divider()
    st.subheader("Wellness vs model-estimated risk (held-out test records)")
    pred = predictor
    if pred is None or not (PRED_DIR / f"{pred.selected_model}_test_predictions.csv").exists():
        st.info("Train the models to see this analysis.")
    else:
        model_for = st.selectbox("Model", pred.available_models, index=pred.available_models.index(pred.selected_model), format_func=lambda m: MODEL_LABELS.get(m, m), key="well_model")
        pr = rcsv(PRED_DIR / f"{model_for}_test_predictions.csv")
        feats = df.set_index("observation_id")
        sub = pr.merge(pd.DataFrame({v: feats[[f"wb_{v}_d{k}" for k in range(1, 8)]].mean(axis=1) for v in WELLNESS_VARS}).reset_index(), on="observation_id")
        var = st.selectbox("Wellness variable (7-day mean)", WELLNESS_VARS, index=2, format_func=lambda v: labels[v][0])
        sub["bin"] = pd.qcut(sub[var], 8, duplicates="drop")
        g = sub.groupby("bin", observed=True).agg(x=(var, "mean"), est=("prob", "mean"), label_rate=("y_true", "mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub[var], y=sub["prob"], mode="markers", name="records", marker=dict(color=BLUE, size=4, opacity=0.25)))
        fig.add_trace(go.Scatter(x=g["x"], y=g["est"], name="mean estimated risk (bins)", mode="lines+markers", line=dict(color=GREEN, width=3)))
        fig.add_trace(go.Scatter(x=g["x"], y=g["label_rate"], name="synthetic label rate (bins)", mode="lines+markers", line=dict(color=AMBER, dash="dot")))
        fig.update_layout(xaxis_title=f"{labels[var][0]} — {labels[var][1]}", yaxis_title="probability / rate")
        show(fig, 380)
        rho = sub[[var, "prob"]].corr(method="spearman").iloc[0, 1]
        st.caption(f"Spearman correlation with the model's estimate: **{rho:+.2f}**. This is an association within synthetic data produced by the simulation "
                   "and the model — it must not be read as a causal effect of wellness on injury.")

# =============================================================================================== PERFORMANCE
elif page == "Model Performance":
    pred = need_models()
    if comparison is None:
        st.error("model_comparison.csv not found. Run `python -m src.training.train`."); st.stop()
    st.caption("All numbers are computed from the saved held-out test predictions of each model (same test records for every model) on SYNTHETIC data. "
               f"Thresholds were chosen on validation data using '{pred.registry['threshold_strategy']}'.")
    show_cols = ["model", "modalities", "threshold", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "accuracy"]
    t = comparison[show_cols].copy(); t["model"] = t["model"].map(lambda m: MODEL_LABELS.get(m, m))
    table(t.round(4))
    best = {m: comparison.loc[comparison[m].idxmax(), "model"] for m in ("roc_auc", "pr_auc", "f1", "recall", "precision")}
    st.caption("Best on this test set — " + " · ".join(f"{k.replace('_', '-').upper()}: {MODEL_LABELS[v]}" for k, v in best.items())
               + f" · lowest Brier: {MODEL_LABELS[comparison.loc[comparison['brier'].idxmin(), 'model']]}. "
                 "With 216 test positives, differences of a few AUC points are within sampling noise.")
    metric_sel = st.multiselect("Metrics in bar chart", ["precision", "recall", "f1", "roc_auc", "pr_auc"], default=["precision", "recall", "f1", "roc_auc", "pr_auc"])
    fig = go.Figure([go.Bar(x=[MODEL_LABELS[m] for m in comparison["model"]], y=comparison[k], name=k) for k in metric_sel])
    fig.update_layout(barmode="group", yaxis_range=[0, 1])
    show(fig, 340)

    preds = {m: rcsv(PRED_DIR / f"{m}_test_predictions.csv") for m in comparison["model"]}
    chosen = st.multiselect("Models in curves", list(preds), default=list(preds), format_func=lambda m: MODEL_LABELS[m])
    c1, c2, c3 = st.columns(3)
    figs = {"roc": go.Figure(), "pr": go.Figure(), "cal": go.Figure()}
    for m in chosen:
        p = preds[m]; cv = curves(p["y_true"], p["prob"]); w = 4 if m == "multimodal_dl" else 2
        figs["roc"].add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], name=MODEL_LABELS[m], line=dict(width=w)))
        figs["pr"].add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], name=MODEL_LABELS[m], line=dict(width=w)))
        figs["cal"].add_trace(go.Scatter(x=cv["calib_pred"], y=cv["calib_true"], name=MODEL_LABELS[m], mode="lines+markers", line=dict(width=w)))
    figs["roc"].add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=MUTED))
    figs["cal"].add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=MUTED))
    prev = float(next(iter(preds.values()))["y_true"].mean())
    figs["pr"].add_hline(y=prev, line_dash="dash", line_color=MUTED, annotation_text=f"prevalence {prev:.3f}")
    figs["roc"].update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    figs["pr"].update_layout(title="Precision–recall curves", xaxis_title="Recall", yaxis_title="Precision")
    figs["cal"].update_layout(title="Calibration curves", xaxis_title="Mean predicted probability", yaxis_title="Observed fraction positive")
    with c1: show(figs["roc"], 360, key="roc")
    with c2: show(figs["pr"], 360, key="pr")
    with c3: show(figs["cal"], 360, key="cal")

    st.subheader("Confusion matrix")
    cm_model = st.selectbox("Model", comparison["model"].tolist(), index=comparison["model"].tolist().index(pred.selected_model), format_func=lambda m: MODEL_LABELS[m], key="cm_model")
    r = comparison[comparison["model"] == cm_model].iloc[0]
    mat = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig = go.Figure(go.Heatmap(z=mat, x=["Predicted: no injury", "Predicted: injury"], y=["Actual: no injury", "Actual: injury"], text=mat, texttemplate="%{text}",
                               colorscale=[[0, "#111B2E"], [1, GREEN]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    a, b = st.columns([1, 1])
    with a: show(fig, 300, key="cm")
    with b:
        st.markdown(f"**Threshold:** {r['threshold']:.3f}  \n**False negatives (missed):** {int(r['fn'])}  \n**False positives (false alarms):** {int(r['fp'])}")
        st.caption("Because injury is rare, the false-positive/false-negative trade-off matters more than accuracy: lowering the threshold "
                   "catches more injuries at the cost of more false alarms.")
    st.subheader("Neural-network training history")
    hist = rcsv(METRICS_DIR / "nn_training_history.csv")
    if hist is not None:
        a, b = st.columns(2)
        fig = go.Figure([go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="train"), go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="validation")])
        fig.update_layout(title="Loss (weighted BCE-with-logits)", xaxis_title="Epoch")
        best_e = int(hist.loc[hist["val_loss"].idxmin(), "epoch"]); fig.add_vline(x=best_e, line_dash="dot", line_color=MUTED, annotation_text=f"best epoch {best_e}")
        with a: show(fig, 300, key="hist_loss")
        fig = go.Figure([go.Scatter(x=hist["epoch"], y=hist["train_pr_auc"], name="train PR-AUC"), go.Scatter(x=hist["epoch"], y=hist["val_pr_auc"], name="val PR-AUC"),
                         go.Scatter(x=hist["epoch"], y=hist["val_roc_auc"], name="val ROC-AUC", line=dict(dash="dot"))])
        fig.update_layout(title="Metrics", xaxis_title="Epoch")
        with b: show(fig, 300, key="hist_auc")
    else:
        st.info("Training history not found.")

# =============================================================================================== EXPLAINABLE AI
elif page == "Explainable AI":
    notice("Importances describe <b>model behaviour on synthetic data</b>. They are associations, not causal explanations, and do not show why a real injury would occur. "
           "SHAP (tree model) and Integrated Gradients (neural network) are different methods and are not directly comparable.")
    st.write("")
    summary_p = EXPL_DIR / "explainability_summary.json"
    if not summary_p.exists():
        st.info("Explainability artifacts not found. Generate them with `python -m src.explainability.shap_explainer`.")
    tabs = st.tabs(["Global (SHAP · XGBoost)", "Neural network (Integrated Gradients)", "Modality importance", "Individual prediction", "Method notes"])

    def importance_chart(path: Path, value: str, title: str, key: str, top: int = 15):
        t = rcsv(path)
        if t is None:
            st.info(f"`{path.name}` not available."); return
        t = t.head(top).iloc[::-1]
        fig = go.Figure(go.Bar(x=t[value], y=t["feature"], orientation="h", marker_color=[MODALITY_COLOURS[m] for m in t["modality"]],
                               customdata=t["modality"], hovertemplate="%{y}<br>%{x:.3f}<br>%{customdata}<extra></extra>"))
        fig.update_layout(title=title, xaxis_title="mean |attribution|")
        show(fig, 460, key=key)

    with tabs[0]:
        importance_chart(EXPL_DIR / "shap_global_importance.csv", "mean_abs_attribution", "Global feature importance (mean |SHAP|, log-odds)", "shap_g")
        img = EXPL_DIR / "shap_summary.png"
        if img.exists():
            st.image(str(img), caption="SHAP summary (beeswarm) — each dot is a held-out synthetic record; colour = feature value.")
        pf = EXPL_DIR / "permutation_feature_importance_xgboost.png"
        if pf.exists():
            with st.expander("Cross-check: permutation importance (XGBoost)"):
                st.image(str(pf))
    with tabs[1]:
        importance_chart(EXPL_DIR / "ig_global_importance.csv", "mean_abs_attribution", "Multimodal DL — mean |Integrated Gradients| (logit)", "ig_g")
        a, b = st.columns(2)
        wd, bd = rcsv(EXPL_DIR / "ig_workload_by_day.csv"), rcsv(EXPL_DIR / "ig_wellness_by_day.csv")
        if wd is not None:
            fig = go.Figure(go.Bar(x=wd["day_index_oldest_to_newest"] - 29, y=wd["mean_abs_attribution"], marker_color=GREEN)); fig.update_layout(title="Workload attribution by day", xaxis_title="Days before observation")
            with a: show(fig, 280, key="ig_wd")
        if bd is not None:
            fig = go.Figure(go.Bar(x=bd["day_index_oldest_to_newest"] - 8, y=bd["mean_abs_attribution"], marker_color=AMBER)); fig.update_layout(title="Wellness attribution by day", xaxis_title="Days before observation")
            with b: show(fig, 280, key="ig_bd")
    with tabs[2]:
        st.caption("Share of total absolute attribution per modality. Methods can disagree — a modality that carries many attributions need not be the one whose removal hurts performance most.")
        cols = st.columns(3)
        for c, (title, path, col) in zip(cols, [("SHAP · XGBoost", "shap_modality_importance.csv", "share_percent"),
                                                ("Integrated Gradients · Multimodal DL", "ig_modality_importance.csv", "share_percent"),
                                                ("SHAP GradientExplainer · Multimodal DL", "shap_gradient_modality_importance_multimodal_dl.csv", "share_percent")]):
            t = rcsv(EXPL_DIR / path)
            with c:
                if t is None:
                    st.info(f"{title}: not available")
                else:
                    fig = go.Figure(go.Bar(x=t["modality"], y=t[col], marker_color=[MODALITY_COLOURS[m] for m in t["modality"]], text=[f"{v:.0f}%" for v in t[col]], textposition="outside"))
                    fig.update_layout(title=title, yaxis_title="% of total", yaxis_range=[0, 100]); show(fig, 300, key=f"mod_{path}")
        pm = rcsv(EXPL_DIR / "permutation_modality_importance_multimodal_dl.csv")
        if pm is not None:
            fig = go.Figure([go.Bar(x=pm["modality"], y=pm["pr_auc_drop_mean"], error_y=dict(type="data", array=pm["pr_auc_drop_std"]), name="PR-AUC drop", marker_color=BLUE),
                             go.Bar(x=pm["modality"], y=pm["roc_auc_drop_mean"], error_y=dict(type="data", array=pm["roc_auc_drop_std"]), name="ROC-AUC drop", marker_color=GREEN)])
            fig.update_layout(title="Permutation test: score drop when one modality is shuffled across records (multimodal DL, test set)", barmode="group")
            show(fig, 320, key="perm_mod")
    with tabs[3]:
        pred = need_models()
        if assign is None:
            st.info("Split assignment not found."); st.stop()
        test_ids = assign[assign["split"] == "test"]
        pid = st.selectbox("Test-set player", sorted(test_ids["player_id"].unique()), key="xai_player")
        ids = test_ids[test_ids["player_id"] == pid]["observation_id"].tolist()
        oid = st.selectbox("Observation", ids, index=len(ids) - 1, format_func=lambda i: df.loc[df["observation_id"] == i, "observation_date"].iloc[0], key="xai_obs")
        mdl = st.radio("Model to explain", ["multimodal_dl", "xgboost", "workload_only", "wellness_only", "static_only"], horizontal=True,
                       format_func=lambda m: MODEL_LABELS[m] + (" · Integrated Gradients" if m == "multimodal_dl" else " · SHAP"))
        row = df[df["observation_id"] == oid].reset_index(drop=True)
        res = pred.predict(row, mdl).iloc[0]
        st.markdown(f"**Estimated probability:** {pct(res['probability'])} · **category:** {res['risk_category']} · **threshold:** {res['threshold']:.3f}")
        with st.spinner("Computing attributions…"):
            t, method = explain_single(pred, row, mdl, steps=cfg["explainability"]["ig_steps"])
        st.caption(f"Method: {method}")
        top = t.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["attribution"], y=top["feature"], orientation="h", marker_color=[GREEN if v > 0 else BLUE for v in top["attribution"]],
                               customdata=np.stack([top["modality"], top["value"].astype(str)], axis=1),
                               hovertemplate="%{y}<br>%{x:+.3f}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
        fig.update_layout(xaxis_title="contribution to model output (green raises, blue lowers)")
        show(fig, 420, key="xai_ind")
        for line in answer_questions(t):
            st.markdown(f"- {line}")
        with st.expander("Full attribution table"):
            table(t.round(4))
    with tabs[4]:
        if summary_p.exists():
            info = read_json(str(summary_p), mtime(summary_p))
            st.json(info)
        st.markdown("""
* **SHAP TreeExplainer** gives exact additive log-odds contributions for the XGBoost model on aggregated features.
* **Integrated Gradients** attributes the neural network's raw logit to its inputs (zero baseline = the training mean after scaling);
  its completeness error is reported above. Sequence attributions are summed over days per channel for feature-level views.
* **Permutation importance** shuffles a feature/modality across records and measures the performance drop — a model-agnostic cross-check.
* None of these prove causation, and none were validated on real players.""")

# =============================================================================================== DATASET EXPLORER
elif page == "Dataset Explorer":
    notice("This is the <b>SYNTHETIC</b> dataset produced by <code>src/data/generate_dataset.py</code>. It does not contain real player records.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    q = c1.text_input("Search player identifier", placeholder="e.g. P017")
    pos = c2.multiselect("Position", POSITIONS, default=POSITIONS)
    lab = c3.radio("Injury label", ["All", "Injury (1)", "No injury (0)"], horizontal=True)
    view = df[df["position"].isin(pos)]
    if q:
        view = view[view["player_id"].str.contains(q.strip(), case=False)]
    if lab != "All":
        view = view[view["injury_next_7_days"] == (1 if lab.startswith("Injury") else 0)]
    st.caption(f"{len(view):,} of {len(df):,} records shown · {view['player_id'].nunique()} players · prevalence {pct(view['injury_next_7_days'].mean() if len(view) else 0, 2)}")
    base_cols = ["observation_id", "player_id", "observation_date", "data_source", "age", "position", "height_cm", "weight_kg", "bmi", "previous_injury_count",
                 "previous_injury_days_missed", "days_since_last_injury", "season_minutes_played", "strength_asymmetry_percent", "average_rest_days"] + SUMMARY_FEATURES + ["injury_next_7_days"]
    extra = st.multiselect("Add most-recent-day sequence columns", [c for c in df.columns if c.endswith("_d28") or c.endswith("_d7")])
    table(view[base_cols + extra].head(500))
    st.caption("Preview limited to the first 500 rows; the download contains every row of the current filter.")
    st.download_button("⬇ Download filtered CSV (synthetic)", view.to_csv(index=False).encode(), "football_injury_synthetic_filtered.csv", "text/csv")
    st.subheader("Dataset statistics")
    table(view[base_cols[4:]].describe().T.round(3).reset_index().rename(columns={"index": "feature"}))
    st.subheader("Data dictionary")
    table(build_dictionary())
    rp = METRICS_DIR / "data_quality_report.json"
    if rp.exists():
        with st.expander("Data-quality report"):
            rep = read_json(str(rp), mtime(rp))
            st.markdown(f"Overall: **{'PASS' if rep['all_passed'] else 'FAIL'}** · {rep['n_records']:,} records · {rep['n_positive']:,} positives ({rep['prevalence']:.2%})")
            table(pd.DataFrame(rep["checks"]))

# =============================================================================================== METHODOLOGY
else:
    st.markdown(f"> {RESEARCH_DISCLAIMER}")
    st.markdown("""
#### Research objective
Predict whether a (synthetic) professional football player will sustain a time-loss injury in the **next seven days**, by fusing three
information modalities with a multimodal deep-learning model, and explain the prediction.

#### Data structure
One row = one player-week assessment. Three modalities: **static & medical** features, a **28-day workload** series (8 daily channels), and a
**7-day wellness** series (7 daily scores). Derived workload features (acute/chronic load, ACWR, monotony, strain, variability, recent change) use documented formulas.
All histories end the day *before* the observation date. Labels come from a documented probabilistic mechanism and are never part of the inputs.

#### Preprocessing
Median imputation, standardisation, one-hot position and per-channel sequence scaling — **fitted on the training split only**. Player-grouped splitting keeps
each player in exactly one split. A positive-class weight and post-hoc Platt calibration (validation data) handle class imbalance.

#### Architecture
`Static encoder (BN → FC → ReLU → BN → Dropout → FC)` · `Workload GRU + attention (+ derived summaries)` · `Wellness GRU + attention` → **concatenate** → fusion MLP → logit (sigmoid at inference; BCE-with-logits for training; AdamW; early stopping on validation loss).

#### Baselines
Logistic regression, random forest, XGBoost, MLP (all on aggregated features) and workload-only, wellness-only and static-only XGBoost models — all evaluated on identical test records.

#### Evaluation & explainability
Precision, recall, specificity, F1, ROC-AUC, PR-AUC, Brier score, confusion matrices, ROC/PR and calibration curves; thresholds are chosen on validation data. SHAP, Integrated Gradients and permutation importance explain model behaviour.
""")
    a, b = st.columns(2)
    with a:
        st.subheader("Specified by the project brief (from the paper's methodology)")
        st.markdown("""
- Synthetic dataset of **12,000** player-week records, ≈ **8.98 %** positives (7-day horizon)
- Static/medical + workload (28 d) + wellness (7 d) modalities
- Three encoders (static MLP, workload LSTM/GRU, wellness LSTM/GRU), concatenation fusion, FC fusion head, sigmoid output
- Split sizes **8,160 / 1,440 / 2,400** (68 / 12 / 20 %)
- Baselines: LR, RF, GB/XGBoost, MLP, three unimodal models
- SHAP-based explainability; BCE-with-logits training with Adam/AdamW""")
    with b:
        st.subheader("Additional implementation choices (assumptions)")
        st.markdown("""
- Simulation mechanism, label weights, exact positive count (1,078), player-grouped split and the 250 × 48 layout
- Hidden sizes, dropout 0.3, GRU + attention, learning rate 1e-3, batch 128, patience 10
- Positive weight (neg/pos)^0.5, Platt calibration, validation-F1 decision threshold
- Illustrative risk categories 20 / 40 / 60 %
- Integrated Gradients and permutation importance alongside SHAP
- Feature aggregation for classical models""")
    st.subheader("Limitations")
    st.markdown("""
- All data are **synthetic**; performance says nothing about real players, and learned associations reflect the simulator, not causal mechanisms.
- Injury is rare (≈ 9 %), so the test set has few positives and metric differences between models are uncertain.
- This implementation's results are **not** the IEEE paper's reported results, and are not compared against them here.
- Not clinically validated; external validation on real, prospectively collected data would be required before any practical use.""")
    for name in ("methodology.md", "architecture.md", "limitations.md"):
        p = ROOT / "docs" / name
        if p.exists():
            with st.expander(f"docs/{name}"):
                st.markdown(p.read_text())
