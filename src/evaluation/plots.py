"""Static matplotlib figures saved to results/figures/ (the dashboard draws interactive versions)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import curves

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "font.size": 9})
LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
          "mlp": "MLP", "workload_only": "Workload-only", "wellness_only": "Wellness-only",
          "static_only": "Static-only", "multimodal_dl": "Multimodal DL"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_calibration(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for kind in ("roc", "pr", "calibration"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, p in preds.items():
            c = curves(p["y_true"], p["prob"])
            lw = 2.4 if name == "multimodal_dl" else 1.2
            if kind == "roc":
                ax.plot(c["fpr"], c["tpr"], lw=lw, label=LABELS.get(name, name))
            elif kind == "pr":
                ax.plot(c["recall"], c["precision"], lw=lw, label=LABELS.get(name, name))
            else:
                ax.plot(c["calib_pred"], c["calib_true"], marker="o", ms=3, lw=lw, label=LABELS.get(name, name))
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (synthetic test set)")
        elif kind == "pr":
            base = float(next(iter(preds.values()))["y_true"].mean())
            ax.axhline(base, color="k", ls="--", lw=0.8, label=f"Prevalence {base:.3f}")
            ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (synthetic test set)")
        else:
            ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Calibration curves (synthetic test set)")
        ax.legend(fontsize=7)
        _save(fig, out_dir / {"roc": "roc_curves.png", "pr": "pr_curves.png", "calibration": "calibration_curves.png"}[kind])


def plot_confusion(name: str, cm: dict, threshold: float, out_dir: Path) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(mat, cmap="Greens")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v}", ha="center", va="center", color="black" if v < mat.max() * 0.6 else "white")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No injury", "Injury"], yticklabels=["No injury", "Injury"],
           xlabel="Predicted", ylabel="Actual (synthetic label)", title=f"{LABELS.get(name, name)}\nthreshold={threshold:.3f}")
    _save(fig, out_dir / f"confusion_matrix_{name}.png")


def plot_training_history(hist: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train"); axes[0].plot(hist["epoch"], hist["val_loss"], label="validation")
    axes[0].set(title="Loss (weighted BCE-with-logits)", xlabel="Epoch")
    axes[1].plot(hist["epoch"], hist["train_roc_auc"], label="train"); axes[1].plot(hist["epoch"], hist["val_roc_auc"], label="validation")
    axes[1].set(title="ROC-AUC", xlabel="Epoch")
    axes[2].plot(hist["epoch"], hist["train_pr_auc"], label="train"); axes[2].plot(hist["epoch"], hist["val_pr_auc"], label="validation")
    axes[2].set(title="PR-AUC", xlabel="Epoch")
    best = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
    for a in axes:
        a.axvline(best, color="grey", ls=":", lw=1); a.legend()
    _save(fig, out_dir / "training_history.png")


def plot_model_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df)); w = 0.16
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * w, df[m], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels([LABELS.get(n, n) for n in df["model"]], rotation=25, ha="right")
    ax.set(title="Model comparison on the held-out synthetic test set", ylim=(0, 1)); ax.legend(ncol=5, fontsize=8)
    _save(fig, out_dir / "model_comparison.png")
