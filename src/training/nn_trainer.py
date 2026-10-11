"""Training loop for the multimodal network (CPU-friendly, early stopping, checkpointing)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.models.multimodal_model import MultimodalInjuryNet, ModelConfig, predict_logits

KEYS = ("static", "wl_seq", "wl_sum", "wb_seq")


def _safe_auc(y, s):
    return (roc_auc_score(y, s), average_precision_score(y, s)) if len(np.unique(y)) > 1 else (float("nan"), float("nan"))


def train_multimodal(model_cfg: ModelConfig, train_in: dict, y_train: np.ndarray, val_in: dict, y_val: np.ndarray,
                     cfg: dict, pos_weight: float, seed: int, checkpoint_path: Path | None = None,
                     max_epochs: int | None = None, verbose: bool = True):
    t = cfg["nn_training"]
    torch.manual_seed(seed)
    model = MultimodalInjuryNet(model_cfg)
    params = [p for p in model.parameters() if p.requires_grad]            # frozen parts are never updated
    opt = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=t["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))   # raw logits, no sigmoid
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32)
    n, bs = len(yt), t["batch_size"]
    gen = torch.Generator().manual_seed(seed)
    best_loss, best_state, bad, history = float("inf"), None, 0, []
    epochs = max_epochs or t["max_epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:                                   # BatchNorm needs >1 sample
                continue
            logits = model(**{k: train_in[k][idx] for k in KEYS})
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, t["grad_clip"])
            opt.step()
            running += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            tr_logits = predict_logits(model, train_in)
            va_logits = predict_logits(model, val_in)
            tr_loss, va_loss = loss_fn(tr_logits, yt).item(), loss_fn(va_logits, yv).item()
        tr_auc, tr_pr = _safe_auc(y_train, tr_logits.numpy())
        va_auc, va_pr = _safe_auc(y_val, va_logits.numpy())
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, train_batch_loss=running / n,
                            train_roc_auc=tr_auc, val_roc_auc=va_auc, train_pr_auc=tr_pr, val_pr_auc=va_pr))
        if verbose:
            print(f"  epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f} | val ROC-AUC {va_auc:.3f} | val PR-AUC {va_pr:.3f}")
        if va_loss < best_loss - 1e-4:
            best_loss, bad = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            bad += 1
            if bad >= t["patience"]:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best val loss {best_loss:.4f})")
                break
    model.load_state_dict(best_state)                          # restore best checkpoint
    model.eval()
    return model, pd.DataFrame(history)
