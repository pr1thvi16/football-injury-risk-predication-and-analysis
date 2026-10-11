import numpy as np
import pytest
import torch
from torch import nn

from src.models.multimodal_model import (
    ModelConfig, MultimodalInjuryNet, load_model, predict_logits, predict_proba, save_model,
)


def batch(n=16, ds=16):
    g = torch.Generator().manual_seed(0)
    return {"static": torch.randn(n, ds, generator=g), "wl_seq": torch.randn(n, 28, 8, generator=g),
            "wl_sum": torch.randn(n, 7, generator=g), "wb_seq": torch.randn(n, 7, 7, generator=g)}


@pytest.mark.parametrize("rnn", ["gru", "lstm"])
@pytest.mark.parametrize("attn", [True, False])
def test_forward_shapes_and_probabilities(rnn, attn):
    model = MultimodalInjuryNet(ModelConfig(static_dim=16, rnn_type=rnn, attention=attn)).eval()
    x = batch()
    logits = model(**x)
    assert logits.shape == (16,)                                  # raw logits, one per record
    p = predict_proba(model, x)
    assert ((p >= 0) & (p <= 1)).all()
    assert torch.allclose(p, torch.sigmoid(logits), atol=1e-6)
    s, w, b = model.embeddings(**x)
    assert s.shape == (16, 32) and w.shape == (16, 32) and b.shape == (16, 32)   # three encoder embeddings


def test_attention_weights_sum_to_one():
    model = MultimodalInjuryNet(ModelConfig(static_dim=16)).eval()
    model(**batch())
    assert model.workload_encoder.last_attention.shape == (16, 28)
    assert torch.allclose(model.workload_encoder.last_attention.sum(1), torch.ones(16), atol=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(1)
    model = MultimodalInjuryNet(ModelConfig(static_dim=16)).eval()
    x = batch()
    save_model(model, tmp_path / "m")
    loaded = load_model(tmp_path / "m")
    assert torch.allclose(predict_logits(model, x), predict_logits(loaded, x), atol=1e-6)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train first"):
        load_model(tmp_path / "nothing")
    model = MultimodalInjuryNet(ModelConfig(static_dim=16))
    save_model(model, tmp_path / "m")
    (tmp_path / "m.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupted"):
        load_model(tmp_path / "m")


def test_frozen_components_are_not_updated():
    torch.manual_seed(0)
    model = MultimodalInjuryNet(ModelConfig(static_dim=16))
    model.freeze("static_encoder")
    before = {k: v.clone() for k, v in model.static_encoder.state_dict().items()}
    other_before = model.head.weight.clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    model.train()
    x = batch(32)
    loss = nn.BCEWithLogitsLoss()(model(**x), torch.randint(0, 2, (32,)).float())
    opt.zero_grad(); loss.backward(); opt.step()
    for k, v in model.static_encoder.state_dict().items():           # weights AND BatchNorm running stats unchanged
        assert torch.equal(v, before[k]), k
    assert not torch.equal(model.head.weight, other_before)
    with pytest.raises(ValueError):
        model.freeze("nonexistent")


def test_loss_decreases_on_learnable_toy_problem():
    torch.manual_seed(0)
    x = batch(256)
    y = (x["wl_seq"][:, -3:, 7].mean(1) + x["static"][:, 0] > 0).float()
    model = MultimodalInjuryNet(ModelConfig(static_dim=16, dropout=0.1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lf = nn.BCEWithLogitsLoss()
    first = None
    for _ in range(40):
        model.train(); loss = lf(model(**x), y); opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.8
