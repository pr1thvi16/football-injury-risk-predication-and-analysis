"""Multimodal PyTorch network: static encoder + two recurrent encoders -> fusion -> logit.

        static ----> StaticEncoder ----------------------+
        workload (28x8) -> GRU/LSTM (+attention) + 7 summaries -> embedding --+--> concat -> Fusion MLP -> logit
        wellness (7x7)  -> GRU/LSTM (+attention) -> embedding ----------------+

Layer sizes are implementation choices (the paper does not specify them); see config/config.yaml.
The network outputs RAW LOGITS: use BCEWithLogitsLoss for training and sigmoid only for inference.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

ENCODERS = ("static_encoder", "workload_encoder", "wellness_encoder")


@dataclass
class ModelConfig:
    static_dim: int
    wl_channels: int = 8
    wl_summary_dim: int = 7
    wb_channels: int = 7
    static_hidden: int = 64
    static_embed: int = 32
    seq_hidden: int = 48
    seq_embed: int = 32
    rnn_type: str = "gru"
    attention: bool = True
    fusion_hidden: list = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3

    @staticmethod
    def from_config(cfg: dict, static_dim: int) -> "ModelConfig":
        m = cfg["multimodal_model"]
        return ModelConfig(static_dim=static_dim, static_hidden=m["static_hidden"], static_embed=m["static_embed"],
                           seq_hidden=m["seq_hidden"], seq_embed=m["seq_embed"], rnn_type=m["rnn_type"],
                           attention=m["attention"], fusion_hidden=list(m["fusion_hidden"]), dropout=m["dropout"])


class StaticEncoder(nn.Module):
    """Input BatchNorm -> Linear -> ReLU -> BatchNorm -> Dropout -> Linear -> ReLU (static embedding)."""

    def __init__(self, in_dim: int, hidden: int, embed: int, dropout: float):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, embed)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.drop(self.bn(torch.relu(self.fc1(x))))
        return torch.relu(self.fc2(x))


class SequenceEncoder(nn.Module):
    """GRU/LSTM over the daily sequence with optional attention pooling; optional side vector of summaries."""

    def __init__(self, in_channels: int, hidden: int, embed: int, rnn_type: str, attention: bool, dropout: float, side_dim: int = 0):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.rnn = rnn_cls(in_channels, hidden, batch_first=True)
        self.use_attention = attention
        self.attn = nn.Linear(hidden, 1) if attention else None
        self.proj = nn.Linear(hidden + side_dim, embed)
        self.drop = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, seq, side=None):
        out, _ = self.rnn(seq)                           # (N, T, H)
        if self.use_attention:
            w = torch.softmax(self.attn(out).squeeze(-1), dim=1)   # (N, T) attention over days
            self.last_attention = w.detach()
            pooled = (out * w.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = out[:, -1, :]
        if side is not None:
            pooled = torch.cat([pooled, side], dim=1)
        return torch.relu(self.proj(self.drop(pooled)))


class MultimodalInjuryNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.static_encoder = StaticEncoder(cfg.static_dim, cfg.static_hidden, cfg.static_embed, cfg.dropout)
        self.workload_encoder = SequenceEncoder(cfg.wl_channels, cfg.seq_hidden, cfg.seq_embed, cfg.rnn_type,
                                                cfg.attention, cfg.dropout, side_dim=cfg.wl_summary_dim)
        self.wellness_encoder = SequenceEncoder(cfg.wb_channels, cfg.seq_hidden, cfg.seq_embed, cfg.rnn_type,
                                                cfg.attention, cfg.dropout)
        layers, d = [], cfg.static_embed + 2 * cfg.seq_embed
        for h in cfg.fusion_hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = h
        self.fusion = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self._frozen: set[str] = set()

    def embeddings(self, static, wl_seq, wl_sum, wb_seq):
        return (self.static_encoder(static), self.workload_encoder(wl_seq, wl_sum), self.wellness_encoder(wb_seq))

    def forward(self, static, wl_seq, wl_sum, wb_seq):
        s, w, b = self.embeddings(static, wl_seq, wl_sum, wb_seq)
        combined = torch.cat([s, w, b], dim=1)           # combined_embedding = concat(static, workload, wellness)
        return self.head(self.fusion(combined)).squeeze(-1)   # raw logits (N,)

    # ---- freezing: frozen encoders get no gradient AND keep BatchNorm/Dropout in eval mode
    def freeze(self, *names: str) -> None:
        for n in names:
            if n not in ENCODERS:
                raise ValueError(f"Unknown component '{n}', choose from {ENCODERS}")
            for p in getattr(self, n).parameters():
                p.requires_grad = False
            self._frozen.add(n)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for n in self._frozen:
            getattr(self, n).eval()
        return self


@torch.no_grad()
def predict_logits(model: nn.Module, inputs: dict, batch_size: int = 1024):
    model.eval()
    n = len(inputs["static"])
    out = [model(**{k: v[i:i + batch_size] for k, v in inputs.items()}) for i in range(0, n, batch_size)]
    return torch.cat(out)


def predict_proba(model: nn.Module, inputs: dict):
    return torch.sigmoid(predict_logits(model, inputs))


def to_tensors(arrays: dict, keys=("static", "wl_seq", "wl_sum", "wb_seq")) -> dict:
    return {k: torch.as_tensor(arrays[k], dtype=torch.float32) for k in keys}


def save_model(model: MultimodalInjuryNet, path_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pt, js = prefix.with_suffix(".pt"), prefix.with_name(prefix.name + "_config.json")
    torch.save(model.state_dict(), pt)
    js.write_text(json.dumps(asdict(model.cfg), indent=2))
    return pt, js


def load_model(path_prefix: str | Path) -> MultimodalInjuryNet:
    prefix = Path(path_prefix)
    pt, js = prefix.with_suffix(".pt"), prefix.with_name(prefix.name + "_config.json")
    for p in (pt, js):
        if not p.exists():
            raise FileNotFoundError(f"Model artifact missing: {p}. Train first: python -m src.training.train")
    try:
        cfg = ModelConfig(**json.loads(js.read_text()))
        model = MultimodalInjuryNet(cfg)
        model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise RuntimeError(f"Model artifact {pt} is corrupted or incompatible: {exc}") from exc
    model.eval()
    return model
