"""Reproducibility helpers (seeding, deterministic CPU PyTorch)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except ImportError:  # torch is optional for data-only use
        pass
