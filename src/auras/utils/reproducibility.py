"""Reproducibility: deterministic seeds for all frameworks."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and MindSpore."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import mindspore

        mindspore.set_seed(seed)
    except ImportError:
        pass
