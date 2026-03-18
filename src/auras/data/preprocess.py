from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def sliding_window(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Create [n_windows, channels, window] windows from [channels, time]."""
    if x.ndim != 2:
        raise ValueError("Expected shape [channels, time]")
    channels, time = x.shape
    if time < window:
        return np.empty((0, channels, window), dtype=x.dtype)
    starts = range(0, time - window + 1, stride)
    return np.stack([x[:, s : s + window] for s in starts], axis=0)
