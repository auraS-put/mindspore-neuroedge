"""Tests for preprocessing utilities."""

import numpy as np
from auras.data.preprocess import zscore, sliding_window


def test_zscore_normalizes():
    x = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    z = zscore(x)
    assert abs(z.mean()) < 1e-5
    assert abs(z.std() - 1.0) < 0.1


def test_sliding_window_shapes():
    x = np.random.randn(4, 1000).astype(np.float32)
    windows = sliding_window(x, window=256, stride=128)
    assert windows.ndim == 3
    assert windows.shape[1] == 4   # channels preserved
    assert windows.shape[2] == 256 # window length


def test_sliding_window_short_signal():
    x = np.random.randn(4, 100).astype(np.float32)
    windows = sliding_window(x, window=256, stride=128)
    assert windows.shape[0] == 0
