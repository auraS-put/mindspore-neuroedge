"""Tests for EEGWindowDataset."""

import numpy as np
import tempfile
from pathlib import Path

from auras.data.dataset import EEGWindowDataset


def test_dataset_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.random.randn(100, 4, 256).astype(np.float32)
        y = np.random.randint(0, 2, size=100).astype(np.int32)
        npz_path = Path(tmpdir) / "test.npz"
        np.savez(str(npz_path), X=X, y=y)

        ds = EEGWindowDataset(npz_path)
        assert len(ds) == 100
        x_sample, y_sample = ds[0]
        assert x_sample.shape == (4, 256)
        assert y_sample.shape == ()


def test_dataset_with_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.random.randn(50, 4, 256).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.int32)
        npz_path = Path(tmpdir) / "test.npz"
        np.savez(str(npz_path), X=X, y=y)

        indices = np.array([0, 5, 10, 15])
        ds = EEGWindowDataset(npz_path, indices=indices)
        assert len(ds) == 4
