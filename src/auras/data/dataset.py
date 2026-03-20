"""MindSpore-compatible dataset for EEG seizure windows.

Loads preprocessed ``.npz`` files and exposes them through the
:class:`mindspore.dataset.GeneratorDataset` interface for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class EEGWindowDataset:
    """Iterable dataset over pre-computed EEG windows.

    Expects ``data.npz`` with keys:
        - ``X``: float32 array of shape ``(N, C, T)``
        - ``y``: int32 array of shape ``(N,)``

    Parameters
    ----------
    npz_path : str | Path
        Path to the ``.npz`` file produced by ``scripts/prepare_dataset.py``.
    indices : array-like, optional
        Subset of sample indices to expose (for train/val/test splits).
    transform : callable, optional
        On-the-fly augmentation applied to each ``(x, y)`` pair.
    """

    def __init__(
        self,
        npz_path: str | Path,
        indices: Optional[np.ndarray] = None,
        transform=None,
    ) -> None:
        data = np.load(npz_path)
        self._X = data["X"]
        self._y = data["y"]
        self._indices = indices if indices is not None else np.arange(len(self._y))
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        real_idx = self._indices[idx]
        x = self._X[real_idx].astype(np.float32)
        y = np.array(self._y[real_idx], dtype=np.int32)
        if self._transform is not None:
            x, y = self._transform(x, y)
        return x, y


def build_mindspore_dataset(
    npz_path: str | Path,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
):
    """Wrap :class:`EEGWindowDataset` in a MindSpore GeneratorDataset.

    Returns a fully configured dataset pipeline ready for ``model.train()``.
    """
    import mindspore.dataset as ds

    source = EEGWindowDataset(npz_path, indices=indices, transform=transform)
    dataset = ds.GeneratorDataset(
        source=source,
        column_names=["eeg", "label"],
        shuffle=shuffle,
        num_parallel_workers=num_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
