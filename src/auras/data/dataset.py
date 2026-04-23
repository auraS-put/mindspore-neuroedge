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
        - ``y`` or a custom ``label_key``: int32 array of shape ``(N,)``

    Parameters
    ----------
    npz_path : str | Path
        Path to the ``.npz`` file produced by ``scripts/prepare_dataset.py``.
    indices : array-like, optional
        Subset of sample indices to expose (for train/val/test splits).
    transform : callable, optional
        On-the-fly augmentation applied to each ``(x, y)`` pair.
    label_key : str, optional
        Which key to use as the target label array.  Defaults to ``"y"``.
        Set to ``"y_sop5"`` / ``"y_sop10"`` / ``"y_sop15"`` for the merged NPZ.
    """

    def __init__(
        self,
        npz_path: str | Path,
        indices: Optional[np.ndarray] = None,
        transform=None,
        _preloaded: Optional[dict] = None,
        label_key: str = "y",
    ) -> None:
        # Accept a pre-loaded dict to avoid re-reading the file for every model.
        # mmap_mode='r' maps the file without reading it all into RAM when not preloaded.
        if _preloaded is not None:
            data = _preloaded
        else:
            data = np.load(npz_path, mmap_mode='r')
        self._X = data["X"]
        self._y = data[label_key]
        # subjects array added by updated prepare_dataset.py (Sprint 1)
        self._subjects = data["subjects"] if "subjects" in data else np.zeros(len(self._y), dtype=np.int32)
        self._indices = indices if indices is not None else np.arange(len(self._y))
        self._transform = transform

    @property
    def subjects(self) -> np.ndarray:
        """Integer subject IDs for all samples (used by LOSO splits)."""
        return self._subjects[self._indices]

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
    preloaded: Optional[dict] = None,
    label_key: str = "y",
):
    """Wrap :class:`EEGWindowDataset` in a MindSpore GeneratorDataset.

    Pass ``preloaded=np.load(path, mmap_mode='r')`` to avoid loading the
    file once per model when running multiple models in sequence.
    """
    import mindspore.dataset as ds

    source = EEGWindowDataset(npz_path, indices=indices, transform=transform,
                              _preloaded=preloaded, label_key=label_key)
    dataset = ds.GeneratorDataset(
        source=source,
        column_names=["eeg", "label"],
        shuffle=shuffle,
        num_parallel_workers=num_workers,
    )
    # drop_remainder=False when dataset is smaller than one batch to avoid empty iterator
    drop = len(source) >= batch_size
    dataset = dataset.batch(batch_size, drop_remainder=drop)
    return dataset
