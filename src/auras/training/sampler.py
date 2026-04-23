"""Weighted random sampler for class-imbalanced datasets.

Computes per-sample weights from inverse class frequency so the
DataLoader sees approximately balanced mini-batches.
"""

from __future__ import annotations

import numpy as np


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute per-sample weights from label array.

    Weight for class c = total_samples / (num_classes * count_c).
    Each sample gets the weight of its class.

    Parameters
    ----------
    y : 1-D int array
        Label array.

    Returns
    -------
    weights : 1-D float array (same length as y)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)

    class_weight = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    weights = np.array([class_weight[label] for label in y], dtype=np.float64)
    return weights


def build_weighted_sampler(y: np.ndarray):
    """Build a MindSpore WeightedRandomSampler from labels.

    Parameters
    ----------
    y : 1-D int array of training labels.

    Returns
    -------
    sampler : mindspore.dataset.Sampler
    """
    import mindspore.dataset as ds

    weights = compute_sample_weights(y)
    return ds.WeightedRandomSampler(weights=weights.tolist(), num_samples=len(y))
