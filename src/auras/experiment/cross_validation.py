"""Cross-validation strategies for seizure detection.

Implements Leave-One-Subject-Out (LOSO) and stratified K-fold,
which are the standard evaluation protocols in EEG research.
"""

from __future__ import annotations

from typing import Generator, Tuple

import numpy as np


def loso_splits(
    subject_ids: np.ndarray, y: np.ndarray
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Leave-One-Subject-Out cross-validation.

    Yields (train_indices, test_indices) for each held-out subject.
    This is the gold standard for patient-independent evaluation.

    Parameters
    ----------
    subject_ids : 1-D array
        Per-sample subject identifier (e.g. "PN00", "PN01", ...).
    y : 1-D array
        Labels (for information; not used for splitting).
    """
    unique_subjects = np.unique(subject_ids)
    for held_out in unique_subjects:
        test_mask = subject_ids == held_out
        train_mask = ~test_mask
        # Skip if held-out subject has no seizures (uninformative fold)
        if y[test_mask].sum() == 0:
            continue
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def stratified_kfold_splits(
    y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Stratified K-Fold preserving class proportions.

    Parameters
    ----------
    y : 1-D array
        Labels.
    n_splits : int
        Number of folds.
    seed : int
        Random seed.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        yield train_idx, test_idx
