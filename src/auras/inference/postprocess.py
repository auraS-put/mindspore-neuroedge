"""Post-processing utilities for seizure prediction inference.

Raw per-window predictions are noisy; these post-processors reduce
false positives (FP/h) without significantly hurting sensitivity.

Implementations based on:
    Paper 01 (Ingolfsson et al. — BrainFuseNet):
        3-window majority voting + collar-based event merging.
    Paper 02 (Busia et al. — EEGformer):
        Majority voting for FP reduction on MCU deployment.
    Paper 03 (Spahr et al. — Episave):
        Ensemble quantile aggregation — tunable sensitivity vs. FAR trade-off.
    Paper 06 (Jana & Mukherjee — NSGA-II):
        Majority voting as standard post-processing for FP reduction.
    Paper 08 (Xie et al. — TSS3D):
        EMA smoothing α=0.3 before threshold for temporal consistency.

All functions are pure NumPy — no framework dependency.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Majority Voting
# ---------------------------------------------------------------------------

def majority_vote(
    predictions: np.ndarray,
    n: int = 3,
    k: int = 2,
) -> np.ndarray:
    """Sliding-window majority vote: alarm only if ≥ k of n consecutive windows positive.

    Reduces isolated single-window false alarms that appear for only one
    window before returning to negative.  Used across Papers 01, 02, 06
    as the standard clinical post-processing step.

    Parameters
    ----------
    predictions : 1-D int array (0/1)
        Per-window binary predictions.
    n : int
        Window size for voting.  Default 3 (Papers 01, 02, 06).
    k : int
        Minimum positives required within the window to trigger an alarm.
        Default 2 (majority threshold for n=3).

    Returns
    -------
    1-D int array — smoothed predictions (same length as input).

    Examples
    --------
    >>> majority_vote(np.array([0, 1, 0, 1, 1, 0]), n=3, k=2)
    array([0, 0, 0, 1, 1, 0])
    """
    predictions = np.asarray(predictions, dtype=np.int32)
    if len(predictions) < n:
        return predictions.copy()
    out = np.zeros_like(predictions)
    half = n // 2
    for i in range(len(predictions)):
        start = max(0, i - half)
        end = min(len(predictions), i + half + 1)
        window = predictions[start:end]
        out[i] = 1 if window.sum() >= k else 0
    return out


# ---------------------------------------------------------------------------
# 2. EMA Smoothing
# ---------------------------------------------------------------------------

def ema_smooth(
    probabilities: np.ndarray,
    alpha: float = 0.3,
    threshold: float = 0.5,
) -> np.ndarray:
    """Exponential Moving Average smoothing of probability scores.

    Applies a causal EMA to the raw probability sequence and then
    thresholds to produce binary predictions.  Paper 08 (Xie et al. — TSS3D)
    uses α=0.3 before threshold to enforce temporal consistency.

    EMA(t) = α × p(t) + (1 − α) × EMA(t−1)

    Parameters
    ----------
    probabilities : 1-D float array
        Raw probability scores for the positive (seizure) class.
    alpha : float
        Smoothing factor in (0, 1].  Closer to 1 → less smoothing.
        Default 0.3 (Paper 08 (Xie et al. — TSS3D)).
    threshold : float
        Decision threshold on smoothed probabilities.  Default 0.5.

    Returns
    -------
    1-D int array — binary predictions after smoothing + thresholding.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ema = np.empty_like(probabilities)
    ema[0] = probabilities[0]
    for i in range(1, len(probabilities)):
        ema[i] = alpha * probabilities[i] + (1.0 - alpha) * ema[i - 1]
    return (ema >= threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 3. Ensemble Quantile Aggregation
# ---------------------------------------------------------------------------

def quantile_aggregate(
    ensemble_probs: np.ndarray,
    quantile: float = 0.6,
) -> np.ndarray:
    """Tunable threshold via quantile aggregation for multi-model ensembles.

    Harrell-Davis style aggregation: apply a percentile threshold across
    ensemble member probabilities rather than a fixed 0.5 cut-off.
    Allows continuous trade-off between sensitivity and FP/h at deployment
    time — Paper 03 (Spahr et al. — Episave).

    Parameters
    ----------
    ensemble_probs : 2-D float array of shape (n_models, n_windows)
        Each row is a model's probability sequence for the seizure class.
    quantile : float in (0, 1)
        Fraction of ensemble members that must agree for a positive call.
        0.5 = simple majority; 0.6 = 60th-percentile threshold (less sensitive,
        fewer FP).  Default 0.6 (conservative clinical deployment, Paper 03 (Spahr et al. — Episave)).

    Returns
    -------
    1-D float array of aggregated probabilities (length n_windows).
    Apply a further threshold (e.g. ≥ 0.5) for binary alarms.

    Notes
    -----
    If a single model is passed (shape (n_windows,)), it is returned unchanged.
    """
    ensemble_probs = np.asarray(ensemble_probs, dtype=np.float64)
    if ensemble_probs.ndim == 1:
        return ensemble_probs  # single model — nothing to aggregate
    # Column-wise quantile across models
    return np.percentile(ensemble_probs, quantile * 100, axis=0)


# ---------------------------------------------------------------------------
# 4. Collar-based Event Merging
# ---------------------------------------------------------------------------

def collar_merge(
    predictions: np.ndarray,
    collar_windows: int = 3,
) -> np.ndarray:
    """Merge predicted seizure events that are separated by a short gap.

    A short break between two positive runs (≤ collar_windows) is filled in
    as positive — preventing a single event from being counted as two separate
    alarms.  Papers 01 and 02 use this to compute event-level FP/h.

    Parameters
    ----------
    predictions : 1-D int array (0/1)
    collar_windows : int
        Maximum gap (in windows) to bridge between consecutive positive runs.
        Default 3 (~12 s at 4 s/window).

    Returns
    -------
    1-D int array — predictions with small gaps filled in.
    """
    predictions = np.asarray(predictions, dtype=np.int32).copy()
    n = len(predictions)
    i = 0
    while i < n:
        if predictions[i] == 1:
            # Find end of this positive run
            j = i
            while j < n and predictions[j] == 1:
                j += 1
            # Look for another positive run within collar distance
            gap_start = j
            gap_end = gap_start
            while gap_end < n and predictions[gap_end] == 0:
                gap_end += 1
                if gap_end - gap_start > collar_windows:
                    break
            if gap_end < n and gap_end - gap_start <= collar_windows:
                predictions[gap_start:gap_end] = 1  # fill gap
            i = gap_end
        else:
            i += 1
    return predictions


# ---------------------------------------------------------------------------
# 5. Threshold sweep (sensitivity–specificity trade-off analysis)
# ---------------------------------------------------------------------------

def threshold_sweep(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
    n_thresholds: int | None = None,
    post_fn=None,
) -> list:
    """Sweep decision thresholds and compute metrics at each point.

    Useful for selecting the operating threshold that satisfies a clinical
    FP/h budget (e.g. ≤ 0.5/h) while maximising sensitivity.

    Parameters
    ----------
    probabilities : 1-D float array
        Raw probability scores for the positive class.
    y_true : 1-D int array
        Ground-truth labels.
    thresholds : 1-D float array, optional
        Thresholds to evaluate.  Default: 51 evenly-spaced points in [0, 1].
    post_fn : callable, optional
        Post-processing function applied to binary predictions before metric
        computation, e.g. ``majority_vote``.

    Returns
    -------
    list of dicts, each with keys: threshold, recall, specificity, precision, f1, fpr, auc_roc
    """
    from auras.training.metrics import compute_metrics

    if thresholds is None:
        n = n_thresholds if n_thresholds is not None else 51
        thresholds = np.linspace(0.0, 1.0, n)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    results = []
    for thr in thresholds:
        y_pred = (probabilities >= thr).astype(np.int32)
        if post_fn is not None:
            y_pred = post_fn(y_pred)
        metrics = compute_metrics(y_true, y_pred, probabilities)
        row = {"threshold": float(thr)}
        row.update(metrics.to_dict())
        results.append(row)
    return results
