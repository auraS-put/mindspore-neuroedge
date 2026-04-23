"""Segment-level and event-level evaluation for seizure prediction.

Two evaluation granularities matter in clinical practice:

    Segment-level: per-window classification metrics (accuracy, recall, …).
    Event-level:   seizure detection rate (SDR) and false positive rate per
                   hour (FP/h) — the clinically primary metric pair.

Paper 01 (Ingolfsson et al. — BrainFuseNet): FP/h primary metric.
Paper 02 (Busia et al. — EEGformer): LOSO evaluation, no cross-patient
    contamination; 15-min postictal exclusion.
Paper 06 (Jana & Mukherjee — NSGA-II): multi-objective sensitivity vs FP/h.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from auras.training.metrics import MetricsResult, compute_metrics


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Combined segment-level and event-level evaluation result.

    Attributes
    ----------
    segment : MetricsResult
        Window-level classification metrics (recall, specificity, …).
    fp_per_hour : float
        Number of false-alarm seizure events per recording hour.
    seizure_detection_rate : float
        Fraction of true seizure events that contain at least one
        predicted-positive window (event-level sensitivity).
    total_recording_hours : float
        Total duration of all evaluated windows combined.
    n_seizures_detected : int
        Count of true seizure events that were detected.
    n_seizures_total : int
        Total number of true seizure events.
    n_false_alarms : int
        Count of predicted events that do not overlap any true seizure.
    """

    segment: MetricsResult
    fp_per_hour: float
    seizure_detection_rate: float
    total_recording_hours: float = 0.0
    n_seizures_detected: int = 0
    n_seizures_total: int = 0
    n_false_alarms: int = 0

    def to_dict(self) -> dict:
        d = self.segment.to_dict()
        d.update(
            {
                "fp_per_hour": self.fp_per_hour,
                "seizure_detection_rate": self.seizure_detection_rate,
                "total_recording_hours": self.total_recording_hours,
                "n_seizures_detected": self.n_seizures_detected,
                "n_seizures_total": self.n_seizures_total,
                "n_false_alarms": self.n_false_alarms,
            }
        )
        return d


@dataclass
class LOSOResult:
    """Aggregated results across all LOSO folds.

    Attributes
    ----------
    fold_results : list of EvaluationResult
        One entry per held-out subject fold.
    subject_ids : list of str, optional
        Identifier of the held-out subject for each fold.
    """

    fold_results: List[EvaluationResult] = field(default_factory=list)
    subject_ids: List[str] = field(default_factory=list)

    def _collect(self, key: str) -> np.ndarray:
        """Gather a scalar metric from all folds into an array."""
        values = []
        for r in self.fold_results:
            d = r.to_dict()
            values.append(d[key])
        return np.array(values, dtype=np.float64)

    def mean_metrics(self) -> dict:
        """Return the mean of each metric over all folds."""
        if not self.fold_results:
            return {}
        keys = self.fold_results[0].to_dict().keys()
        return {k: float(self._collect(k).mean()) for k in keys}

    def std_metrics(self) -> dict:
        """Return the std-dev of each metric over all folds."""
        if not self.fold_results:
            return {}
        keys = self.fold_results[0].to_dict().keys()
        return {k: float(self._collect(k).std()) for k in keys}

    def summary(self) -> str:
        """Human-readable LOSO summary line."""
        if not self.fold_results:
            return "No LOSO folds recorded."
        mean = self.mean_metrics()
        std = self.std_metrics()
        return (
            f"LOSO ({len(self.fold_results)} folds) — "
            f"sensitivity={mean['recall']:.3f}±{std['recall']:.3f}  "
            f"FP/h={mean['fp_per_hour']:.2f}±{std['fp_per_hour']:.2f}  "
            f"SDR={mean['seizure_detection_rate']:.3f}±{std['seizure_detection_rate']:.3f}"
        )


# ---------------------------------------------------------------------------
# Event-level helpers
# ---------------------------------------------------------------------------


def _find_contiguous_events(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Identify contiguous runs of 1s in a 1-D label array.

    Returns
    -------
    list of (start_idx, end_idx) tuples (inclusive, 0-based).
    """
    events: List[Tuple[int, int]] = []
    in_event = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_event:
            in_event = True
            start = i
        elif v == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(labels) - 1))
    return events


def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_length_s: float,
) -> Tuple[float, float, float, int, int, int]:
    """Compute event-level false-positive rate per hour and seizure detection rate.

    A predicted event (run of positive windows) is considered a *true alarm*
    if it overlaps with any true seizure event; otherwise it is a false alarm.
    A true seizure event is *detected* if at least one predicted event overlaps it.

    Parameters
    ----------
    y_true : 1-D int array
        Ground-truth labels (1 = preictal/seizure, 0 = interictal).
    y_pred : 1-D int array
        Predicted labels.
    window_length_s : float
        Duration of each analysis window in seconds.

    Returns
    -------
    fp_per_hour : float
    seizure_detection_rate : float
    total_hours : float
    n_detected : int
    n_total_seizures : int
    n_false_alarms : int
    """
    n_windows = len(y_true)
    total_hours = n_windows * window_length_s / 3600.0

    true_events = _find_contiguous_events(y_true)
    pred_events = _find_contiguous_events(y_pred)

    n_total_seizures = len(true_events)
    detected_set: set = set()
    n_false_alarms = 0

    for p_start, p_end in pred_events:
        is_true_alarm = False
        for i, (t_start, t_end) in enumerate(true_events):
            if p_start <= t_end and p_end >= t_start:   # overlap
                detected_set.add(i)
                is_true_alarm = True
                break
        if not is_true_alarm:
            n_false_alarms += 1

    n_detected = len(detected_set)
    sdr = n_detected / max(n_total_seizures, 1)
    fp_per_hour = n_false_alarms / max(total_hours, 1e-12)

    return fp_per_hour, sdr, total_hours, n_detected, n_total_seizures, n_false_alarms


# ---------------------------------------------------------------------------
# Main evaluation entry-points
# ---------------------------------------------------------------------------


def evaluate_epoch(
    model,
    batch_iterator,
    window_length_s: float = 4.0,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Run model inference over a batch iterator and compute all metrics.

    Parameters
    ----------
    model : nn.Cell (BaseSeizureModel)
        Trained MindSpore model.  Will be put into eval mode.
    batch_iterator : iterable of (Tensor, Tensor)
        Yields ``(X, y)`` mini-batches, e.g. from
        ``dataset.create_tuple_iterator()``.
    window_length_s : float
        Window size in seconds used for event-level FP/h computation.
    threshold : float
        Probability threshold for the positive (seizure) class.

    Returns
    -------
    EvaluationResult
    """
    import mindspore.ops as ops

    all_y_true: list = []
    all_y_pred: list = []
    all_y_prob: list = []

    model.set_train(False)
    for X, y_true in batch_iterator:
        logits = model(X)
        probs = ops.softmax(logits, axis=-1).asnumpy()
        y_pred = (probs[:, 1] >= threshold).astype(np.int32)
        all_y_true.extend(y_true.asnumpy().tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(probs[:, 1].tolist())

    y_true_np = np.array(all_y_true, dtype=np.int32)
    y_pred_np = np.array(all_y_pred, dtype=np.int32)
    y_prob_np = np.array(all_y_prob, dtype=np.float32)

    segment = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    fp_per_hour, sdr, total_hours, n_det, n_total, n_fa = compute_event_metrics(
        y_true_np, y_pred_np, window_length_s
    )

    return EvaluationResult(
        segment=segment,
        fp_per_hour=fp_per_hour,
        seizure_detection_rate=sdr,
        total_recording_hours=total_hours,
        n_seizures_detected=n_det,
        n_seizures_total=n_total,
        n_false_alarms=n_fa,
    )
