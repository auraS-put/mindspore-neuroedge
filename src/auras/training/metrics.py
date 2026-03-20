"""Medical-grade evaluation metrics for seizure detection.

Primary metric: **Recall (Sensitivity)** — missing a seizure is unacceptable.
Secondary: F1, Precision, Specificity, False Positive Rate, AUC-ROC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MetricsResult:
    """Container for a full metric evaluation."""

    accuracy: float
    recall: float       # sensitivity = TP / (TP + FN)
    precision: float
    specificity: float  # TN / (TN + FP)
    f1: float
    fpr: float          # false positive rate = FP / (FP + TN)
    auc_roc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "recall": self.recall,
            "precision": self.precision,
            "specificity": self.specificity,
            "f1": self.f1,
            "fpr": self.fpr,
            "auc_roc": self.auc_roc,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> MetricsResult:
    """Compute all metrics from ground-truth and predictions.

    Parameters
    ----------
    y_true : array of int
        Ground truth labels (0 = normal, 1 = seizure).
    y_pred : array of int
        Predicted class labels.
    y_prob : array of float, optional
        Probability scores for the positive class (for AUC-ROC).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)

    return MetricsResult(
        accuracy=accuracy_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred, zero_division=0.0),
        precision=precision_score(y_true, y_pred, zero_division=0.0),
        specificity=specificity,
        f1=f1_score(y_true, y_pred, zero_division=0.0),
        fpr=fpr,
        auc_roc=auc,
    )
