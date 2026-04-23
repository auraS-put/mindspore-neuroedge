"""Tests for evaluation metrics."""

import numpy as np
from auras.training.metrics import compute_metrics


def test_perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m.recall == 1.0
    assert m.precision == 1.0
    assert m.f1 == 1.0
    assert m.fpr == 0.0


def test_all_false_negatives():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m.recall == 0.0  # missed all seizures


def test_with_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m.auc_roc > 0.5
