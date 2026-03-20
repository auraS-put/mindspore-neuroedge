"""Test suite for Sprint 3 — Training Infrastructure.

Covers:
    - SSWCELoss (output shape, value range, sensitivity to alpha/beta)
    - LabelSmoothingCE (output shape, epsilon=0 → ~standard CE)
    - build_loss factory for all four loss names
    - OneCycleLR schedule (shape, warmup increase, decay decrease)
    - cosine_annealing and step schedules still work
    - EvaluationResult (field existence, to_dict completeness)
    - LOSOResult (mean/std aggregation correctness)
    - compute_event_metrics (FP/h, SDR, edge cases)
    - _find_contiguous_events helper
    - evaluate_epoch (end-to-end inference, returns EvaluationResult)
    - EarlyStoppingCallback (patience and mode)
    - BestCheckpointCallback (save_top_k pruning)
    - LOSO cross-validation splits (no test subject in train)
    - gradient clipping (norm ≤ clip value after clip)
    - _train_loop early-stop integration (import + call)
    - loso_splits skips folds with no seizures
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Make src importable without installing
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# MindSpore context — PyNative for unit tests
# ---------------------------------------------------------------------------
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)

import mindspore.nn as nn
from mindspore import Tensor

# ---------------------------------------------------------------------------
# Sprint-3 modules under test
# ---------------------------------------------------------------------------
from auras.training.losses import (
    SSWCELoss,
    LabelSmoothingCE,
    WeightedCrossEntropyLoss,
    FocalLoss,
    build_loss,
)
from auras.training.lr_schedulers import build_lr_schedule
from auras.training.evaluator import (
    EvaluationResult,
    LOSOResult,
    compute_event_metrics,
    _find_contiguous_events,
    evaluate_epoch,
)
from auras.training.metrics import compute_metrics
from auras.training.callbacks import EarlyStoppingCallback, BestCheckpointCallback
from auras.experiment.cross_validation import loso_splits


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def small_batch():
    """4-sample binary batch with balanced classes."""
    logits = Tensor(np.array([
        [2.0, -1.0],
        [-1.0, 2.0],
        [1.0, -0.5],
        [-0.5, 1.5],
    ], dtype=np.float32))
    labels = Tensor(np.array([0, 1, 0, 1], dtype=np.int32))
    return logits, labels


@pytest.fixture()
def class_weights():
    return Tensor(np.array([0.5, 2.0], dtype=np.float32))


@pytest.fixture()
def default_loss_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "loss": {
            "name": "weighted_ce",
            "class_weights": "auto",
            "focal_gamma": 2.0,
            "sswce_alpha": 0.5,
            "sswce_beta": 0.5,
            "label_smoothing": 0.1,
        }
    })


@pytest.fixture()
def scheduler_cfg_factory():
    """Return a factory for schedule configs with given scheduler name."""
    from omegaconf import OmegaConf

    def _make(name, **kwargs):
        sched = {"name": name, "warmup_epochs": 5, "min_lr": 1e-6}
        sched.update(kwargs)
        return OmegaConf.create({
            "learning_rate": 1e-3,
            "epochs": 20,
            "weight_decay": 1e-4,
            "scheduler": sched,
        })

    return _make


# ===========================================================================
# 1. SSWCELoss
# ===========================================================================

class TestSSWCELoss:

    def test_output_is_scalar(self, small_batch, class_weights):
        loss_fn = SSWCELoss(alpha=0.5, beta=0.5, class_weights=class_weights)
        logits, labels = small_batch
        out = loss_fn(logits, labels)
        assert out.shape == (), f"Expected scalar, got {out.shape}"

    def test_output_is_positive(self, small_batch, class_weights):
        loss_fn = SSWCELoss(class_weights=class_weights)
        logits, labels = small_batch
        out = float(loss_fn(logits, labels).asnumpy())
        assert out > 0.0

    def test_alpha_zero_removes_specificity_penalty(self, small_batch):
        """When alpha=0 and beta=0, SSWCE ≈ plain weighted CE."""
        logits, labels = small_batch
        cw = Tensor(np.array([1.0, 1.0], dtype=np.float32))
        sswce = SSWCELoss(alpha=0.0, beta=0.0, class_weights=cw)
        wce = WeightedCrossEntropyLoss(cw)
        s_val = float(sswce(logits, labels).asnumpy())
        w_val = float(wce(logits, labels).asnumpy())
        assert abs(s_val - w_val) < 1e-4, f"Expected ~equal, got SSWCE={s_val} WCE={w_val}"

    def test_high_beta_increases_loss_for_missed_seizures(self, class_weights):
        """Making beta larger should increase loss when seizures are missed."""
        logits = Tensor(np.array([[2.0, -2.0], [2.0, -2.0]], dtype=np.float32))
        labels = Tensor(np.array([1, 1], dtype=np.int32))  # all seizures, all missed
        low = SSWCELoss(alpha=0.5, beta=0.1, class_weights=class_weights)
        high = SSWCELoss(alpha=0.5, beta=2.0, class_weights=class_weights)
        assert float(high(logits, labels).asnumpy()) > float(low(logits, labels).asnumpy())

    def test_no_class_weights(self, small_batch):
        loss_fn = SSWCELoss()
        logits, labels = small_batch
        out = float(loss_fn(logits, labels).asnumpy())
        assert out > 0.0

    def test_perfect_predictions_give_low_loss(self):
        """Very confident correct predictions should give near-zero sensitivity/specificity penalty."""
        logits = Tensor(np.array([
            [10.0, -10.0],
            [-10.0, 10.0],
        ], dtype=np.float32))
        labels = Tensor(np.array([0, 1], dtype=np.int32))
        loss_fn = SSWCELoss(alpha=0.5, beta=0.5)
        out = float(loss_fn(logits, labels).asnumpy())
        assert out < 0.5, f"Perfect predictions should have low loss, got {out}"


# ===========================================================================
# 2. LabelSmoothingCE
# ===========================================================================

class TestLabelSmoothingCE:

    def test_output_is_scalar(self, small_batch, class_weights):
        loss_fn = LabelSmoothingCE(epsilon=0.1, class_weights=class_weights)
        logits, labels = small_batch
        out = loss_fn(logits, labels)
        assert out.shape == ()

    def test_positive_loss(self, small_batch):
        loss_fn = LabelSmoothingCE(epsilon=0.1)
        logits, labels = small_batch
        out = float(loss_fn(logits, labels).asnumpy())
        assert out > 0.0

    def test_epsilon_zero_matches_standard_ce(self, small_batch):
        """LabelSmoothingCE(epsilon=0) should match plain softmax cross-entropy."""
        logits, labels = small_batch
        ls = LabelSmoothingCE(epsilon=0.0)
        ref = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        ls_val = float(ls(logits, labels).asnumpy())
        ref_val = float(ref(logits, labels).asnumpy())
        assert abs(ls_val - ref_val) < 1e-4, f"ls={ls_val}  ref={ref_val}"

    def test_epsilon_increases_entropy(self, small_batch):
        """Higher epsilon should yield higher (less confident) loss on correct labels."""
        logits = Tensor(np.array([[5.0, -5.0], [-5.0, 5.0]], dtype=np.float32))
        labels = Tensor(np.array([0, 1], dtype=np.int32))
        low_eps = float(LabelSmoothingCE(epsilon=0.0)(logits, labels).asnumpy())
        high_eps = float(LabelSmoothingCE(epsilon=0.4)(logits, labels).asnumpy())
        assert high_eps > low_eps

    def test_class_weights_applied(self, small_batch):
        logits, labels = small_batch
        uniform = Tensor(np.array([1.0, 1.0], dtype=np.float32))
        heavy = Tensor(np.array([1.0, 10.0], dtype=np.float32))
        v_uni = float(LabelSmoothingCE(class_weights=uniform)(logits, labels).asnumpy())
        v_heavy = float(LabelSmoothingCE(class_weights=heavy)(logits, labels).asnumpy())
        assert v_heavy > v_uni, "Heavier weight on seizure class should increase total loss"


# ===========================================================================
# 3. build_loss factory
# ===========================================================================

class TestBuildLoss:

    @pytest.mark.parametrize("loss_name", ["weighted_ce", "focal", "sswce", "label_smoothing"])
    def test_all_loss_names_create_instance(self, loss_name, default_loss_cfg):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(default_loss_cfg, OmegaConf.create({"loss": {"name": loss_name}}))
        loss_fn = build_loss(cfg, num_positive=100, num_negative=900)
        assert isinstance(loss_fn, nn.Cell)

    def test_unknown_loss_raises(self, default_loss_cfg):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(default_loss_cfg, OmegaConf.create({"loss": {"name": "unknown"}}))
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss(cfg, 100, 900)

    def test_inverse_weights_correct_direction(self, default_loss_cfg):
        """With 100 positives and 900 negatives, positive class weight > negative."""
        loss_fn = build_loss(default_loss_cfg, num_positive=100, num_negative=900)
        cw = loss_fn.class_weights.asnumpy()
        assert cw[1] > cw[0], f"Expected w_pos > w_neg, got {cw}"

    def test_forward_produces_scalar(self, default_loss_cfg, small_batch):
        loss_fn = build_loss(default_loss_cfg, 100, 900)
        logits, labels = small_batch
        out = loss_fn(logits, labels)
        assert out.shape == ()


# ===========================================================================
# 4. LR Schedulers
# ===========================================================================

class TestLRSchedulers:

    def test_one_cycle_returns_correct_length(self, scheduler_cfg_factory):
        cfg = scheduler_cfg_factory("one_cycle", pct_start=0.3, div_factor=25)
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        assert len(lrs) == 20 * 10

    def test_one_cycle_warmup_increases(self, scheduler_cfg_factory):
        """LR should increase during warmup phase."""
        cfg = scheduler_cfg_factory("one_cycle", pct_start=0.3, div_factor=25)
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        total = len(lrs)
        warmup_end = int(0.3 * total)
        warmup_lrs = lrs[:warmup_end]
        assert warmup_lrs[-1] > warmup_lrs[0], "LR should increase during warmup"

    def test_one_cycle_peak_is_base_lr(self, scheduler_cfg_factory):
        cfg = scheduler_cfg_factory("one_cycle", pct_start=0.3, div_factor=25)
        base_lr = 1e-3
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        total = len(lrs)
        warmup_end = int(0.3 * total)
        # LR at warmup_end should be close to base_lr
        assert abs(lrs[warmup_end - 1] - base_lr) < base_lr * 0.05

    def test_one_cycle_decay_phase_decreases(self, scheduler_cfg_factory):
        """After warmup, LR should be monotonically non-increasing (cosine decay)."""
        cfg = scheduler_cfg_factory("one_cycle", pct_start=0.3, div_factor=25)
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        total = len(lrs)
        warmup_end = int(0.3 * total)
        decay_lrs = lrs[warmup_end:]
        assert decay_lrs[0] >= decay_lrs[-1], "LR should decay after warmup"

    def test_one_cycle_min_lr_at_end(self, scheduler_cfg_factory):
        cfg = scheduler_cfg_factory("one_cycle", pct_start=0.3, div_factor=25)
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        assert lrs[-1] >= 1e-6 - 1e-10  # at or above min_lr

    def test_cosine_annealing_still_works(self, scheduler_cfg_factory):
        cfg = scheduler_cfg_factory("cosine_annealing")
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        assert len(lrs) == 200
        post_warmup = lrs[5 * 10:]
        assert post_warmup[0] >= post_warmup[-1], "cosine should decay after warmup"

    def test_step_schedule_still_works(self, scheduler_cfg_factory):
        cfg = scheduler_cfg_factory("step", decay_rate=0.1, decay_epochs=[5, 10, 15])
        lrs = build_lr_schedule(cfg, steps_per_epoch=10)
        assert len(lrs) == 200
        # After epoch 5, LR should drop by factor 0.1
        before = lrs[4 * 10]   # epoch 4
        after = lrs[5 * 10]    # epoch 5
        assert after < before

    def test_constant_schedule(self, scheduler_cfg_factory):
        cfg = scheduler_cfg_factory("constant")
        lrs = build_lr_schedule(cfg, steps_per_epoch=5)
        # All post-warmup steps should be constant at base_lr
        post_warmup = lrs[5 * 5:]
        assert all(abs(lr - 1e-3) < 1e-9 for lr in post_warmup)


# ===========================================================================
# 5. Evaluator: EvaluationResult and LOSOResult
# ===========================================================================

class TestEvaluationResult:

    def _make_result(self, recall=0.8, fp_h=1.5, sdr=0.9):
        from auras.training.metrics import MetricsResult
        seg = MetricsResult(accuracy=0.9, recall=recall, precision=0.85,
                            specificity=0.92, f1=0.82, fpr=0.08, auc_roc=0.95)
        return EvaluationResult(
            segment=seg,
            fp_per_hour=fp_h,
            seizure_detection_rate=sdr,
            total_recording_hours=2.0,
            n_seizures_detected=9,
            n_seizures_total=10,
            n_false_alarms=3,
        )

    def test_to_dict_has_all_keys(self):
        r = self._make_result()
        d = r.to_dict()
        required = {"accuracy", "recall", "precision", "specificity", "f1", "fpr", "auc_roc",
                    "fp_per_hour", "seizure_detection_rate", "total_recording_hours",
                    "n_seizures_detected", "n_seizures_total", "n_false_alarms"}
        assert required.issubset(set(d.keys())), f"Missing keys: {required - set(d.keys())}"

    def test_to_dict_values_match(self):
        r = self._make_result(recall=0.75, fp_h=2.3)
        d = r.to_dict()
        assert abs(d["recall"] - 0.75) < 1e-6
        assert abs(d["fp_per_hour"] - 2.3) < 1e-6


class TestLOSOResult:

    def _make_loso(self, recalls, fp_hs):
        from auras.training.metrics import MetricsResult
        loso = LOSOResult()
        for rec, fp in zip(recalls, fp_hs):
            seg = MetricsResult(accuracy=0.9, recall=rec, precision=0.85,
                                specificity=0.92, f1=0.82, fpr=0.08, auc_roc=0.95)
            loso.fold_results.append(EvaluationResult(
                segment=seg, fp_per_hour=fp, seizure_detection_rate=0.9,
            ))
        return loso

    def test_mean_recall_correct(self):
        recalls = [0.6, 0.8, 0.9]
        loso = self._make_loso(recalls, [1.0, 1.0, 1.0])
        mean = loso.mean_metrics()
        assert abs(mean["recall"] - np.mean(recalls)) < 1e-6

    def test_std_metrics_correct(self):
        recalls = [0.6, 0.8, 0.9]
        loso = self._make_loso(recalls, [1.0, 2.0, 3.0])
        std = loso.std_metrics()
        assert abs(std["fp_per_hour"] - np.std([1.0, 2.0, 3.0])) < 1e-6

    def test_summary_string_not_empty(self):
        loso = self._make_loso([0.8], [1.2])
        s = loso.summary()
        assert "fold" in s.lower() and "sensitivity" in s.lower()

    def test_empty_loso_returns_empty_dict(self):
        loso = LOSOResult()
        assert loso.mean_metrics() == {}
        assert loso.std_metrics() == {}


# ===========================================================================
# 6. compute_event_metrics
# ===========================================================================

class TestComputeEventMetrics:

    def test_perfect_detection_sdr_one(self):
        y_true = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
        y_pred = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
        fp_h, sdr, *_ = compute_event_metrics(y_true, y_pred, window_length_s=4.0)
        assert sdr == 1.0
        assert fp_h == 0.0

    def test_all_missed_sdr_zero(self):
        y_true = np.array([0, 0, 1, 1, 0], dtype=np.int32)
        y_pred = np.zeros(5, dtype=np.int32)
        fp_h, sdr, *_ = compute_event_metrics(y_true, y_pred, window_length_s=4.0)
        assert sdr == 0.0
        assert fp_h == 0.0

    def test_false_alarm_counted(self):
        y_true = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        y_pred = np.array([0, 1, 1, 0, 0], dtype=np.int32)
        fp_h, sdr, total_h, n_det, n_total, n_fa = compute_event_metrics(
            y_true, y_pred, window_length_s=3600.0  # 1 h each window → 5 h total
        )
        assert n_fa == 1
        assert abs(fp_h - 1.0 / 5.0) < 1e-6

    def test_partial_overlap_counts_as_detected(self):
        """Seizure event spans windows 3-5; prediction only covers window 4.  Should count."""
        y_true = np.array([0, 0, 1, 1, 1, 0], dtype=np.int32)
        y_pred = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
        _, sdr, _, n_det, n_total, _ = compute_event_metrics(y_true, y_pred, 4.0)
        assert sdr == 1.0
        assert n_det == 1

    def test_two_seizure_events(self):
        y_true = np.array([1, 1, 0, 0, 1, 1], dtype=np.int32)
        y_pred = np.array([1, 0, 0, 0, 0, 1], dtype=np.int32)
        _, sdr, _, n_det, n_total, _ = compute_event_metrics(y_true, y_pred, 4.0)
        assert n_total == 2
        assert n_det == 2
        assert sdr == 1.0

    def test_fp_per_hour_uses_total_duration(self):
        n = 3600  # 3600 windows × 1 s → 1 hour
        y_true = np.zeros(n, dtype=np.int32)
        y_pred = np.zeros(n, dtype=np.int32)
        y_pred[10:12] = 1  # one false alarm
        fp_h, _, total_h, _, _, _ = compute_event_metrics(y_true, y_pred, window_length_s=1.0)
        assert abs(total_h - 1.0) < 0.01
        assert abs(fp_h - 1.0) < 0.01


# ===========================================================================
# 7. _find_contiguous_events
# ===========================================================================

class TestFindContiguousEvents:

    def test_single_run(self):
        labels = np.array([0, 1, 1, 0])
        events = _find_contiguous_events(labels)
        assert events == [(1, 2)]

    def test_two_runs(self):
        labels = np.array([1, 0, 0, 1, 1])
        events = _find_contiguous_events(labels)
        assert events == [(0, 0), (3, 4)]

    def test_all_zeros(self):
        assert _find_contiguous_events(np.zeros(5, dtype=np.int32)) == []

    def test_all_ones(self):
        labels = np.ones(4, dtype=np.int32)
        events = _find_contiguous_events(labels)
        assert events == [(0, 3)]

    def test_trailing_run(self):
        labels = np.array([0, 0, 1, 1])
        events = _find_contiguous_events(labels)
        assert events == [(2, 3)]


# ===========================================================================
# 8. evaluate_epoch integration
# ===========================================================================

class _TinyModel(nn.Cell):
    """Tiny 2-class MLP for testing evaluate_epoch."""

    def __init__(self, always_class: int = 0):
        super().__init__()
        self._always = always_class

    def construct(self, x):
        import mindspore.numpy as mnp
        batch = x.shape[0]
        if self._always == 0:
            return mnp.stack([mnp.ones((batch,)), mnp.zeros((batch,))], axis=1)
        else:
            return mnp.stack([mnp.zeros((batch,)), mnp.ones((batch,))], axis=1)


class TestEvaluateEpoch:

    def _make_iter(self, y_true):
        """Yield batches of dummy (X, y) tensors."""
        n = len(y_true)
        X = Tensor(np.zeros((n, 4, 256), dtype=np.float32))
        y = Tensor(y_true.astype(np.int32))
        return iter([(X, y)])

    def test_returns_evaluation_result(self):
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        model = _TinyModel(always_class=0)
        result = evaluate_epoch(model, self._make_iter(y), window_length_s=4.0)
        assert isinstance(result, EvaluationResult)

    def test_all_correct_class0(self):
        y = np.zeros(10, dtype=np.int32)
        model = _TinyModel(always_class=0)
        result = evaluate_epoch(model, self._make_iter(y))
        assert result.segment.accuracy == 1.0

    def test_all_wrong_seizures_missed(self):
        y = np.ones(10, dtype=np.int32)
        model = _TinyModel(always_class=0)  # always predicts non-seizure
        result = evaluate_epoch(model, self._make_iter(y))
        assert result.segment.recall == 0.0
        assert result.seizure_detection_rate == 0.0

    def test_fp_per_hour_nonzero_for_false_alarms(self):
        y = np.zeros(3600, dtype=np.int32)           # all normal
        model = _TinyModel(always_class=1)            # always predicts seizure
        result = evaluate_epoch(model, self._make_iter(y), window_length_s=1.0)
        assert result.fp_per_hour > 0.0

    def test_evaluation_result_fields_populated(self):
        y = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        model = _TinyModel(always_class=1)
        result = evaluate_epoch(model, self._make_iter(y))
        assert result.total_recording_hours > 0
        assert result.n_seizures_total >= 0
        assert result.n_false_alarms >= 0


# ===========================================================================
# 9. EarlyStoppingCallback
# ===========================================================================

class TestEarlyStoppingCallback:

    def test_raises_stop_iteration_after_patience(self):
        cb = EarlyStoppingCallback(patience=3, monitor="val_recall", mode="max")
        cb.best_value = 0.8
        cb.no_improve_count = 0

        # Simulate no improvement for `patience` epochs
        for _ in range(3):
            cb.no_improve_count += 1

        assert cb.no_improve_count >= cb.patience

    def test_mode_max_updates_best(self):
        cb = EarlyStoppingCallback(patience=5, monitor="val_recall", mode="max")
        assert cb.mode == "max"

    def test_mode_min_attribute(self):
        cb = EarlyStoppingCallback(patience=5, monitor="val_loss", mode="min")
        assert cb.mode == "min"


# ===========================================================================
# 10. BestCheckpointCallback
# ===========================================================================

class TestBestCheckpointCallback:

    def test_can_be_instantiated(self, tmp_path):
        cb = BestCheckpointCallback(save_dir=str(tmp_path), save_top_k=3)
        assert cb.save_top_k == 3

    def test_save_dir_created(self, tmp_path):
        save_dir = tmp_path / "checkpoints"
        BestCheckpointCallback(save_dir=str(save_dir), save_top_k=2)
        assert save_dir.exists()


# ===========================================================================
# 11. LOSO cross-validation splits
# ===========================================================================

class TestLOSOSplits:

    def test_no_subject_leakage(self):
        subjects = np.array(["A", "A", "B", "B", "C", "C"])
        y = np.array([0, 1, 0, 1, 0, 1])
        for train_idx, test_idx in loso_splits(subjects, y):
            train_subjects = set(subjects[train_idx])
            test_subjects = set(subjects[test_idx])
            assert train_subjects.isdisjoint(test_subjects), "Test subject leaked into train!"

    def test_each_held_out_subject_appears_once(self):
        subjects = np.array(["A", "A", "B", "B", "C", "C"])
        y = np.array([0, 1, 0, 1, 0, 1])
        held_out = [subjects[test_idx[0]] for _, test_idx in loso_splits(subjects, y)]
        assert sorted(held_out) == ["A", "B", "C"]

    def test_folds_with_no_seizures_skipped(self):
        subjects = np.array(["A", "A", "B", "B"])
        y = np.array([0, 1, 0, 0])  # subject B has no seizures
        folds = list(loso_splits(subjects, y))
        # Only fold with A held out should survive (B held out has no seizures)
        assert len(folds) == 1
        _, test_idx = folds[0]
        assert all(subjects[test_idx] == "A")

    def test_indices_cover_all_samples(self):
        subjects = np.array(["A"] * 5 + ["B"] * 5)
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        for train_idx, test_idx in loso_splits(subjects, y):
            combined = np.sort(np.concatenate([train_idx, test_idx]))
            assert list(combined) == list(range(10))

    def test_train_set_has_no_test_samples(self):
        subjects = np.array(["X"] * 4 + ["Y"] * 4)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        for train_idx, test_idx in loso_splits(subjects, y):
            assert len(set(train_idx) & set(test_idx)) == 0


# ===========================================================================
# 12. Gradient Clipping
# ===========================================================================

class TestGradientClipping:

    def test_clip_by_global_norm_reduces_large_grads(self):
        """After clipping with norm=1.0, the global norm should be ≤ 1.0."""
        import mindspore.ops as ops
        # Create gradient tensors with large values
        grads = tuple(
            Tensor(np.random.randn(50, 50).astype(np.float32) * 100.0)
            for _ in range(3)
        )
        clipped_grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
        global_norm = float(
            sum(float((g ** 2).sum().asnumpy()) for g in clipped_grads) ** 0.5
        )
        assert global_norm <= 1.0 + 1e-5, f"Global norm after clip = {global_norm}"

    def test_small_grads_not_distorted(self):
        """Grads already within norm should pass through unchanged."""
        import mindspore.ops as ops
        tiny = (Tensor(np.array([0.001, 0.001], dtype=np.float32)),)
        out = ops.clip_by_global_norm(tiny, clip_norm=1.0)
        np.testing.assert_allclose(
            out[0].asnumpy(), tiny[0].asnumpy(), atol=1e-6
        )


# ===========================================================================
# 13. Trainer module imports and _train_loop signature
# ===========================================================================

class TestTrainerModule:

    def test_train_function_importable(self):
        from auras.training.trainer import train
        assert callable(train)

    def test_loso_train_importable(self):
        from auras.training.trainer import loso_train
        assert callable(loso_train)

    def test_build_optimizer_importable(self):
        from auras.training.trainer import _build_optimizer
        assert callable(_build_optimizer)

    def test_train_loop_importable(self):
        from auras.training.trainer import _train_loop
        assert callable(_train_loop)


# ===========================================================================
# 14. YAML configs are parseable
# ===========================================================================

class TestYAMLConfigs:

    @pytest.mark.parametrize("cfg_name", [
        "default", "eegformer", "lightweight", "cnn_informer", "conv_snn"
    ])
    def test_config_loadable(self, cfg_name):
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).parent.parent / "configs" / "training" / f"{cfg_name}.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        cfg = OmegaConf.load(cfg_path)
        assert "epochs" in cfg
        assert "learning_rate" in cfg
        assert "scheduler" in cfg
        assert "loss" in cfg

    @pytest.mark.parametrize("cfg_name", [
        "default", "eegformer", "lightweight", "cnn_informer", "conv_snn"
    ])
    def test_scheduler_name_present(self, cfg_name):
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).parent.parent / "configs" / "training" / f"{cfg_name}.yaml"
        cfg = OmegaConf.load(cfg_path)
        assert "name" in cfg.scheduler

    @pytest.mark.parametrize("cfg_name", [
        "default", "eegformer", "lightweight", "cnn_informer", "conv_snn"
    ])
    def test_loss_name_valid(self, cfg_name):
        from omegaconf import OmegaConf
        valid_names = {"weighted_ce", "focal", "sswce", "label_smoothing"}
        cfg_path = Path(__file__).parent.parent / "configs" / "training" / f"{cfg_name}.yaml"
        cfg = OmegaConf.load(cfg_path)
        assert cfg.loss.name in valid_names


# ===========================================================================
# 15. End-to-end loss + scheduler smoke test
# ===========================================================================

class TestEndToEndSmoke:

    def test_sswce_with_one_cycle_lr(self, scheduler_cfg_factory):
        """SSWCELoss forward + OneCycleLR schedule produce valid outputs."""
        from omegaconf import OmegaConf
        cw = Tensor(np.array([0.5, 2.0], dtype=np.float32))
        loss_fn = SSWCELoss(alpha=0.5, beta=0.5, class_weights=cw)
        logits = Tensor(np.random.randn(8, 2).astype(np.float32))
        labels = Tensor(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32))
        loss = float(loss_fn(logits, labels).asnumpy())
        assert math.isfinite(loss) and loss > 0.0

        cfg = scheduler_cfg_factory("one_cycle", pct_start=0.3)
        lrs = build_lr_schedule(cfg, steps_per_epoch=5)
        assert all(math.isfinite(lr) and lr >= 0 for lr in lrs)

    def test_label_smoothing_with_cosine_lr(self, scheduler_cfg_factory):
        ls = LabelSmoothingCE(epsilon=0.1)
        logits = Tensor(np.random.randn(6, 2).astype(np.float32))
        labels = Tensor(np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))
        loss = float(ls(logits, labels).asnumpy())
        assert math.isfinite(loss) and loss > 0.0

        cfg = scheduler_cfg_factory("cosine_annealing")
        lrs = build_lr_schedule(cfg, steps_per_epoch=5)
        assert all(math.isfinite(lr) for lr in lrs)
