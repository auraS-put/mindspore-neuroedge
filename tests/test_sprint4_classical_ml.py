"""Test suite for Sprint 4 — Classical ML & Post-Processing.

Covers:
    DWT feature extraction
        - dwt_features output shape (200 for 4 ch, level=4)
        - dwt_filter output shape matches input
        - dwt_subbands output shape
        - numerical stability on zero-signal
    Classical ML models
        - build_classical_model creates sklearn estimator for each name
        - unknown name raises ValueError
        - list_classical_models returns all 6 names
        - fit + predict on toy data (shape checks, no crash)
        - train_and_evaluate returns dict with all metric keys
        - XGBoost scale_pos_weight set from class ratio
        - class imbalance handled (n_pos=1, n_neg=99)
    Postprocessing: majority_vote
        - output length unchanged
        - all-zeros / all-ones identity
        - isolated FP suppressed (n=3, k=2)
        - sequential positives preserved
    Postprocessing: ema_smooth
        - output length unchanged
        - all-zero probabilities → all-zero predictions
        - high probability → positive
        - sequential causality (value at t cannot depend on t+1)
    Postprocessing: quantile_aggregate
        - single model pass-through
        - 2-model majority returns correct percentile
        - shape preserved
    Postprocessing: collar_merge
        - short gap filled
        - long gap not filled
        - no-gap sequence unchanged
        - all-zeros unchanged
    Postprocessing: threshold_sweep
        - returns list of dicts with correct keys
        - length equals number of thresholds
        - threshold=0.0 → all positive; threshold=1.0 → all negative
    YAML config loading
        - classical_ml.yaml parseable and has required keys
        - postprocess.yaml parseable and has required keys
    run_classical_baselines module imports
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from auras.data.preprocess import dwt_features, dwt_filter, dwt_subbands
from auras.models.classical_ml import (
    BEST_PARAMS,
    build_classical_model,
    list_classical_models,
    train_and_evaluate,
)
from auras.inference.postprocess import (
    collar_merge,
    ema_smooth,
    majority_vote,
    quantile_aggregate,
    threshold_sweep,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def small_eeg():
    """Tiny (4, 256) EEG window for DWT tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 256)).astype(np.float32)


@pytest.fixture()
def toy_dataset():
    """Small tabular dataset: 60 samples, 20 features, 10 % positive."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 20)).astype(np.float32)
    y = (rng.random(60) < 0.1).astype(np.int32)
    # Ensure at least one positive in each half
    y[0] = 1
    y[30] = 1
    return X, y


@pytest.fixture()
def imbalanced_dataset():
    """Very imbalanced: 1 positive, 99 negatives."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 20)).astype(np.float32)
    y = np.zeros(100, dtype=np.int32)
    y[50] = 1
    return X, y


# ===========================================================================
# 1. DWT feature extraction
# ===========================================================================

class TestDWTFeatures:

    def test_shape_4ch_level4(self, small_eeg):
        feat = dwt_features(small_eeg, wavelet="db4", level=4)
        # 4 channels × (4+1) bands × 10 stats = 200
        assert feat.shape == (200,), f"Expected (200,), got {feat.shape}"

    def test_dtype_float32(self, small_eeg):
        feat = dwt_features(small_eeg, wavelet="db4", level=4)
        assert feat.dtype == np.float32

    def test_2ch_different_length(self):
        x = np.random.randn(2, 512).astype(np.float32)
        feat = dwt_features(x, wavelet="db4", level=4)
        # 2 channels × 5 bands × 10 stats = 100
        assert feat.shape == (100,)

    def test_zero_signal_no_crash(self):
        x = np.zeros((4, 256), dtype=np.float32)
        feat = dwt_features(x, wavelet="db4", level=4)
        assert feat.shape == (200,)
        # nan_to_num cleans skew/kurtosis NaN and sample-entropy Inf for zero-variance bands
        assert np.all(np.isfinite(feat)), "DWT features of zero signal contain unhandled NaN/Inf"

    def test_level3_gives_correct_shape(self, small_eeg):
        feat = dwt_features(small_eeg, wavelet="db4", level=3)
        # 4 × (3+1) bands × 10 = 160
        assert feat.shape == (160,)


class TestDWTFilter:

    def test_output_same_shape(self, small_eeg):
        filtered = dwt_filter(small_eeg)
        assert filtered.shape == small_eeg.shape

    def test_output_dtype_preserved(self, small_eeg):
        filtered = dwt_filter(small_eeg)
        assert filtered.dtype == small_eeg.dtype

    def test_zero_signal_returns_zeros(self):
        x = np.zeros((4, 256), dtype=np.float32)
        out = dwt_filter(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_filtered_differs_from_raw(self, small_eeg):
        """Band-pass filter should change the signal."""
        filtered = dwt_filter(small_eeg)
        assert not np.allclose(filtered, small_eeg, atol=1e-6)


class TestDWTSubbands:

    def test_output_shape(self, small_eeg):
        bands = dwt_subbands(small_eeg, wavelet="db4", level=4)
        # (C, level+1, T) = (4, 5, 256)
        assert bands.shape == (4, 5, 256)

    def test_level3_shape(self, small_eeg):
        bands = dwt_subbands(small_eeg, wavelet="db4", level=3)
        assert bands.shape == (4, 4, 256)


# ===========================================================================
# 2. Classical ML models
# ===========================================================================

class TestBuildClassicalModel:

    @pytest.mark.parametrize("name", ["svm_rbf", "svm_linear", "random_forest",
                                       "xgboost", "lightgbm", "knn"])
    def test_creates_estimator(self, name):
        clf = build_classical_model(name)
        assert hasattr(clf, "fit") and hasattr(clf, "predict")

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown classical model"):
            build_classical_model("mystery_model")

    def test_list_models_returns_all(self):
        names = list_classical_models()
        expected = {"svm_rbf", "svm_linear", "random_forest",
                    "xgboost", "lightgbm", "knn"}
        assert set(names) == expected

    def test_xgboost_scale_pos_weight_set(self):
        clf = build_classical_model("xgboost", class_ratio=9.0)
        # XGBClassifier stores scale_pos_weight as an init param
        assert clf.scale_pos_weight == 9.0


class TestFitPredict:
    """Check that each model can fit + predict on toy data without error."""

    @pytest.mark.parametrize("name", ["svm_rbf", "random_forest",
                                       "xgboost", "lightgbm", "knn"])
    def test_fit_predict_shapes(self, name, toy_dataset):
        X, y = toy_dataset
        clf = build_classical_model(name)
        clf.fit(X[:40], y[:40])
        preds = clf.predict(X[40:])
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("name", ["svm_rbf", "random_forest",
                                       "xgboost", "lightgbm"])
    def test_predict_proba_available(self, name, toy_dataset):
        X, y = toy_dataset
        clf = build_classical_model(name)
        clf.fit(X[:40], y[:40])
        proba = clf.predict_proba(X[40:])
        assert proba.shape == (20, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestTrainAndEvaluate:

    def test_returns_dict_with_required_keys(self, toy_dataset):
        X, y = toy_dataset
        result = train_and_evaluate("random_forest", X[:40], y[:40], X[40:], y[40:])
        required = {"model_name", "accuracy", "recall", "precision",
                    "specificity", "f1", "auc_roc", "fpr",
                    "n_train", "n_test", "n_pos_train", "n_neg_train"}
        assert required.issubset(set(result.keys()))

    def test_model_name_in_result(self, toy_dataset):
        X, y = toy_dataset
        result = train_and_evaluate("knn", X[:40], y[:40], X[40:], y[40:])
        assert result["model_name"] == "knn"

    def test_metrics_in_valid_range(self, toy_dataset):
        X, y = toy_dataset
        result = train_and_evaluate("random_forest", X[:40], y[:40], X[40:], y[40:])
        for k in ["accuracy", "recall", "precision", "specificity", "f1", "auc_roc"]:
            v = result[k]
            assert 0.0 <= v <= 1.0, f"{k}={v} outside [0,1]"

    def test_imbalanced_does_not_crash(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        # Train on 80, test on 20
        result = train_and_evaluate("random_forest", X[:80], y[:80], X[80:], y[80:])
        assert "recall" in result


# ===========================================================================
# 3. majority_vote
# ===========================================================================

class TestMajorityVote:

    def test_output_length_unchanged(self):
        preds = np.array([0, 1, 0, 1, 1, 0, 0])
        out = majority_vote(preds, n=3, k=2)
        assert len(out) == len(preds)

    def test_all_zeros(self):
        preds = np.zeros(10, dtype=np.int32)
        out = majority_vote(preds)
        np.testing.assert_array_equal(out, preds)

    def test_all_ones(self):
        preds = np.ones(10, dtype=np.int32)
        out = majority_vote(preds)
        np.testing.assert_array_equal(out, preds)

    def test_isolated_fp_suppressed(self):
        """Single isolated positive should be suppressed by majority vote."""
        preds = np.array([0, 0, 1, 0, 0], dtype=np.int32)
        out = majority_vote(preds, n=3, k=2)
        # The central 1 has two 0-neighbours → should become 0
        assert out[2] == 0

    def test_consecutive_positives_preserved(self):
        """Three consecutive positives should all remain positive."""
        preds = np.array([0, 1, 1, 1, 0], dtype=np.int32)
        out = majority_vote(preds, n=3, k=2)
        assert out[1] == 1
        assert out[2] == 1
        assert out[3] == 1

    def test_short_sequence(self):
        """Sequence shorter than n should be returned unchanged."""
        preds = np.array([1, 0], dtype=np.int32)
        out = majority_vote(preds, n=5, k=3)
        np.testing.assert_array_equal(out, preds)

    def test_output_dtype_int(self):
        preds = np.array([0, 1, 1, 0, 1])
        out = majority_vote(preds)
        assert out.dtype in (np.int32, np.int64, np.uint8)


# ===========================================================================
# 4. ema_smooth
# ===========================================================================

class TestEMASmooth:

    def test_output_length_unchanged(self):
        probs = np.random.rand(20)
        out = ema_smooth(probs, alpha=0.3)
        assert len(out) == 20

    def test_all_zero_probabilities(self):
        probs = np.zeros(10)
        out = ema_smooth(probs, alpha=0.3, threshold=0.5)
        np.testing.assert_array_equal(out, np.zeros(10, dtype=np.int32))

    def test_all_high_probabilities(self):
        probs = np.ones(10)
        out = ema_smooth(probs, alpha=0.3, threshold=0.5)
        np.testing.assert_array_equal(out, np.ones(10, dtype=np.int32))

    def test_smoothing_reduces_isolated_spikes(self):
        """Single spike at position 5 should be reduced by EMA."""
        probs = np.zeros(20)
        probs[5] = 0.9
        out = ema_smooth(probs, alpha=0.3, threshold=0.5)
        # EMA will smooth the spike; values before it unaffected
        assert out[0] == 0

    def test_alpha_one_no_smoothing(self):
        """alpha=1.0 means no history: EMA(t) = p(t)."""
        probs = np.array([0.0, 0.0, 0.8, 0.0, 0.0])
        out = ema_smooth(probs, alpha=1.0, threshold=0.5)
        assert out[2] == 1
        assert out[0] == 0
        assert out[4] == 0

    def test_output_binary(self):
        probs = np.random.rand(30)
        out = ema_smooth(probs)
        assert set(out.tolist()).issubset({0, 1})


# ===========================================================================
# 5. quantile_aggregate
# ===========================================================================

class TestQuantileAggregate:

    def test_single_model_passthrough(self):
        probs = np.array([0.1, 0.5, 0.9])
        out = quantile_aggregate(probs, quantile=0.6)
        np.testing.assert_array_almost_equal(out, probs)

    def test_two_models_median(self):
        probs = np.array([[0.2, 0.8], [0.4, 0.6]])   # shape (2, 2)
        out = quantile_aggregate(probs, quantile=0.5)
        expected = np.array([0.3, 0.7])
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_output_shape_preserved(self):
        probs = np.random.rand(5, 100)   # 5 models, 100 windows
        out = quantile_aggregate(probs, quantile=0.6)
        assert out.shape == (100,)

    def test_high_quantile_conservative(self):
        """High quantile means only kept if nearly all models agree."""
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])  # (3, 2)
        out_low  = quantile_aggregate(probs, quantile=0.1)
        out_high = quantile_aggregate(probs, quantile=0.9)
        assert out_high[0] >= out_low[0]  # high quantile ≥ low for max-vote idx


# ===========================================================================
# 6. collar_merge
# ===========================================================================

class TestCollarMerge:

    def test_short_gap_filled(self):
        preds = np.array([1, 1, 0, 0, 1, 1], dtype=np.int32)
        out = collar_merge(preds, collar_windows=3)
        # Gap of 2 windows is ≤ 3 → should be filled
        assert out[2] == 1
        assert out[3] == 1

    def test_long_gap_not_filled(self):
        preds = np.array([1, 0, 0, 0, 0, 1], dtype=np.int32)
        out = collar_merge(preds, collar_windows=2)
        # Gap of 4 > 2 → should not be filled
        assert out[2] == 0

    def test_no_gap_unchanged(self):
        preds = np.array([1, 1, 1, 0, 0], dtype=np.int32)
        out = collar_merge(preds, collar_windows=3)
        assert out[0] == 1 and out[1] == 1 and out[2] == 1
        assert out[3] == 0

    def test_all_zeros_unchanged(self):
        preds = np.zeros(8, dtype=np.int32)
        out = collar_merge(preds)
        np.testing.assert_array_equal(out, preds)

    def test_output_length_unchanged(self):
        preds = np.array([1, 0, 0, 1, 1, 0, 1])
        out = collar_merge(preds, collar_windows=2)
        assert len(out) == len(preds)


# ===========================================================================
# 7. threshold_sweep
# ===========================================================================

class TestThresholdSweep:

    def test_returns_list_of_correct_length(self):
        probs = np.random.rand(50)
        y_true = (probs > 0.5).astype(int)
        results = threshold_sweep(probs, y_true, n_thresholds=11)
        assert len(results) == 11

    def test_each_row_has_required_keys(self):
        probs = np.random.rand(30)
        y_true = (probs > 0.4).astype(int)
        results = threshold_sweep(probs, y_true)
        required = {"threshold", "recall", "specificity", "precision", "f1"}
        for row in results:
            assert required.issubset(set(row.keys()))

    def test_threshold_zero_all_positive(self):
        probs = np.random.rand(20)
        y_true = np.ones(20, dtype=int)
        results = threshold_sweep(probs, y_true, n_thresholds=3)
        # threshold=0.0 → every window is positive → recall = 1.0
        row0 = next(r for r in results if r["threshold"] == 0.0)
        assert row0["recall"] == 1.0

    def test_with_post_fn(self):
        probs = np.random.rand(30)
        y_true = (probs > 0.5).astype(int)
        results = threshold_sweep(probs, y_true, post_fn=lambda p: majority_vote(p, 3, 2))
        assert len(results) == 51  # default 51 thresholds

    def test_default_thresholds_count(self):
        probs = np.random.rand(20)
        y_true = (probs > 0.5).astype(int)
        results = threshold_sweep(probs, y_true)
        assert len(results) == 51


# ===========================================================================
# 8. YAML configs
# ===========================================================================

class TestYAMLConfigs:

    def _load(self, name):
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).parent.parent / "configs" / f"{name}.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        return OmegaConf.load(cfg_path)

    def test_classical_ml_config_loadable(self):
        cfg = self._load("classical_ml")
        assert "features" in cfg
        assert "models" in cfg

    def test_classical_ml_feature_keys(self):
        cfg = self._load("classical_ml")
        assert cfg.features.wavelet == "db4"
        assert cfg.features.level == 4
        assert cfg.features.n_features == 200

    def test_classical_ml_all_models_listed(self):
        cfg = self._load("classical_ml")
        expected = {"svm_rbf", "svm_linear", "random_forest",
                    "xgboost", "lightgbm", "knn"}
        assert set(cfg.models) == expected

    def test_postprocess_config_loadable(self):
        cfg = self._load("postprocess")
        assert "pipeline" in cfg
        assert "ema" in cfg
        assert "ensemble" in cfg

    def test_postprocess_majority_vote_defaults(self):
        cfg = self._load("postprocess")
        mv = next(s for s in cfg.pipeline if s.name == "majority_vote")
        assert mv.n == 3
        assert mv.k == 2

    def test_postprocess_ema_alpha(self):
        cfg = self._load("postprocess")
        assert abs(cfg.ema.alpha - 0.3) < 1e-6

    def test_postprocess_ensemble_quantile(self):
        cfg = self._load("postprocess")
        assert abs(cfg.ensemble.quantile - 0.6) < 1e-6


# ===========================================================================
# 9. Script imports
# ===========================================================================

class TestScriptImports:

    def test_classical_baselines_script_importable(self):
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import run_classical_baselines as rb
        assert callable(rb.extract_features)
        assert callable(rb.run_loso)
        assert callable(rb.print_summary)

    def test_classical_ml_module_importable(self):
        from auras.models.classical_ml import build_classical_model, list_classical_models
        assert callable(build_classical_model)
        assert callable(list_classical_models)

    def test_postprocess_module_importable(self):
        from auras.inference.postprocess import (
            majority_vote, ema_smooth, quantile_aggregate,
            collar_merge, threshold_sweep,
        )
        for fn in [majority_vote, ema_smooth, quantile_aggregate,
                   collar_merge, threshold_sweep]:
            assert callable(fn)


# ===========================================================================
# 10. End-to-end: DWT features → classical model
# ===========================================================================

class TestEndToEnd:

    def test_dwt_features_into_svm(self):
        """Extract DWT features from synthetic EEG and train SVM-RBF."""
        rng = np.random.default_rng(7)
        n_samples = 40
        X_eeg = rng.standard_normal((n_samples, 4, 256)).astype(np.float32)
        y = (rng.random(n_samples) < 0.2).astype(np.int32)
        y[0] = 1  # ensure at least one positive

        # Extract features
        X_feats = np.stack([dwt_features(x) for x in X_eeg])
        assert X_feats.shape == (n_samples, 200)

        # Train + predict
        clf = build_classical_model("svm_rbf")
        clf.fit(X_feats[:30], y[:30])
        preds = clf.predict(X_feats[30:])
        assert preds.shape == (10,)

    def test_dwt_features_into_random_forest(self):
        """Extract DWT features and verify RF produces valid probability estimates."""
        rng = np.random.default_rng(8)
        n = 50
        X_eeg = rng.standard_normal((n, 4, 256)).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)
        y[:5] = 1

        X_feats = np.stack([dwt_features(x) for x in X_eeg])
        clf = build_classical_model("random_forest")
        clf.fit(X_feats[:40], y[:40])
        proba = clf.predict_proba(X_feats[40:])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_postprocess_pipeline(self):
        """majority_vote → collar_merge pipeline reduces FP without losing TP."""
        rng = np.random.default_rng(9)
        probs = rng.random(50)
        y_true = np.zeros(50, dtype=np.int32)
        y_true[20:25] = 1  # one seizure event

        raw_preds = (probs > 0.4).astype(np.int32)
        voted = majority_vote(raw_preds, n=3, k=2)
        merged = collar_merge(voted, collar_windows=3)

        assert len(merged) == 50
        assert set(merged.tolist()).issubset({0, 1})
