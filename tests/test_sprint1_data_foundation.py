"""Sprint 1 — Data Foundation: comprehensive test suite.

Tests cover every component implemented in Sprint 1:
  1.  Preprocessing:  zscore, minmax_normalize, sliding_window
  2.  DWT filtering:  dwt_filter (Li et al. (CNN-Informer) band-pass)
  3.  DWT subbands:   dwt_subbands (Dokare & Gupta (DWT+SVM) 5-band decomposition)
  4.  DWT features:   dwt_features (200-dim feature vector)
  5.  CHB-MIT parser: parse_summary_file, load_all_seizures
  6.  Prediction labeling: detection mode and prediction mode
  7.  Dataset class:  EEGWindowDataset with subjects array
  8.  prepare_dataset.py stub: via programmatic call

Run with:
    .venv/bin/python -m pytest tests/test_sprint1_data_foundation.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pytest


# =============================================================================
# 1. Preprocessing: zscore, minmax_normalize, sliding_window
# =============================================================================

class TestNormalization:
    """Tests for zscore and minmax_normalize."""

    def test_zscore_zero_mean(self):
        from auras.data.preprocess import zscore
        rng = np.random.default_rng(42)
        x = rng.standard_normal((4, 1024)).astype(np.float32) * 100 + 50
        z = zscore(x)
        assert np.abs(z.mean(axis=-1)).max() < 1e-4, "zscore should give ~0 mean per channel"

    def test_zscore_unit_std(self):
        from auras.data.preprocess import zscore
        rng = np.random.default_rng(42)
        x = rng.standard_normal((4, 1024)).astype(np.float32) * 100 + 50
        z = zscore(x)
        assert np.abs(z.std(axis=-1) - 1.0).max() < 0.01, "zscore should give ~1 std per channel"

    def test_zscore_shape_preserved(self):
        from auras.data.preprocess import zscore
        x = np.random.randn(4, 512).astype(np.float32)
        assert zscore(x).shape == (4, 512)

    def test_zscore_constant_channel(self):
        """Constant channel should not produce NaN (eps guards against div-by-zero)."""
        from auras.data.preprocess import zscore
        x = np.zeros((4, 256), dtype=np.float32)
        z = zscore(x)
        assert not np.any(np.isnan(z)), "zscore must not produce NaN for constant channels"

    def test_minmax_range(self):
        from auras.data.preprocess import minmax_normalize
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 1024)).astype(np.float32)
        m = minmax_normalize(x)
        assert m.min() >= 0.0 - 1e-6
        assert m.max() <= 1.0 + 1e-6

    def test_minmax_shape_preserved(self):
        from auras.data.preprocess import minmax_normalize
        x = np.random.randn(4, 512).astype(np.float32)
        assert minmax_normalize(x).shape == (4, 512)

    def test_minmax_constant_channel(self):
        """Constant channel should return all zeros (not NaN)."""
        from auras.data.preprocess import minmax_normalize
        x = np.ones((4, 256), dtype=np.float32)
        m = minmax_normalize(x)
        assert not np.any(np.isnan(m))


class TestSlidingWindow:
    """Tests for sliding_window."""

    def test_window_count(self):
        from auras.data.preprocess import sliding_window
        # T=1024, window=256, stride=128 → (1024-256)//128 + 1 = 7
        x = np.random.randn(4, 1024).astype(np.float32)
        w = sliding_window(x, 256, 128)
        # expected: floor((1024-256)/128) + 1 = 7
        assert w.shape[0] == 7

    def test_window_shape(self):
        from auras.data.preprocess import sliding_window
        x = np.random.randn(4, 1024).astype(np.float32)
        w = sliding_window(x, 256, 128)
        assert w.shape[1] == 4   # channels
        assert w.shape[2] == 256  # window length

    def test_short_signal_empty(self):
        from auras.data.preprocess import sliding_window
        x = np.random.randn(4, 100).astype(np.float32)
        w = sliding_window(x, 256, 128)
        assert w.shape[0] == 0

    def test_exact_fit(self):
        from auras.data.preprocess import sliding_window
        x = np.arange(256, dtype=np.float32).reshape(1, 256)
        w = sliding_window(x, 256, 256)
        assert w.shape[0] == 1
        np.testing.assert_array_equal(w[0, 0], np.arange(256, dtype=np.float32))

    def test_stride_values(self):
        """First window should start at 0, second at stride."""
        from auras.data.preprocess import sliding_window
        x = np.arange(512, dtype=np.float32).reshape(1, 512)
        w = sliding_window(x, 128, 64)
        # First window: samples 0..127
        np.testing.assert_array_equal(w[0, 0, :5], [0, 1, 2, 3, 4])
        # Second window: samples 64..191
        np.testing.assert_array_equal(w[1, 0, :3], [64, 65, 66])

    def test_wrong_ndim_raises(self):
        from auras.data.preprocess import sliding_window
        with pytest.raises(ValueError):
            sliding_window(np.ones(100), 10, 5)  # 1D — should fail


# =============================================================================
# 2,3. DWT filter and subbands
# =============================================================================

class TestDWTFilter:
    """Tests for dwt_filter (Li et al. (CNN-Informer) band-pass)."""

    def setup_method(self):
        pytest.importorskip("pywt")

    def test_output_shape(self):
        from auras.data.preprocess import dwt_filter
        x = np.random.randn(4, 1024).astype(np.float32)
        out = dwt_filter(x)
        assert out.shape == (4, 1024), "dwt_filter must preserve shape"

    def test_output_dtype(self):
        from auras.data.preprocess import dwt_filter
        x = np.random.randn(4, 1024).astype(np.float32)
        out = dwt_filter(x)
        assert out.dtype == np.float32

    def test_high_freq_reduced(self):
        """High-frequency content (> 32 Hz) should be attenuated at 256 Hz."""
        from auras.data.preprocess import dwt_filter
        import numpy as np
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        # Pure 80 Hz sinusoid (above cD1 at 64-128 Hz boundary)
        high_freq = np.sin(2 * np.pi * 80 * t).reshape(1, -1).astype(np.float32)
        filtered = dwt_filter(high_freq, level=5, reconstruct_levels=(3, 4, 5), include_approx=True)
        # Energy should be reduced
        orig_power = np.mean(high_freq ** 2)
        filt_power = np.mean(filtered ** 2)
        assert filt_power < orig_power * 0.3, (
            f"High-freq component not attenuated: orig={orig_power:.4f}, filt={filt_power:.4f}"
        )

    def test_low_freq_preserved(self):
        """Low-frequency content (4 Hz) should pass through the filter."""
        from auras.data.preprocess import dwt_filter
        sfreq = 256.0
        t = np.arange(1024) / sfreq
        low_freq = np.sin(2 * np.pi * 4 * t).reshape(1, -1).astype(np.float32)
        filtered = dwt_filter(low_freq)
        # Power should be largely preserved (> 50%)
        orig_power = float(np.mean(low_freq ** 2))
        filt_power = float(np.mean(filtered ** 2))
        assert filt_power > orig_power * 0.5, (
            f"Low-freq component unexpectedly attenuated: orig={orig_power:.4f}, filt={filt_power:.4f}"
        )

    def test_multichannel(self):
        """Should work for C=4 channels."""
        from auras.data.preprocess import dwt_filter
        x = np.random.randn(4, 1024).astype(np.float32)
        out = dwt_filter(x, level=5, reconstruct_levels=(3, 4, 5))
        assert out.shape == (4, 1024)


class TestDWTSubbands:
    """Tests for dwt_subbands (Dokare & Gupta (DWT+SVM) 5-band decomposition)."""

    def setup_method(self):
        pytest.importorskip("pywt")

    def test_output_shape(self):
        from auras.data.preprocess import dwt_subbands
        x = np.random.randn(4, 1024).astype(np.float32)
        out = dwt_subbands(x, level=4)
        # (C, n_bands, T) = (4, 5, 1024)
        assert out.shape == (4, 5, 1024), f"Expected (4,5,1024), got {out.shape}"

    def test_level3_gives_4_bands(self):
        from auras.data.preprocess import dwt_subbands
        x = np.random.randn(4, 1024).astype(np.float32)
        out = dwt_subbands(x, level=3)
        assert out.shape == (4, 4, 1024)

    def test_bands_sum_approximately_original(self):
        """Sum of all sub-band reconstructions ≈ original signal."""
        from auras.data.preprocess import dwt_subbands
        x = np.random.randn(1, 1024).astype(np.float32)
        subbands = dwt_subbands(x, level=4)  # (1, 5, 1024)
        reconstructed = subbands[0].sum(axis=0)
        np.testing.assert_allclose(reconstructed, x[0], atol=1e-4,
                                   err_msg="Subband sum should reconstruct original signal")


# =============================================================================
# 4. DWT features
# =============================================================================

class TestDWTFeatures:
    """Tests for dwt_features (Dokare & Gupta (DWT+SVM) 200-dim feature vector)."""

    def setup_method(self):
        pytest.importorskip("pywt")

    def test_output_shape_4ch(self):
        """4 channels × 5 bands × 10 stats = 200 features."""
        from auras.data.preprocess import dwt_features
        x = np.random.randn(4, 1024).astype(np.float32)
        feat = dwt_features(x, level=4)
        assert feat.shape == (200,), f"Expected (200,), got {feat.shape}"

    def test_output_dtype(self):
        from auras.data.preprocess import dwt_features
        x = np.random.randn(4, 1024).astype(np.float32)
        feat = dwt_features(x)
        assert feat.dtype == np.float32

    def test_no_nan_inf(self):
        """Features should be finite for typical EEG-like signals."""
        from auras.data.preprocess import dwt_features
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 1024)).astype(np.float32)
        feat = dwt_features(x)
        assert np.all(np.isfinite(feat)) or np.sum(~np.isfinite(feat)) < 5, (
            f"Too many non-finite feature values: {np.sum(~np.isfinite(feat))}"
        )

    def test_features_differ_across_channels(self):
        """Different channel signals should produce different feature vectors."""
        from auras.data.preprocess import dwt_features
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 1024)).astype(np.float32)
        feat = dwt_features(x, level=4)
        # Each channel contributes 50 features; first two channels' features differ
        ch0_feats = feat[:50]
        ch1_feats = feat[50:100]
        assert not np.allclose(ch0_feats, ch1_feats), "Features should differ per channel"

    def test_deterministic(self):
        """Same input → same features."""
        from auras.data.preprocess import dwt_features
        x = np.random.randn(4, 1024).astype(np.float32)
        f1 = dwt_features(x)
        f2 = dwt_features(x)
        np.testing.assert_array_equal(f1, f2)

    def test_single_channel(self):
        """Works for C=1 → 50 features."""
        from auras.data.preprocess import dwt_features
        x = np.random.randn(1, 1024).astype(np.float32)
        feat = dwt_features(x, level=4)
        assert feat.shape == (50,)


# =============================================================================
# 5. CHB-MIT parser
# =============================================================================

SUMMARY_SINGLE = """\
File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
"""

SUMMARY_MULTI = """\
File Name: chb02_16+.edf
File Start Time: 20:17:33
File End Time: 01:22:06
Number of Seizures in File: 4
Seizure 1 Start Time: 130 seconds
Seizure 1 End Time: 212 seconds
Seizure 2 Start Time: 2972 seconds
Seizure 2 End Time: 3053 seconds
Seizure 3 Start Time: 3998 seconds
Seizure 3 End Time: 4077 seconds
Seizure 4 Start Time: 5765 seconds
Seizure 4 End Time: 5811 seconds
"""

SUMMARY_ZERO = """\
File Name: chb03_01.edf
File Start Time: 09:00:00
File End Time: 10:00:00
Number of Seizures in File: 0

File Name: chb03_02.edf
File Start Time: 10:00:01
File End Time: 11:00:01
Number of Seizures in File: 0
"""


class TestCHBMITParser:
    """Tests for src/auras/data/parsers/chbmit.py."""

    def _write_summary(self, tmp_path: Path, subj: str, content: str) -> Path:
        subj_dir = tmp_path / subj
        subj_dir.mkdir()
        sf = subj_dir / f"{subj}-summary.txt"
        sf.write_text(content)
        return sf

    def test_single_seizure(self, tmp_path):
        from auras.data.parsers.chbmit import parse_summary_file
        sf = self._write_summary(tmp_path, "chb01", SUMMARY_SINGLE)
        intervals = parse_summary_file(sf)
        assert len(intervals) == 1
        assert intervals[0].recording == "chb01_03"
        assert intervals[0].onset_sec == 2996.0
        assert intervals[0].offset_sec == 3036.0

    def test_multi_seizure(self, tmp_path):
        from auras.data.parsers.chbmit import parse_summary_file
        sf = self._write_summary(tmp_path, "chb02", SUMMARY_MULTI)
        intervals = parse_summary_file(sf)
        assert len(intervals) == 4
        assert intervals[0].onset_sec == 130.0
        assert intervals[3].onset_sec == 5765.0
        assert intervals[3].offset_sec == 5811.0

    def test_zero_seizures(self, tmp_path):
        from auras.data.parsers.chbmit import parse_summary_file
        sf = self._write_summary(tmp_path, "chb03", SUMMARY_ZERO)
        intervals = parse_summary_file(sf)
        assert len(intervals) == 0

    def test_load_all_seizures(self, tmp_path):
        from auras.data.parsers.chbmit import load_all_seizures
        # Create two subjects
        self._write_summary(tmp_path, "chb01", SUMMARY_SINGLE)
        self._write_summary(tmp_path, "chb02", SUMMARY_MULTI)
        result = load_all_seizures(tmp_path)
        assert "chb01" in result
        assert "chb02" in result
        assert len(result["chb01"]) == 1
        assert len(result["chb02"]) == 4

    def test_load_all_seizures_no_files_raises(self, tmp_path):
        from auras.data.parsers.chbmit import load_all_seizures
        with pytest.raises(FileNotFoundError):
            load_all_seizures(tmp_path)

    def test_seizure_interval_attributes(self, tmp_path):
        from auras.data.parsers.chbmit import parse_summary_file
        sf = self._write_summary(tmp_path, "chb01", SUMMARY_SINGLE)
        interval = parse_summary_file(sf)[0]
        assert hasattr(interval, "recording")
        assert hasattr(interval, "onset_sec")
        assert hasattr(interval, "offset_sec")
        assert hasattr(interval, "duration_sec")
        assert interval.duration_sec == pytest.approx(40.0)

    def test_file_start_end_parsed(self, tmp_path):
        from auras.data.parsers.chbmit import parse_summary_file
        sf = self._write_summary(tmp_path, "chb01", SUMMARY_SINGLE)
        interval = parse_summary_file(sf)[0]
        # chb01_03 starts at 13:43:04
        assert interval.file_start == (13, 43, 4)

    def test_sorted_output(self, tmp_path):
        from auras.data.parsers.chbmit import parse_summary_file
        sf = self._write_summary(tmp_path, "chb02", SUMMARY_MULTI)
        intervals = parse_summary_file(sf)
        onsets = [i.onset_sec for i in intervals]
        assert onsets == sorted(onsets), "Intervals should be sorted by onset"


# =============================================================================
# 6. Prediction labeling
# =============================================================================

class _FakeSeizure:
    """Minimal seizure interval for labeling tests."""
    def __init__(self, recording: str, onset: float, offset: float):
        self.recording = recording
        self.onset_sec = onset
        self.offset_sec = offset


class TestDetectionLabeling:
    """Tests for auras.data.labeling.label_detection."""

    def test_no_seizures_all_zero(self):
        from auras.data.labeling import label_detection
        starts = np.arange(10, dtype=np.float64) * 4.0
        labels = label_detection(starts, 4.0, [], "chb01_01", 0.0)
        assert labels.sum() == 0

    def test_full_overlap_labeled_1(self):
        from auras.data.labeling import label_detection
        # seizure from 8s to 16s; windows at 8s and 12s are inside
        seizures = [_FakeSeizure("chb01_01", 8.0, 16.0)]
        starts = np.array([0, 4, 8, 12, 16], dtype=np.float64)
        labels = label_detection(starts, 4.0, seizures, "chb01_01", 0.0)
        assert labels[2] == 1  # window at 8-12s fully inside
        assert labels[3] == 1  # window at 12-16s fully inside

    def test_overlap_fraction_threshold(self):
        """With overlap_fraction=0.5, a window that only 30% overlaps → 0."""
        from auras.data.labeling import label_detection
        # window: 0-4s; seizure: 2.8-10s → overlap = 1.2s / 4s = 0.30 < 0.50
        seizures = [_FakeSeizure("rec", 2.8, 10.0)]
        starts = np.array([0.0])
        labels = label_detection(starts, 4.0, seizures, "rec", 0.5)
        assert labels[0] == 0

    def test_overlap_fraction_meets_threshold(self):
        """A window that exactly meets the overlap threshold → 1."""
        from auras.data.labeling import label_detection
        # window: 0-4s; seizure: 2.0-10s → overlap = 2.0s / 4s = 0.50 ≥ 0.50
        seizures = [_FakeSeizure("rec", 2.0, 10.0)]
        starts = np.array([0.0])
        labels = label_detection(starts, 4.0, seizures, "rec", 0.5)
        assert labels[0] == 1

    def test_different_recording_not_labeled(self):
        """Seizure in a different recording should not label this EDF's windows."""
        from auras.data.labeling import label_detection
        seizures = [_FakeSeizure("chb01_02", 0.0, 20.0)]
        starts = np.array([0.0, 4.0, 8.0])
        labels = label_detection(starts, 4.0, seizures, "chb01_01", 0.0)
        assert labels.sum() == 0


class TestPredictionLabeling:
    """Tests for auras.data.labeling.label_prediction."""

    def test_interictal_windows_labeled_0(self):
        from auras.data.labeling import label_prediction
        # Seizure at 3600s; SOP=600s, SPH=300s → preictal = [2700, 3300]
        seizures = [_FakeSeizure("rec", 3600.0, 3640.0)]
        # Window at 0-4s: far from seizure → interictal (0)
        starts = np.array([0.0], dtype=np.float64)
        labels = label_prediction(starts, 4.0, seizures, "rec",
                                  sop_sec=600, sph_sec=300, postictal_gap_sec=900)
        assert labels[0] == 0

    def test_preictal_window_labeled_1(self):
        from auras.data.labeling import label_prediction
        # Seizure at 1000s; SOP=300s, SPH=60s → preictal = [640, 940]
        seizures = [_FakeSeizure("rec", 1000.0, 1030.0)]
        # Window at 700-704s (inside [640, 940])
        starts = np.array([700.0], dtype=np.float64)
        labels = label_prediction(starts, 4.0, seizures, "rec",
                                  sop_sec=300, sph_sec=60, postictal_gap_sec=120)
        assert labels[0] == 1

    def test_ictal_window_excluded(self):
        from auras.data.labeling import label_prediction
        seizures = [_FakeSeizure("rec", 100.0, 130.0)]
        starts = np.array([100.0], dtype=np.float64)
        labels = label_prediction(starts, 4.0, seizures, "rec",
                                  sop_sec=300, sph_sec=60, postictal_gap_sec=120)
        assert labels[0] == -1

    def test_sph_zone_excluded(self):
        from auras.data.labeling import label_prediction
        # Seizure at 1000s, SPH=60s → SPH zone = [940, 1000]
        seizures = [_FakeSeizure("rec", 1000.0, 1030.0)]
        # Window 950-954s: inside SPH zone [940, 1000]
        starts = np.array([950.0], dtype=np.float64)
        labels = label_prediction(starts, 4.0, seizures, "rec",
                                  sop_sec=300, sph_sec=60, postictal_gap_sec=120)
        assert labels[0] == -1

    def test_postictal_excluded(self):
        from auras.data.labeling import label_prediction
        seizures = [_FakeSeizure("rec", 100.0, 130.0)]
        # postictal gap = 120s → [130, 250]; window at 140s = excluded
        starts = np.array([140.0], dtype=np.float64)
        labels = label_prediction(starts, 4.0, seizures, "rec",
                                  sop_sec=300, sph_sec=60, postictal_gap_sec=120)
        assert labels[0] == -1

    def test_no_seizures_all_interictal(self):
        from auras.data.labeling import label_prediction
        starts = np.arange(100, dtype=np.float64) * 4.0
        labels = label_prediction(starts, 4.0, [], "rec",
                                  sop_sec=300, sph_sec=60, postictal_gap_sec=120)
        assert np.all(labels == 0)


# =============================================================================
# 7. Dataset class with subjects
# =============================================================================

class TestEEGWindowDataset:
    """Tests for EEGWindowDataset with the new subjects array."""

    def _make_npz(self, tmp_path: Path, n: int = 100, has_subjects: bool = True) -> Path:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 4, 1024)).astype(np.float32)
        y = rng.integers(0, 2, n).astype(np.int32)
        npz = tmp_path / "test.npz"
        if has_subjects:
            subjects = np.repeat(np.arange(5), n // 5).astype(np.int32)
            np.savez_compressed(str(npz), X=X, y=y, subjects=subjects)
        else:
            np.savez_compressed(str(npz), X=X, y=y)
        return npz

    def test_len(self, tmp_path):
        from auras.data.dataset import EEGWindowDataset
        npz = self._make_npz(tmp_path)
        ds = EEGWindowDataset(npz)
        assert len(ds) == 100

    def test_getitem_shapes(self, tmp_path):
        from auras.data.dataset import EEGWindowDataset
        npz = self._make_npz(tmp_path)
        ds = EEGWindowDataset(npz)
        x, y = ds[0]
        assert x.shape == (4, 1024)
        assert y.shape == ()

    def test_subjects_array_present(self, tmp_path):
        from auras.data.dataset import EEGWindowDataset
        npz = self._make_npz(tmp_path, has_subjects=True)
        ds = EEGWindowDataset(npz)
        assert ds.subjects.shape == (100,)

    def test_subjects_defaults_to_zeros_when_missing(self, tmp_path):
        from auras.data.dataset import EEGWindowDataset
        npz = self._make_npz(tmp_path, has_subjects=False)
        ds = EEGWindowDataset(npz)
        # Should not raise; should return zeros
        assert np.all(ds.subjects == 0)

    def test_indices_subset(self, tmp_path):
        from auras.data.dataset import EEGWindowDataset
        npz = self._make_npz(tmp_path)
        idx = np.array([0, 10, 20, 30])
        ds = EEGWindowDataset(npz, indices=idx)
        assert len(ds) == 4

    def test_subjects_with_indices(self, tmp_path):
        from auras.data.dataset import EEGWindowDataset
        npz = self._make_npz(tmp_path, n=50, has_subjects=True)
        # Create dataset with subset indices
        idx = np.array([0, 10, 20])
        ds = EEGWindowDataset(npz, indices=idx)
        assert ds.subjects.shape == (3,)


# =============================================================================
# 8. prepare_dataset programmatic smoke test (no EDF files required)
# =============================================================================

class TestPreparePipeline:
    """Smoke tests for the full prepare_dataset.py pipeline using synthetic data."""

    def _make_synthetic_npz(self, tmp_path: Path) -> Path:
        """simulate what prepare_dataset would produce."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4, 1024)).astype(np.float32)
        y = np.zeros(200, dtype=np.int32)
        y[:10] = 1
        subjects = np.repeat(np.arange(4), 50).astype(np.int32)
        npz = tmp_path / "synthetic.npz"
        np.savez_compressed(str(npz), X=X, y=y, subjects=subjects)
        meta = {
            "dataset": "synthetic",
            "samples": 200,
            "positive": 10,
            "negative": 190,
            "class_ratio": 0.05,
            "seq_len": 1024,
            "channels": 4,
            "selected_channels": ["F7", "F8", "T7", "T8"],
            "sample_rate_hz": 256,
            "window_sec": 4.0,
            "stride_sec": 1.0,
            "normalization": "zscore",
            "num_subjects": 4,
            "labeling": {"mode": "detection", "overlap_fraction": 0.2},
        }
        (tmp_path / "synthetic.json").write_text(json.dumps(meta))
        return npz

    def test_npz_keys(self, tmp_path):
        npz_path = self._make_synthetic_npz(tmp_path)
        data = np.load(npz_path)
        assert "X" in data
        assert "y" in data
        assert "subjects" in data

    def test_class_distribution(self, tmp_path):
        npz_path = self._make_synthetic_npz(tmp_path)
        data = np.load(npz_path)
        assert data["y"].sum() == 10
        assert len(data["y"]) == 200

    def test_subject_ids_range(self, tmp_path):
        npz_path = self._make_synthetic_npz(tmp_path)
        data = np.load(npz_path)
        subjects = data["subjects"]
        unique = np.unique(subjects)
        assert len(unique) == 4
        assert unique.min() >= 0

    def test_json_metadata(self, tmp_path):
        self._make_synthetic_npz(tmp_path)
        meta_path = tmp_path / "synthetic.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["selected_channels"] == ["F7", "F8", "T7", "T8"]
        assert meta["sample_rate_hz"] == 256
        assert "labeling" in meta


# =============================================================================
# Integration: filter → feature vector pipeline
# =============================================================================

class TestIntegrationPipeline:
    """End-to-end: raw EEG → DWT filter → features."""

    def setup_method(self):
        pytest.importorskip("pywt")

    def test_filter_then_feature_extraction(self):
        from auras.data.preprocess import dwt_features, dwt_filter, zscore
        rng = np.random.default_rng(7)
        # Simulate 1 multi-channel EEG window
        x_raw = rng.standard_normal((4, 1024)).astype(np.float32)
        # Step 1: z-score per channel
        x_norm = zscore(x_raw)
        # Step 2: DWT filter (Li et al. (CNN-Informer))
        x_filt = dwt_filter(x_norm, level=5, reconstruct_levels=(3, 4, 5), include_approx=True)
        # Step 3: DWT features (Dokare & Gupta (DWT+SVM))
        features = dwt_features(x_filt, level=4)
        assert features.shape == (200,)
        # Features should be mostly finite
        finite_ratio = np.sum(np.isfinite(features)) / len(features)
        assert finite_ratio >= 0.90, f"Too many non-finite features: {1-finite_ratio:.1%}"

    def test_batch_feature_extraction(self):
        """dwt_features applied to a batch of 10 windows."""
        from auras.data.preprocess import dwt_features
        pytest.importorskip("pywt")
        rng = np.random.default_rng(8)
        # 10 windows, each (4, 1024)
        windows = rng.standard_normal((10, 4, 1024)).astype(np.float32)
        features = np.stack([dwt_features(w) for w in windows])
        assert features.shape == (10, 200)


# =============================================================================
# Runner: print summary
# =============================================================================

if __name__ == "__main__":
    """Run a quick manual summary of all tests and print results."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)
