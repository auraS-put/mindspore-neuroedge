"""EEG preprocessing utilities.

All functions operate on numpy arrays with shape (C, T) where C = channels, T = time samples.

Normalization:
  - zscore:          per-channel z-score (Paper 18 (Mehrabi et al. — ConvSNN))
  - minmax_normalize: per-channel [0, 1] (Paper 20 (Wang Y. et al. — CAM-CNN))

DWT functions (require PyWavelets ≥ 1.5):
  - dwt_filter:    Db4 wavelet reconstruction for band-pass filtering (Paper 10 (Li et al. — CNN-Informer))
  - dwt_subbands:  Decompose into per-band time-domain signals (Paper 22 (Dokare & Gupta — DWT-SVM))
  - dwt_features:  Extract 10 statistical features × (level+1) bands × C channels (Paper 22 (Dokare & Gupta — DWT-SVM))

Windowing:
  - sliding_window: overlapping window extraction
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalisation (Paper 18 (Mehrabi et al. — ConvSNN) — per-channel z-score).

    Parameters
    ----------
    x : (C, T) float array
    eps : stability constant

    Returns
    -------
    (C, T) normalised array
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel min-max normalisation to [0, 1] (Paper 20 (Wang Y. et al. — CAM-CNN) — CAM-CNN).

    Parameters
    ----------
    x : (C, T) float array
    eps : stability constant to handle zero-range channels

    Returns
    -------
    (C, T) array with values in [0, 1]
    """
    x_min = x.min(axis=-1, keepdims=True)
    x_max = x.max(axis=-1, keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sliding_window(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Extract overlapping windows from a multi-channel signal.

    Parameters
    ----------
    x      : (C, T) float array
    window : window length in samples
    stride : step between consecutive window starts

    Returns
    -------
    (N, C, window) array where N = number of windows
    """
    if x.ndim != 2:
        raise ValueError(f"Expected shape (C, T), got {x.shape}")
    channels, time = x.shape
    if time < window:
        return np.empty((0, channels, window), dtype=x.dtype)
    starts = range(0, time - window + 1, stride)
    return np.stack([x[:, s: s + window] for s in starts], axis=0)


# ---------------------------------------------------------------------------
# DWT helpers (private)
# ---------------------------------------------------------------------------

def _sample_entropy(ts: np.ndarray, m: int = 2, r_factor: float = 0.2,
                    max_n: int = 256) -> float:
    """Sample entropy — complexity measure (Paper 22 (Dokare & Gupta — DWT-SVM), SAMPEN).

    Truncates to *max_n* samples to keep O(n²) computation tractable.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) > max_n:
        ts = ts[:max_n]
    n = len(ts)
    r = r_factor * ts.std(ddof=1)
    if r == 0:
        return 0.0

    def _count_pairs(m_len: int) -> int:
        count = 0
        for i in range(n - m_len):
            for j in range(i + 1, n - m_len):
                if np.max(np.abs(ts[i: i + m_len] - ts[j: j + m_len])) <= r:
                    count += 1
        return count

    b = _count_pairs(m)
    a = _count_pairs(m + 1)
    if b == 0:
        return 0.0
    return -float(np.log(a / b)) if a > 0 else float("inf")


def _permutation_entropy(ts: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Permutation entropy — Bandt & Pompe (2002) (Paper 22 (Dokare & Gupta — DWT-SVM))."""
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    if n < order:
        return 0.0
    counts: dict = {}
    for i in range(n - (order - 1) * delay):
        window = ts[i: i + order * delay: delay]
        key = tuple(np.argsort(window))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _shannon_entropy(ts: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy via histogram (Paper 22 (Dokare & Gupta — DWT-SVM))."""
    ts = np.asarray(ts, dtype=np.float64)
    hist, _ = np.histogram(ts, bins=bins)
    hist = hist[hist > 0].astype(np.float64)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def _band_stats(coeff_arr: np.ndarray) -> list:
    """Return 10 statistics for a 1-D coefficient array.

    Order: min, max, mean, var, std, skewness, kurtosis,
           sample_entropy, permutation_entropy, shannon_entropy.
    """
    arr = np.asarray(coeff_arr, dtype=np.float64)
    return [
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.var(arr)),
        float(np.std(arr)),
        float(scipy_skew(arr)),
        float(scipy_kurtosis(arr)),
        _sample_entropy(arr),
        _permutation_entropy(arr),
        _shannon_entropy(arr),
    ]


# ---------------------------------------------------------------------------
# DWT public API
# ---------------------------------------------------------------------------

def dwt_filter(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
    reconstruct_levels: tuple = (3, 4, 5),
    include_approx: bool = True,
) -> np.ndarray:
    """DWT-based band-pass filter via selective coefficient reconstruction.

    Implements the preprocessing from **Paper 10 (Li et al. — CNN-Informer)**:
      - Db4 wavelet, 5-scale decomposition
      - Reconstruct from cD3 + cD4 + cD5 + cA5 → approximately **3–29 Hz**
        at 256 Hz native sampling rate.

    Parameters
    ----------
    x                 : (C, T) float array  — raw multi-channel EEG
    wavelet           : PyWavelets wavelet name (default: 'db4')
    level             : decomposition depth (default: 5 → Paper 10 (Li et al. — CNN-Informer))
    reconstruct_levels: detail-coefficient levels to keep (tuple of ints).
                        At 256 Hz / level=5: (3,4,5) → cD3+cD4+cD5 ≈ 4–32 Hz.
    include_approx    : whether to include the approximation coefficients
                        (cA_level) in reconstruction (default: True).

    Returns
    -------
    (C, T) filtered array — same shape as input
    """
    import pywt  # lazy import; only needed when DWT mode is active

    c_dim, t_dim = x.shape
    result = np.zeros_like(x, dtype=np.float64)

    for c in range(c_dim):
        coeffs = pywt.wavedec(x[c].astype(np.float64), wavelet, level=level)
        # coeffs = [cA_level, cD_level, cD_{level-1}, ..., cD_1]
        # Index k=0 → cA_level; k≥1 → cD_{level+1-k}
        new_coeffs = []
        for k, c_arr in enumerate(coeffs):
            if k == 0:
                new_coeffs.append(c_arr if include_approx else np.zeros_like(c_arr))
            else:
                detail_level = level + 1 - k
                new_coeffs.append(c_arr if detail_level in reconstruct_levels
                                   else np.zeros_like(c_arr))
        reconstructed = pywt.waverec(new_coeffs, wavelet)
        result[c] = reconstructed[:t_dim]

    return result.astype(x.dtype)


def dwt_subbands(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """Decompose each channel into frequency sub-bands via DWT reconstruction.

    Implements the 5-band decomposition from **Paper 22 (Dokare & Gupta — DWT-SVM)**:
      At 256 Hz with level=4 (Db4):
        δ  (cA4):  0–4 Hz
        θ  (cD4):  4–8 Hz
        α  (cD3):  8–16 Hz
        β  (cD2): 16–32 Hz
        γ  (cD1): 32–64 Hz

    Each sub-band is reconstructed to the original time length T.

    Parameters
    ----------
    x      : (C, T) float array
    wavelet: PyWavelets wavelet name
    level  : decomposition depth (default: 4 → Paper 22 (Dokare & Gupta — DWT-SVM)'s 5 bands)

    Returns
    -------
    (C, level+1, T) array — sub-band dimension first per channel
    Convention: axis-1 index 0 = approx (lowest freq), last = highest detail
    """
    import pywt

    c_dim, t_dim = x.shape
    n_bands = level + 1
    result = np.zeros((c_dim, n_bands, t_dim), dtype=x.dtype)

    zeros_template: list | None = None

    for c in range(c_dim):
        coeffs = pywt.wavedec(x[c].astype(np.float64), wavelet, level=level)
        if zeros_template is None:
            zeros_template = [np.zeros_like(c_arr) for c_arr in coeffs]

        for band_idx, keep_k in enumerate(range(len(coeffs))):
            # keep only coefficients at index keep_k; zero all others
            band_coeffs = [np.zeros_like(c_arr) for c_arr in coeffs]
            band_coeffs[keep_k] = coeffs[keep_k]
            reconstructed = pywt.waverec(band_coeffs, wavelet)
            result[c, band_idx] = reconstructed[:t_dim]

    return result


def dwt_features(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """Extract 10 statistical features per DWT sub-band per channel.

    Implements the 200-dimensional feature vector from **Paper 22 (Dokare & Gupta — DWT-SVM)**:
      - 4 channels × (level+1=5) bands × 10 stats = **200 features**
      - Features: min, max, mean, var, std, skewness, kurtosis,
                  sample entropy, permutation entropy, Shannon entropy.

    Unlike :func:`dwt_subbands`, features are computed directly on the
    DWT coefficient arrays (not reconstructed time-domain signals), which
    is faster and equivalent for statistical characterisation.

    Parameters
    ----------
    x      : (C, T) float array
    wavelet: PyWavelets wavelet name
    level  : decomposition depth (default: 4 → 5 bands)

    Returns
    -------
    (C*(level+1)*10,) float32 feature vector
    For C=4, level=4: shape (200,)
    """
    import pywt

    features: list = []
    for c_idx in range(x.shape[0]):
        coeffs = pywt.wavedec(x[c_idx].astype(np.float64), wavelet, level=level)
        # Order: [cA_level, cD_level, ..., cD1]
        # Paper 22 (Dokare & Gupta — DWT-SVM) order: δ first (cA4), γ last (cD1) — same as pywt default
        for c_arr in coeffs:
            features.extend(_band_stats(c_arr))

    return np.array(features, dtype=np.float32)
