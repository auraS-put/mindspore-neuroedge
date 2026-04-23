"""Process raw EDF recordings into model-ready .npz windows.

Usage
-----
    python scripts/prepare_dataset.py --config configs/data/siena.yaml
    python scripts/prepare_dataset.py --config configs/data/chbmit.yaml

Reads raw EDF files, applies filtering / optional DWT preprocessing, creates
sliding windows, assigns labels in either **detection** or **prediction** mode,
and writes:

    data/processed/{name}.npz   — keys: X (float32), y (int32), subjects (int32)
    data/processed/{name}.json  — metadata

Labeling modes
--------------
detection  (default)
    y = 1 if ≥ ``labeling.overlap_fraction`` of the window overlaps an
    annotated ictal interval, else 0.

prediction
    y = 1 if the window falls inside the **preictal** zone for any seizure:
        preictal = [onset − sph_min − sop_min, onset − sph_min]
    y = 0 if the window is a confirmed **interictal** segment (> postictal_gap
    minutes after any seizure end AND outside any other seizure's preictal zone).
    Excluded windows (ictal, SPH zone, postictal gap < postictal_gap_min) are
    dropped from the dataset entirely.

Preprocessing modes
-------------------
raw       Band-pass + notch + normalisation only (default)
dwt       DWT Db4 band-pass filter: reconstruct cD3–cD5 + cA5 → ~3–29 Hz
          (Li et al. (CNN-Informer) — CNN-Informer)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

from auras.data.channels import select_channels
from auras.data.labeling import label_detection, label_prediction
from auras.data.preprocess import (
    dwt_filter,
    minmax_normalize,
    sliding_window,
    zscore,
)


# ---------------------------------------------------------------------------
# EDF loading
# ---------------------------------------------------------------------------

def _load_edf(
    path: Path,
    target_channels: List[str],
    target_rate: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Load EDF, select channels, resample.  Returns (data, sfreq) or (None, None)."""
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    # Normalise channel names:
    #   1. Strip leading "EEG " / "EOG " prefixes common in Siena EDFs
    #   2. Strip whitespace and uppercase
    #   3. Remap old 10-20 names → new (T3→T7, T4→T8, T5→P7, T6→P8)
    _OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    rename_map = {}
    for ch in raw.ch_names:
        norm = ch.strip()
        for prefix in ("EEG ", "EOG ", "EMG ", "ECG ", "EKG "):
            if norm.upper().startswith(prefix):
                norm = norm[len(prefix):]
                break
        norm = norm.strip().upper()
        norm = _OLD_TO_NEW.get(norm, norm)
        if norm != ch:
            rename_map[ch] = norm
    if rename_map:
        raw.rename_channels(rename_map)

    available = [ch for ch in target_channels if ch in raw.ch_names]
    if not available:
        return None, None

    raw.pick(available)
    if raw.info["sfreq"] != target_rate:
        raw.resample(target_rate, verbose=False)

    return raw.get_data(), raw.info["sfreq"]


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def _apply_filters(data: np.ndarray, sfreq: float, cfg) -> np.ndarray:
    """Band-pass then notch filter, then normalise."""
    from scipy.signal import butter, filtfilt, iirnotch

    prep = cfg.preprocessing

    if prep.get("bandpass_low_hz") and prep.get("bandpass_high_hz"):
        nyq = sfreq / 2.0
        low = prep.bandpass_low_hz / nyq
        high = prep.bandpass_high_hz / nyq
        # clip to valid Butterworth range
        high = min(high, 0.999)
        b, a = butter(4, [low, high], btype="band")
        data = filtfilt(b, a, data, axis=-1)

    if prep.get("notch_hz"):
        b, a = iirnotch(prep.notch_hz, Q=30, fs=sfreq)
        data = filtfilt(b, a, data, axis=-1)

    return data


def _apply_normalization(data: np.ndarray, mode: str) -> np.ndarray:
    """Apply chosen normalisation: 'zscore' | 'minmax' | None."""
    if mode == "zscore":
        return zscore(data)
    if mode == "minmax":
        return minmax_normalize(data)
    return data


def _apply_dwt(data: np.ndarray, cfg) -> np.ndarray:
    """Apply DWT-based band-pass filter (Li et al. (CNN-Informer)) if configured."""
    dwt_cfg = cfg.preprocessing.get("dwt", None)
    if dwt_cfg is None:
        return data
    mode = str(dwt_cfg.get("mode", "off")).lower()
    if mode in ("off", "false", "0", "no"):
        return data
    if mode == "filter":
        # Li et al. (CNN-Informer): Db4 level-5, keep cD3+cD4+cD5+cA5 → ~3–29 Hz
        wavelet = dwt_cfg.get("wavelet", "db4")
        level = dwt_cfg.get("level", 5)
        keep_levels = tuple(dwt_cfg.get("reconstruct_levels", [3, 4, 5]))
        include_approx = dwt_cfg.get("include_approx", True)
        return dwt_filter(data, wavelet=wavelet, level=level,
                          reconstruct_levels=keep_levels,
                          include_approx=include_approx)
    raise ValueError(f"Unknown DWT mode: {mode!r}. Choose 'off' or 'filter'.")


# ---------------------------------------------------------------------------
# Window labeling — re-export from auras.data.labeling for backwards compat
# ---------------------------------------------------------------------------
_label_detection = label_detection
_label_prediction = label_prediction


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _get_parser(name: str):
    """Return the seizure parser module matching the dataset name."""
    if "siena" in name.lower():
        from auras.data.parsers import siena
        return siena
    if "chbmit" in name.lower() or "chb" in name.lower():
        from auras.data.parsers import chbmit
        return chbmit
    raise ValueError(f"No parser found for dataset name '{name}'. "
                     "Expected 'siena' or 'chbmit'.")


def prepare(cfg) -> None:  # noqa: C901 (acceptable complexity for a data pipeline)
    root = Path(cfg.root)
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rate = int(cfg.get("target_rate_hz", cfg.sampling_rate_hz))
    win_samples = int(cfg.window.seconds * target_rate)
    stride_samples = int(cfg.window.stride_seconds * target_rate)
    win_sec = float(cfg.window.seconds)

    # --- labeling configuration ---
    lab_cfg = cfg.get("labeling", {})
    label_mode = lab_cfg.get("mode", "detection")
    overlap_fraction = float(lab_cfg.get("overlap_fraction", 0.0))
    sop_sec = float(lab_cfg.get("sop_minutes", 15)) * 60
    sph_sec = float(lab_cfg.get("sph_minutes", 5)) * 60
    postictal_gap_sec = float(lab_cfg.get("postictal_gap_minutes", 15)) * 60

    # --- preprocessing normalization ---
    norm_mode = cfg.preprocessing.get("normalize", "zscore")

    # --- load seizure annotations ---
    parser = _get_parser(cfg.name)
    all_seizures = parser.load_all_seizures(root)
    print(f"Loaded seizure annotations for {len(all_seizures)} subjects.")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subjects: List[np.ndarray] = []

    # Build a stable integer ID for each subject
    subject_names = sorted(all_seizures.keys())
    # Include subjects present in EDF files but possibly missing from annotations
    subject_to_int: Dict[str, int] = {}
    subject_counter = 0

    record_count = 0
    skipped_no_channels = 0

    for edf_path in sorted(root.rglob("*.edf")):
        subject_id = edf_path.parent.name

        data, sfreq = _load_edf(edf_path, list(cfg.channels.selected), target_rate)
        if data is None:
            skipped_no_channels += 1
            continue

        # Assign integer subject ID
        if subject_id not in subject_to_int:
            subject_to_int[subject_id] = subject_counter
            subject_counter += 1
        subj_int = subject_to_int[subject_id]

        # filtering
        data = _apply_filters(data, sfreq, cfg)

        # DWT preprocessing (optional)
        data = _apply_dwt(data, cfg)

        # normalisation
        data = _apply_normalization(data, norm_mode)

        # windowing
        windows = sliding_window(data, win_samples, stride_samples)
        if windows.shape[0] == 0:
            continue

        n_windows = windows.shape[0]
        starts_sec = np.array([
            s / target_rate for s in range(0, data.shape[-1] - win_samples + 1, stride_samples)
        ], dtype=np.float64)

        seizures = all_seizures.get(subject_id, [])

        if label_mode == "detection":
            labels = label_detection(
                starts_sec, win_sec, seizures, edf_path.stem,
                overlap_fraction=overlap_fraction,
            )
            mask = np.ones(n_windows, dtype=bool)  # keep all

        elif label_mode == "prediction":
            labels = label_prediction(
                starts_sec, win_sec, seizures, edf_path.stem,
                sop_sec=sop_sec, sph_sec=sph_sec,
                postictal_gap_sec=postictal_gap_sec,
            )
            mask = labels >= 0  # drop excluded windows (-1)
            labels = labels[mask]
            windows = windows[mask]

        else:
            raise ValueError(f"Unknown labeling mode: {label_mode!r}")

        if windows.shape[0] == 0:
            continue

        all_X.append(windows.astype(np.float32))
        all_y.append(labels.astype(np.int32))
        all_subjects.append(np.full(windows.shape[0], subj_int, dtype=np.int32))
        record_count += 1

    if not all_X:
        print("WARNING: No windows were generated. Check dataset root and channel names.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)

    npz_path = out_dir / f"{cfg.name}.npz"
    # Save uncompressed so numpy can memory-map the array (instant load).
    # Compressed saves ~40% disk but makes every load decompress the full file.
    np.savez(str(npz_path), X=X, y=y, subjects=subjects)

    # subject ID → name lookup
    int_to_subject = {v: k for k, v in subject_to_int.items()}

    meta = {
        "dataset": cfg.name,
        "dataset_cfg": f"configs/data/{cfg.name}.yaml",
        "raw_root": str(cfg.root),
        "processed_npz": str(npz_path),
        "label_mode": label_mode,
        "samples": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "class_ratio": float(y.mean()),
        "seq_len": win_samples,
        "channels": int(data.shape[0]),
        "selected_channels": list(cfg.channels.selected),
        "sample_rate_hz": target_rate,
        "window_sec": float(cfg.window.seconds),
        "stride_sec": float(cfg.window.stride_seconds),
        "normalization": norm_mode,
        "record_count": record_count,
        "skipped_no_channels": skipped_no_channels,
        "num_subjects": len(subject_to_int),
        "subject_map": int_to_subject,
        "labeling": {
            "mode": label_mode,
            "overlap_fraction": overlap_fraction,
            "sop_minutes": sop_sec / 60,
            "sph_minutes": sph_sec / 60,
            "postictal_gap_minutes": postictal_gap_sec / 60,
        },
    }
    json_path = out_dir / f"{cfg.name}.json"
    json_path.write_text(json.dumps(meta, indent=2))

    print(
        f"Done: {len(y):,} windows  "
        f"({y.sum():,} positive / {(len(y)-y.sum()):,} negative)  "
        f"→  {npz_path}"
    )
    print(f"      {len(subject_to_int)} subjects, {record_count} recordings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EEG dataset from raw EDF files")
    parser.add_argument("--config", required=True, help="Path to data YAML config")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    prepare(cfg)


if __name__ == "__main__":
    main()

