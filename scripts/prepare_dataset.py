"""Process raw EDF recordings into model-ready .npz windows.

Usage:
    python scripts/prepare_dataset.py --config configs/data/siena.yaml

Reads raw EDF files, applies filtering and windowing per config, and writes:
    - data/processed/{name}.npz   (X: float32, y: int32)
    - data/processed/{name}.json  (metadata)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from auras.data.channels import select_channels
from auras.data.preprocess import sliding_window, zscore


def _load_edf(path: Path, target_channels, target_rate: int):
    """Load an EDF file using MNE, pick channels, resample."""
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    # Standardize channel names (strip whitespace, upper-case)
    raw.rename_channels({ch: ch.strip().upper() for ch in raw.ch_names})

    available = [ch for ch in target_channels if ch in raw.ch_names]
    if not available:
        return None, None

    raw.pick(available)

    if raw.info["sfreq"] != target_rate:
        raw.resample(target_rate)

    return raw.get_data(), raw.info["sfreq"]


def _apply_filters(data: np.ndarray, sfreq: float, cfg):
    """Band-pass and notch filter the raw signal."""
    from scipy.signal import butter, filtfilt, iirnotch

    prep = cfg.preprocessing

    # Band-pass
    if prep.get("bandpass_low_hz") and prep.get("bandpass_high_hz"):
        low = prep.bandpass_low_hz / (sfreq / 2)
        high = prep.bandpass_high_hz / (sfreq / 2)
        b, a = butter(4, [low, high], btype="band")
        data = filtfilt(b, a, data, axis=-1)

    # Notch
    if prep.get("notch_hz"):
        b, a = iirnotch(prep.notch_hz, Q=30, fs=sfreq)
        data = filtfilt(b, a, data, axis=-1)

    # Z-score
    if prep.get("zscore"):
        data = zscore(data)

    return data


def prepare(cfg) -> None:
    root = Path(cfg.root)
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rate = cfg.get("target_rate_hz", cfg.sampling_rate_hz)
    win_samples = int(cfg.window.seconds * target_rate)
    stride_samples = int(cfg.window.stride_seconds * target_rate)

    # Load seizure annotations for the dataset
    from auras.data.parsers import siena as parser

    all_seizures = parser.load_all_seizures(root)

    all_X, all_y = [], []
    record_count = 0

    for edf_path in sorted(root.rglob("*.edf")):
        data, sfreq = _load_edf(edf_path, cfg.channels.selected, target_rate)
        if data is None:
            continue

        data = _apply_filters(data, sfreq, cfg)
        windows = sliding_window(data, win_samples, stride_samples)
        if windows.shape[0] == 0:
            continue

        # Build labels: 1 if window overlaps with any seizure, else 0
        subject_id = edf_path.parent.name
        seizures = all_seizures.get(subject_id, [])
        labels = np.zeros(windows.shape[0], dtype=np.int32)

        for i, start_sample in enumerate(range(0, data.shape[-1] - win_samples + 1, stride_samples)):
            win_start = start_sample / sfreq
            win_end = (start_sample + win_samples) / sfreq
            for sz in seizures:
                if edf_path.stem in sz.recording and sz.onset_sec < win_end and sz.offset_sec > win_start:
                    labels[i] = 1
                    break

        all_X.append(windows)
        all_y.append(labels)
        record_count += 1

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int32)

    npz_path = out_dir / f"{cfg.name}.npz"
    np.savez_compressed(str(npz_path), X=X, y=y)

    meta = {
        "dataset": cfg.name,
        "dataset_cfg": f"configs/data/{cfg.name}.yaml",
        "raw_root": str(cfg.root),
        "processed_npz": str(npz_path),
        "samples": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "seq_len": win_samples,
        "channels": len(cfg.channels.selected),
        "record_count": record_count,
        "sample_rate_hz_used": target_rate,
        "selected_channels": list(cfg.channels.selected),
    }
    json_path = out_dir / f"{cfg.name}.json"
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(y)} samples ({y.sum()} positive) → {npz_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset from raw EDF files")
    parser.add_argument("--config", type=str, required=True, help="Path to data YAML config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    prepare(cfg)


if __name__ == "__main__":
    main()
