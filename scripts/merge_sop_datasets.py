"""Merge three SOP-variant NPZ files into a single multi-target NPZ.

The three files (siena_sop5, siena_sop10, siena_sop15) share identical X and
subjects arrays — only the preictal ``y`` labels differ.  This script creates:

    data/processed/siena_sop_merged.npz
        X          (N, 4, 1024)  float32   — shared signal windows
        subjects   (N,)          int32     — subject IDs
        y_sop5     (N,)          int32     — preictal labels, SOP = 5 min
        y_sop10    (N,)          int32     — preictal labels, SOP = 10 min
        y_sop15    (N,)          int32     — preictal labels, SOP = 15 min

    data/processed/siena_sop_merged.json  — combined metadata

Usage
-----
    python scripts/merge_sop_datasets.py
    python scripts/merge_sop_datasets.py --processed-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing the SOP variant NPZ files.",
    )
    args = parser.parse_args()

    proc = Path(args.processed_dir)

    VARIANTS = [
        ("y_sop5",  "siena_sop5",  5),
        ("y_sop10", "siena_sop10", 10),
        ("y_sop15", "siena_sop15", 15),
    ]

    print("Loading SOP variant datasets …")
    X = None
    subjects = None
    label_arrays: dict[str, np.ndarray] = {}
    meta_by_sop: dict[int, dict] = {}

    for key, name, sop in VARIANTS:
        npz_path  = proc / f"{name}.npz"
        json_path = proc / f"{name}.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing: {npz_path}  (run prepare_dataset.py first)")

        print(f"  Loading {npz_path.name} …", end=" ", flush=True)
        d = np.load(str(npz_path), mmap_mode="r")
        y_i = d["y"]

        if X is None:
            # Take shared arrays from the first variant
            X = d["X"][:]                  # copy to RAM
            subjects = d["subjects"][:]
            print(f"X={X.shape}  dtype={X.dtype}", end="")

        # Shape consistency check
        assert len(y_i) == len(X), (
            f"Shape mismatch: {name} has {len(y_i)} windows, "
            f"but first variant has {len(X)}"
        )
        label_arrays[key] = y_i[:]        # copy int array
        pos = int(y_i.sum())
        print(f"  pos={pos} ({pos/len(y_i)*100:.1f}%)")

        if json_path.exists():
            meta_by_sop[sop] = json.load(open(json_path))

    # Build merged metadata
    base_meta = meta_by_sop[15] if 15 in meta_by_sop else next(iter(meta_by_sop.values()))
    merged_meta = {
        "dataset": "siena_sop_merged",
        "dataset_cfg": "configs/data/siena_sop_merged.yaml",
        "processed_npz": str(proc / "siena_sop_merged.npz"),
        "description": "Single merged NPZ with one y column per SOP variant. X_dwt contains precomputed DWT features (200-dim per window).",
        "samples": int(len(X)),
        "channels": int(X.shape[1]),
        "seq_len": int(X.shape[2]),
        "sample_rate_hz": base_meta.get("sample_rate_hz", 256),
        "window_sec": base_meta.get("window_sec", 4.0),
        "stride_sec": base_meta.get("stride_sec", 1.0),
        "normalization": base_meta.get("normalization", "zscore"),
        "record_count": base_meta.get("record_count", 41),
        "num_subjects": base_meta.get("num_subjects", 14),
        "subject_map": base_meta.get("subject_map", {}),
        "selected_channels": base_meta.get("selected_channels", ["F7", "F8", "T7", "T8"]),
        "label_keys": {
            "y_sop5":  {"sop_minutes": 5,  "positive": int(label_arrays["y_sop5"].sum())},
            "y_sop10": {"sop_minutes": 10, "positive": int(label_arrays["y_sop10"].sum())},
            "y_sop15": {"sop_minutes": 15, "positive": int(label_arrays["y_sop15"].sum())},
        },
        "labeling_common": {
            "mode":                   base_meta.get("labeling", {}).get("mode", "prediction"),
            "overlap_fraction":       base_meta.get("labeling", {}).get("overlap_fraction", 0.2),
            "sph_minutes":            base_meta.get("labeling", {}).get("sph_minutes", 5),
            "postictal_gap_minutes":  base_meta.get("labeling", {}).get("postictal_gap_minutes", 15),
        },
    }

    # Precompute DWT features (200-dim per window) — stored as X_dwt
    print("\nComputing DWT features (parallelised) …", flush=True)
    from auras.data.preprocess import dwt_features  # noqa: PLC0415

    n_windows = len(X)
    batch = 1000

    def _feat_batch(start: int) -> np.ndarray:
        end = min(start + batch, n_windows)
        return np.stack([dwt_features(X[i]) for i in range(start, end)], axis=0)

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_feat_batch)(s) for s in range(0, n_windows, batch)
    )
    X_dwt = np.concatenate(results, axis=0)  # (N, 200)
    print(f"X_dwt shape: {X_dwt.shape}  dtype: {X_dwt.dtype}")

    # Save merged NPZ
    out_npz = proc / "siena_sop_merged.npz"
    print(f"\nSaving {out_npz} …", end=" ", flush=True)
    np.savez(
        str(out_npz),
        X=X,
        X_dwt=X_dwt,
        subjects=subjects,
        **label_arrays,
    )
    size_gb = out_npz.stat().st_size / 1e9
    print(f"{size_gb:.2f} GB")

    # Save merged JSON
    out_json = proc / "siena_sop_merged.json"
    out_json.write_text(json.dumps(merged_meta, indent=2))
    print(f"Saved {out_json}")

    print("\nMerge complete:")
    print(f"  {out_npz.name}  ({size_gb:.2f} GB)")
    print(f"  Keys: X, X_dwt, subjects, y_sop5, y_sop10, y_sop15")
    for sop_key, info in merged_meta["label_keys"].items():
        pos = info["positive"]
        print(f"    {sop_key}: {pos:,} positive ({pos/merged_meta['samples']*100:.1f}%)")


if __name__ == "__main__":
    main()
