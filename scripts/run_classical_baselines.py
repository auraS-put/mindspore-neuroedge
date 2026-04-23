"""Run all classical ML baselines with DWT features on preprocessed EEG data.

Workflow:
  1. Load a preprocessed NPZ file (produced by prepare_dataset.py).
  2. Extract 200-dim DWT feature vectors per window.
  3. Evaluate each model via Leave-One-Subject-Out cross-validation.
  4. Collect per-fold metrics, aggregate mean ± std.
  5. Print a summary table and save full results to JSON.

Usage
-----
    python scripts/run_classical_baselines.py \\
        --npz data/processed/siena_sop15.npz \\
        --output experiments/classical_ml_results.json

    # Run only specific models:
    python scripts/run_classical_baselines.py \\
        --npz data/processed/siena_sop15.npz \\
        --models svm_rbf random_forest xgboost

Paper references
----------------
    Paper 22 (Dokare & Gupta — DWT-SVM)  — SVM-RBF, 97.7% accuracy, 86.7% sensitivity
    Paper 07 (Dash et al. — TF-Wearable) — SVM/RF/XGB/LGB/KNN comparison
    Paper 11 (Djemal et al. — XGBoost-CS) — XGBoost, n_estimators=200, max_depth=8
    Paper 05 (GAT)                         — RF n_estimators=100 baseline
    Paper 14 (Manzouri et al. — EE-Implantable) — RF balanced, max_depth=10
    Paper 17 (Sánchez-Reyes et al. — PCA+DWT+SVM) — KNN k=3, distance-weighted
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make src importable when running as a script
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from auras.data.preprocess import dwt_features
from auras.experiment.cross_validation import loso_splits
from auras.models.classical_ml import list_classical_models, train_and_evaluate


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    X: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """Extract DWT features for every window in the dataset.

    Parameters
    ----------
    X : (N, C, T) float array — raw windows
    wavelet : str — PyWavelets wavelet identifier
    level   : int — decomposition depth (default 4 → 5 bands × 4 ch × 10 stats = 200 features)

    Returns
    -------
    (N, F) float32 feature matrix, F = C*(level+1)*10
    """
    print(f"Extracting DWT features for {len(X)} windows "
          f"(wavelet={wavelet}, level={level}) …")
    features = []
    for i, window in enumerate(X):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(X)}")
        features.append(dwt_features(window, wavelet=wavelet, level=level))
    feat_matrix = np.stack(features, axis=0)
    print(f"  Feature matrix shape: {feat_matrix.shape}  "
          f"(expected last dim = {X.shape[1]*(level+1)*10})")
    return feat_matrix


# ---------------------------------------------------------------------------
# LOSO evaluation loop
# ---------------------------------------------------------------------------

def run_loso(
    model_name: str,
    X_feats: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> dict:
    """Run Leave-One-Subject-Out evaluation for a single classical ML model.

    Returns
    -------
    dict with:
        model_name : str
        folds : list of per-fold metric dicts
        mean  : dict of mean metrics over folds
        std   : dict of std metrics over folds
    """
    folds = []
    for train_idx, test_idx in loso_splits(subjects, y):
        held_out = str(subjects[test_idx[0]])
        n_pos_train = int(y[train_idx].sum())
        n_neg_train = int(len(train_idx) - n_pos_train)

        if n_pos_train == 0:
            print(f"    [skip] {model_name} / {held_out}: no positive (seizure) samples in train")
            continue

        result = train_and_evaluate(
            model_name,
            X_feats[train_idx], y[train_idx],
            X_feats[test_idx],  y[test_idx],
        )
        result["held_out_subject"] = held_out
        folds.append(result)

    if not folds:
        return {"model_name": model_name, "folds": [], "mean": {}, "std": {}}

    # Aggregate
    metric_keys = [k for k in folds[0].keys()
                   if isinstance(folds[0][k], (int, float)) and k != "n_train"]
    mean_metrics = {k: float(np.mean([f[k] for f in folds])) for k in metric_keys}
    std_metrics  = {k: float(np.std ([f[k] for f in folds])) for k in metric_keys}

    return {
        "model_name": model_name,
        "n_folds": len(folds),
        "folds": folds,
        "mean": mean_metrics,
        "std": std_metrics,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLS = ["recall", "specificity", "precision", "f1", "auc_roc"]


def print_summary(all_results: list) -> None:
    """Print a compact comparison table to stdout."""
    header = f"{'Model':<20}{'Recall':>8}{'Spec':>8}{'Prec':>8}{'F1':>8}{'AUC':>8}"
    print("\n" + "=" * len(header))
    print("CLASSICAL ML LOSO RESULTS (mean ± std)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in all_results:
        name = r["model_name"]
        m = r.get("mean", {})
        s = r.get("std", {})
        if not m:
            print(f"  {name:<18}  (no folds)")
            continue
        row = f"{name:<20}"
        for col in _COLS:
            row += f"  {m.get(col, 0):.3f}"
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run classical ML baselines with DWT features (LOSO evaluation)"
    )
    parser.add_argument(
        "--npz", type=str, required=True,
        help="Path to preprocessed .npz file (X, y, subjects arrays)"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/classical_ml_results.json",
        help="Output JSON path for full results"
    )
    parser.add_argument(
        "--models", nargs="+", default=list_classical_models(),
        help="Space-separated list of model names to run (default: all)"
    )
    parser.add_argument(
        "--wavelet", type=str, default="db4",
        help="PyWavelets wavelet for DWT feature extraction (default: db4)"
    )
    parser.add_argument(
        "--level", type=int, default=4,
        help="DWT decomposition depth (default: 4 → 5 bands × 10 stats × C channels)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"ERROR: NPZ file not found: {npz_path}")
        print("Run: python scripts/prepare_dataset.py --config configs/data/siena.yaml")
        sys.exit(1)

    data = np.load(npz_path)
    X = data["X"]                   # (N, C, T)
    y = data["y"]                   # (N,)
    subjects = (
        data["subjects"]
        if "subjects" in data
        else np.zeros(len(y), dtype="U10")
    )
    print(f"Loaded: {X.shape}  y={np.bincount(y)}  subjects={len(np.unique(subjects))}")

    # ── Feature extraction ─────────────────────────────────────────────────
    t0 = time.time()
    X_feats = extract_features(X, wavelet=args.wavelet, level=args.level)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── LOSO evaluation ────────────────────────────────────────────────────
    all_results = []
    invalid = [m for m in args.models if m not in list_classical_models()]
    if invalid:
        print(f"WARNING: Unknown model names ignored: {invalid}")
    valid_models = [m for m in args.models if m in list_classical_models()]

    for model_name in valid_models:
        print(f"Running: {model_name} …")
        t0 = time.time()
        result = run_loso(model_name, X_feats, y, subjects)
        elapsed = time.time() - t0
        n_folds = result.get("n_folds", 0)
        mean_recall = result.get("mean", {}).get("recall", 0.0)
        print(f"  {n_folds} folds  recall={mean_recall:.3f}  ({elapsed:.1f}s)\n")
        all_results.append(result)

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary(all_results)

    # ── Save JSON ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
