#!/usr/bin/env bash
# =============================================================================
# dry_run.sh — Validate data, optionally upload to OBS, then run dry training.
#
# Usage:
#   ./scripts/dry_run.sh                   # local only (no upload)
#   ./scripts/dry_run.sh --upload          # upload data to OBS first
#   ./scripts/dry_run.sh --upload-only     # upload but skip training
#   ./scripts/dry_run.sh --obs-dry-run     # show what would be uploaded, no training
#
# After the run, results and a full training log land in:
#   experiments/runs/dry_run/
#     training_YYYYMMDD_HHMMSS.log   ← complete console output
#     results_summary.json           ← per-model metrics
#     siena/<model>/rep_0/
#       model.ckpt                   ← MindSpore checkpoint
#       metrics.json                 ← test-set metrics for this model
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$ROOT/.venv/bin/python"
PROCESSED="$ROOT/data/processed/siena_sop_merged.npz"
EXPERIMENT_CFG="configs/experiment/dry_run.yaml"

DO_UPLOAD=0
UPLOAD_ONLY=0
OBS_DRY_RUN=0

# Parse args
for arg in "$@"; do
  case "$arg" in
    --upload)       DO_UPLOAD=1 ;;
    --upload-only)  DO_UPLOAD=1; UPLOAD_ONLY=1 ;;
    --obs-dry-run)  DO_UPLOAD=1; OBS_DRY_RUN=1; UPLOAD_ONLY=1 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── 0. Activate venv if not already active ─────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

# Load .env for credentials
if [[ -f "$ROOT/.env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +o allexport
fi

# ── 1. Validate processed data ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  auraS dry-run launcher"
echo "═══════════════════════════════════════════════"
echo ""

if [[ ! -f "$PROCESSED" ]]; then
  echo "ERROR: Processed data not found at $PROCESSED"
  echo "Run:  python scripts/prepare_dataset.py --config configs/data/siena.yaml"
  exit 1
fi

SIZE_GB=$(python3 -c "import os; print(f'{os.path.getsize(\"$PROCESSED\")/1e9:.2f}')")
echo "[1/3] Data file: $PROCESSED ($SIZE_GB GB) ✓"

# Quick sanity check on NPZ keys
echo "      Verifying NPZ structure …"
python3 - <<'PY'
import sys, numpy as np
d = np.load("data/processed/siena_sop_merged.npz", mmap_mode='r')
required = {"X", "subjects", "y_sop5", "y_sop10", "y_sop15"}
missing = required - set(d.keys())
if missing:
    print(f"ERROR: missing NPZ keys: {missing}")
    print("Regenerate with:  python scripts/merge_sop_datasets.py")
    sys.exit(1)
print(f"      X: {d['X'].shape}  subjects: {len(set(d['subjects'].tolist()))} unique")
for k in ['y_sop5', 'y_sop10', 'y_sop15']:
    y = d[k]; print(f"      {k}: pos={int(y.sum())} ({y.mean()*100:.1f}%)")
PY

# ── 2. Upload to OBS (optional) ────────────────────────────────────────────
if [[ $DO_UPLOAD -eq 1 ]]; then
  echo ""
  echo "[2/3] Uploading data to OBS …"
  OBS_ARGS=""
  if [[ $OBS_DRY_RUN -eq 1 ]]; then
    OBS_ARGS="--dry-run"
    echo "      (dry-run mode — no actual transfer)"
  fi
  # shellcheck disable=SC2086
  python "$SCRIPT_DIR/upload_obs.py" $OBS_ARGS
else
  echo ""
  echo "[2/3] OBS upload skipped (pass --upload to enable)"
fi

[[ $UPLOAD_ONLY -eq 1 ]] && { echo ""; echo "Upload-only mode — done."; exit 0; }

# ── 3. Run dry training ────────────────────────────────────────────────────
echo ""
echo "[3/3] Starting dry training run …"
echo "      Config: $EXPERIMENT_CFG"
echo "      Output: experiments/runs/dry_run/"
echo ""

cd "$ROOT"
python scripts/run_experiment.py \
  --config "$EXPERIMENT_CFG" \
  --backend local

echo ""
echo "═══════════════════════════════════════════════"
echo "  Dry run complete."
echo ""
echo "  Artifacts:"
echo "    experiments/runs/dry_run/training_*.log    ← full log"
echo "    experiments/runs/dry_run/results_summary.json"
echo "    experiments/runs/dry_run/siena/<model>/rep_0/model.ckpt"
echo "    experiments/runs/dry_run/siena/<model>/rep_0/metrics.json"
echo "═══════════════════════════════════════════════"
