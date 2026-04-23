#!/usr/bin/env bash
# =============================================================================
# prepare_all_sop.sh — Generate processed NPZ files for all three SOP variants.
#
# Creates:
#   data/processed/siena_sop5.npz   (SOP =  5 min)
#   data/processed/siena_sop10.npz  (SOP = 10 min — Paper 06)
#   data/processed/siena_sop15.npz  (SOP = 15 min — our primary)
#
# Each run takes ~10–15 min on CPU (resampling + filtering 21 EDFs).
# Total wall time: ~30–45 min. Can be parallelised with --parallel flag.
#
# Usage:
#   ./scripts/prepare_all_sop.sh              # sequential
#   ./scripts/prepare_all_sop.sh --parallel   # all 3 in background
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/scripts/prepare_dataset.py"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "$ROOT/.venv/bin/activate"
fi
if [[ -f "$ROOT/.env" ]]; then
  set -o allexport; source "$ROOT/.env"; set +o allexport
fi

PARALLEL=0
for arg in "$@"; do
  [[ "$arg" == "--parallel" ]] && PARALLEL=1
done

CONFIGS=(
  "configs/data/siena_sop5.yaml:data/processed/siena_sop5.npz:SOP=5min"
  "configs/data/siena_sop10.yaml:data/processed/siena_sop10.npz:SOP=10min (Paper 06)"
  "configs/data/siena_sop15.yaml:data/processed/siena_sop15.npz:SOP=15min (primary)"
)

echo ""
echo "══════════════════════════════════════════════"
echo "  Preparing SOP ablation datasets"
echo "══════════════════════════════════════════════"

if [[ $PARALLEL -eq 1 ]]; then
  echo "  Mode: parallel (all 3 in background)"
  echo ""
  PIDS=()
  for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r cfg npz label <<< "$entry"
    if [[ -f "$ROOT/$npz" ]]; then
      echo "  [$label] already exists — skipping"
      continue
    fi
    echo "  [$label] starting..."
    "$PYTHON" "$SCRIPT" --config "$cfg" 2>&1 | grep -v RuntimeWarning &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
else
  echo "  Mode: sequential"
  echo ""
  for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r cfg npz label <<< "$entry"
    if [[ -f "$ROOT/$npz" ]]; then
      echo "  [$label] already exists at $npz — skipping"
      continue
    fi
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Generating: $label"
    echo "  Config:     $cfg"
    echo "──────────────────────────────────────────────"
    cd "$ROOT"
    "$PYTHON" "$SCRIPT" --config "$cfg" 2>&1 | grep -v RuntimeWarning
  done
fi

echo ""
echo "══════════════════════════════════════════════"
echo "  Done. Checking outputs:"
for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r cfg npz label <<< "$entry"
  if [[ -f "$ROOT/$npz" ]]; then
    size=$(du -h "$ROOT/$npz" | cut -f1)
    echo "  ✓ $npz  ($size)"
  else
    echo "  ✗ $npz  MISSING — check errors above"
  fi
done
echo "══════════════════════════════════════════════"
echo ""
echo "Next step:"
echo "  python scripts/run_experiment.py --config configs/experiment/sop_ablation.yaml"
