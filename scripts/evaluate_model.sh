#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:-experiments/runs}
if [[ -f "$RUN_DIR/metrics.json" ]]; then
  cat "$RUN_DIR/metrics.json"
else
  LATEST=$(cat experiments/runs/latest 2>/dev/null || true)
  if [[ -n "$LATEST" && -f "$LATEST/metrics.json" ]]; then
    cat "$LATEST/metrics.json"
  else
    echo "No metrics found for evaluation placeholder."
  fi
fi
