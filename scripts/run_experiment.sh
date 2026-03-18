#!/usr/bin/env bash
set -euo pipefail

MODEL_CFG=${1:-configs/model/lstm.yaml}
TRAIN_CFG=${2:-configs/train/default.yaml}
PYTHON_BIN=${PYTHON_BIN:-python3}

PYTHONPATH=src "$PYTHON_BIN" -m auras.training.train --model "$MODEL_CFG" --train "$TRAIN_CFG" --out experiments/runs

echo "Experiment run scaffold generated."
