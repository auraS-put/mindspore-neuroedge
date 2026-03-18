#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw data/interim data/processed data/external
mkdir -p experiments/notebooks experiments/runs
mkdir -p logs

echo "Environment directories initialized."
