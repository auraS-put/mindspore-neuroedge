#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-experiments/runs/latest/checkpoints/best.ckpt}
OUT_DIR=artifacts/lite
mkdir -p "$OUT_DIR"

echo "Placeholder export for $CKPT" > "$OUT_DIR/model.ms"
echo "Placeholder quantization" > "$OUT_DIR/model_int8.ms"

echo "MindSpore Lite artifact placeholders created in $OUT_DIR"
