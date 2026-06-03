#!/bin/bash
# SageMaker invokes: docker run <image> train
# This script handles the "train" command by running cloud_boot_benchmark.py

set -e

if [ "$1" = "train" ]; then
    echo "=== SageMaker Training Container Start ==="
    echo "CUDA visible devices: $NVIDIA_VISIBLE_DEVICES"
    python -c "import mindspore; print(f'MindSpore {mindspore.__version__}')"
    
    # Code channel is downloaded to /opt/ml/input/data/code/
    CODE_TAR="/opt/ml/input/data/code/auras_code.tar.gz"
    CODE_DIR="/opt/ml/code"
    
    if [ -f "$CODE_TAR" ]; then
        echo "Extracting code tarball from code channel..."
        tar -xzf "$CODE_TAR" -C "$CODE_DIR"
    fi
    
    # Copy boot script
    BOOT_PY="/opt/ml/input/data/code/boot.py"
    if [ -f "$BOOT_PY" ]; then
        cp "$BOOT_PY" "$CODE_DIR/cloud_boot_benchmark.py"
    fi
    
    # Run the benchmark script
    cd "$CODE_DIR"
    exec python cloud_boot_benchmark.py
elif [ "$1" = "serve" ]; then
    echo "Serving not implemented"
    exit 1
else
    # Default: run whatever was passed
    exec "$@"
fi
