"""Inference benchmarking — latency, throughput, model size.

Measures the practical edge-deployment metrics critical for
24/7 on-device seizure monitoring:
  - Inference latency (ms per window)
  - Throughput (windows/sec)
  - Model size (MB)
  - Estimated energy cost (mAs per inference)

Usage:
    python -m auras.deployment.benchmark --checkpoint path/to/model.ckpt --model ghostnet1d
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def benchmark_model(
    checkpoint_path: str,
    model_name: str,
    num_channels: int = 4,
    seq_len: int = 1024,
    n_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """Benchmark inference performance of a trained model.

    Returns
    -------
    dict with keys: latency_ms, throughput_wps, model_size_mb, params
    """
    import mindspore as ms
    from omegaconf import OmegaConf

    from auras.models.factory import create_model
    from auras.utils.io import load_checkpoint

    model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
    model = create_model(model_cfg, num_channels=num_channels)
    model = load_checkpoint(model, checkpoint_path)
    model.set_train(False)

    dummy = ms.Tensor(np.random.randn(1, num_channels, seq_len).astype(np.float32))

    # Warmup
    for _ in range(warmup):
        model(dummy)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    latency_ms = np.median(times)
    throughput = 1000.0 / latency_ms
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    n_params = model.count_params()

    results = {
        "model": model_name,
        "latency_ms": round(latency_ms, 2),
        "throughput_wps": round(throughput, 1),
        "model_size_mb": round(model_size_mb, 2),
        "params": n_params,
    }

    print(f"═══ Benchmark: {model_name} ═══")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="ghostnet1d")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()

    benchmark_model(args.checkpoint, args.model, args.channels, args.seq_len, args.n_runs)


if __name__ == "__main__":
    main()
