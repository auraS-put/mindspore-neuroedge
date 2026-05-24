#!/usr/bin/env python
"""Standalone timing benchmark — run on any machine (GPU/CPU).

Trains each model for 3 epochs with full paper-verified settings and
reports per-epoch wall-clock time. This lets you estimate total training
cost without committing to a full experiment.

Usage
-----
    # Full timing (3 epochs, full dataset — needs 7.4 GB NPZ on this machine):
    python scripts/timing_benchmark.py

    # Quick timing (3 epochs, 5000 windows subset — ~2 min per model on GPU):
    python scripts/timing_benchmark.py --max-windows 5000

    # Single model only:
    python scripts/timing_benchmark.py --model cnn_bilstm_attn

    # Specify data path (e.g., mounted OBS, NFS, or local copy):
    python scripts/timing_benchmark.py --data-dir /cache/data/processed
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Model → training config mapping (paper-verified)
MODEL_TRAINING_MAP = {
    "cnn_bilstm_attn": "conv_snn",       # Paper 18: AdamW, LR=1e-3, batch=64
    "eegformer": "eegformer",             # Paper 02: Adam, LR=5e-5, batch=16
    "cnn_informer": "cnn_informer",       # Paper 10: Adam, LR=1e-4, batch=32
    "pyramidal_cnn_bilstm": "lightweight", # Paper 04: Adam, LR=2e-5, batch=32
    "cam_cnn_bilstm": "cam_cnn_bilstm",   # Paper 19: Adam, LR=1e-3, batch=32
}


def run_timing(model_name: str, training_cfg_name: str, args) -> dict:
    """Train one model for N epochs and return timing info."""
    import mindspore as ms
    import numpy as np
    from omegaconf import OmegaConf

    from auras.models.factory import create_model
    from auras.training.losses import build_loss
    from auras.training.lr_schedulers import build_lr_schedule
    from auras.training.trainer import _build_datasets, _build_optimizer, _train_loop
    from auras.utils.reproducibility import seed_everything

    seed_everything(42)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # Build config
    data_cfg = OmegaConf.load(_ROOT / "configs/data/siena_sop15.yaml")
    model_cfg = OmegaConf.load(_ROOT / f"configs/model/{model_name}.yaml")
    training_cfg = OmegaConf.load(_ROOT / f"configs/training/{training_cfg_name}.yaml")

    # Override epochs to timing cap
    training_cfg.epochs = args.epochs

    run_cfg = OmegaConf.create({
        "seed": 42,
        "project_name": "auraS_timing",
        "output_dir": f"experiments/runs/timing/{model_name}",
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
    })

    if args.data_dir:
        run_cfg.data.processed_dir = args.data_dir
    if args.max_windows:
        run_cfg.data["dry_run_max_windows"] = args.max_windows

    # Suppress step logging for clean timing output
    run_cfg.training.log_every_steps = 9999

    print(f"\n{'─'*60}")
    print(f"  Model   : {model_name}")
    print(f"  Training: {training_cfg_name}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {training_cfg.batch_size}")
    print(f"  LR      : {training_cfg.learning_rate}")
    print(f"{'─'*60}")

    # Load data
    t_load = time.time()
    train_ds, val_ds, test_ds, meta = _build_datasets(run_cfg)
    load_time = time.time() - t_load

    n_train = meta["train_samples"]
    steps_per_epoch = max(n_train // run_cfg.training.batch_size, 1)
    print(f"  Data: {n_train} train samples, {steps_per_epoch} steps/epoch")
    print(f"  Data load time: {load_time:.1f}s")

    # Build model
    n_channels = len(run_cfg.data.channels.selected)
    model = create_model(run_cfg.model, num_channels=n_channels)
    n_params = model.count_params()
    print(f"  Params: {n_params:,}")

    # Build training components
    loss_fn = build_loss(run_cfg.training, meta["train_positive"], meta["train_negative"])
    lr_schedule = build_lr_schedule(run_cfg.training, steps_per_epoch)
    optimizer = _build_optimizer(model, lr_schedule, run_cfg.training)

    # Train with timing
    t_train = time.time()
    _train_loop(model, loss_fn, optimizer, train_ds, run_cfg.training,
                val_iterator=None)
    train_time = time.time() - t_train
    time_per_epoch = train_time / args.epochs

    result = {
        "model": model_name,
        "training_config": training_cfg_name,
        "params": n_params,
        "batch_size": int(training_cfg.batch_size),
        "train_samples": n_train,
        "steps_per_epoch": steps_per_epoch,
        "epochs_run": args.epochs,
        "total_train_time_s": round(train_time, 1),
        "time_per_epoch_s": round(time_per_epoch, 1),
        "time_per_step_s": round(train_time / (steps_per_epoch * args.epochs), 4),
        "data_load_time_s": round(load_time, 1),
    }

    # Extrapolate to full training
    full_epochs = {
        "cnn_bilstm_attn": 100,
        "eegformer": 150,
        "cnn_informer": 100,
        "pyramidal_cnn_bilstm": 300,
        "cam_cnn_bilstm": 40,
    }
    full_ep = full_epochs.get(model_name, 100)
    est_total_hours = (time_per_epoch * full_ep) / 3600
    result["full_training_epochs"] = full_ep
    result["estimated_total_hours"] = round(est_total_hours, 2)

    print(f"\n  ⏱  {args.epochs} epochs in {train_time:.1f}s"
          f" ({time_per_epoch:.1f}s/epoch, {result['time_per_step_s']:.4f}s/step)")
    print(f"  📊 Estimated full training ({full_ep} epochs): {est_total_hours:.1f}h")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODEL_TRAINING_MAP.keys()),
                        default=None, help="Run only this model (default: all 5)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for timing (default: 3)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Cap dataset size (e.g., 5000 for quick test)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory path")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_TRAINING_MAP.keys())

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  auraS Timing Benchmark                                 ║")
    print(f"║  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<55}║")
    print(f"║  Models: {len(models):<48}║")
    print(f"║  Epochs: {args.epochs:<48}║")
    if args.max_windows:
        print(f"║  Max windows: {args.max_windows:<43}║")
    print(f"╚══════════════════════════════════════════════════════════╝")

    # Detect device
    import mindspore as ms
    ms.set_context(mode=ms.PYNATIVE_MODE)
    device = ms.get_context("device_target")
    print(f"\n  Device: {device}")

    results = []
    for model_name in models:
        training_cfg = MODEL_TRAINING_MAP[model_name]
        try:
            result = run_timing(model_name, training_cfg, args)
            results.append(result)
        except Exception as exc:
            import traceback
            print(f"\n  ✗ {model_name} FAILED: {exc}")
            traceback.print_exc()
            results.append({"model": model_name, "status": "error", "error": str(exc)})

    # Summary table
    print(f"\n{'═'*70}")
    print(f"  TIMING SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Model':<25} {'Params':>8} {'s/epoch':>8} {'s/step':>8} {'Est. hours':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    total_hours = 0
    for r in results:
        if "time_per_epoch_s" in r:
            print(f"  {r['model']:<25} {r['params']:>8,} {r['time_per_epoch_s']:>8.1f}"
                  f" {r['time_per_step_s']:>8.4f} {r['estimated_total_hours']:>10.1f}")
            total_hours += r["estimated_total_hours"]
        else:
            print(f"  {r['model']:<25} {'ERROR':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    print(f"  {'TOTAL (sequential)':<52} {total_hours:>10.1f}h")
    print(f"{'═'*70}\n")

    # Save results
    out_path = Path("experiments/runs/timing/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
