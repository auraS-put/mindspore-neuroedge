"""Main training loop for auraS seizure detection models.

Orchestrates: data loading → model creation → training → evaluation,
with integrated logging to WandB / ModelArts / console.

Usage:
    python -m auras.training.trainer                           # defaults
    python -m auras.training.trainer model=ghostnet1d data=siena
    python -m auras.training.trainer --mode eval --checkpoint path/to/model.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from auras.utils.reproducibility import seed_everything


def _load_configs():
    """Load and merge configs from CLI overrides."""
    parser = argparse.ArgumentParser(description="auraS Trainer")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Load sub-configs if they are string references
    if isinstance(cfg.get("data"), str):
        cfg.data = OmegaConf.load(f"configs/data/{cfg.data}.yaml")
    elif cfg.get("defaults"):
        for default in cfg.defaults:
            if isinstance(default, dict):
                for key, val in default.items():
                    if key != "_self_":
                        sub_cfg = OmegaConf.load(f"configs/{key}/{val}.yaml")
                        cfg[key] = sub_cfg

    if isinstance(cfg.get("model"), str):
        cfg.model = OmegaConf.load(f"configs/model/{cfg.model}.yaml")
    if isinstance(cfg.get("training"), str):
        cfg.training = OmegaConf.load(f"configs/training/{cfg.training}.yaml")

    # Apply CLI overrides like model=resnet1d
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg, args.mode, args.checkpoint


def _build_datasets(cfg):
    """Load processed .npz and build train/val/test splits."""
    from auras.data.dataset import build_mindspore_dataset

    npz_path = Path(cfg.data.processed_dir) / f"{cfg.data.name}.npz"
    if not npz_path.exists():
        print(f"ERROR: Processed data not found at {npz_path}")
        print("Run:  python scripts/prepare_dataset.py --config configs/data/siena.yaml")
        sys.exit(1)

    data = np.load(npz_path)
    y = data["y"]
    n = len(y)

    # Simple stratified split
    from sklearn.model_selection import train_test_split

    indices = np.arange(n)
    test_size = cfg.data.split.get("test_size", 0.2)
    val_size = cfg.data.split.get("val_size", 0.1)

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=cfg.seed
    )
    relative_val = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val, stratify=y[train_val_idx], random_state=cfg.seed
    )

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.get("num_workers", 4)

    train_ds = build_mindspore_dataset(npz_path, train_idx, batch_size, shuffle=True, num_workers=num_workers)
    val_ds = build_mindspore_dataset(npz_path, val_idx, batch_size, shuffle=False, num_workers=num_workers)
    test_ds = build_mindspore_dataset(npz_path, test_idx, batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_positive": int(y[train_idx].sum()),
        "train_negative": int(len(train_idx) - y[train_idx].sum()),
    }
    return train_ds, val_ds, test_ds, meta


def train(cfg):
    """Full training pipeline."""
    import mindspore as ms
    import mindspore.nn as nn
    from mindspore import Model

    from auras.models.factory import create_model
    from auras.training.losses import build_loss
    from auras.training.lr_schedulers import build_lr_schedule
    from auras.training.callbacks import MetricLoggerCallback, EarlyStoppingCallback, BestCheckpointCallback
    from auras.monitoring import wandb_logger

    seed_everything(cfg.seed)
    ms.set_context(mode=ms.GRAPH_MODE)

    # Data
    train_ds, val_ds, test_ds, meta = _build_datasets(cfg)
    print(f"Data: {meta['train_samples']} train / {meta['val_samples']} val / {meta['test_samples']} test")
    print(f"Class balance: {meta['train_positive']} pos / {meta['train_negative']} neg")

    # Model
    n_channels = len(cfg.data.channels.selected)
    model = create_model(cfg.model, num_channels=n_channels)
    print(f"Model: {cfg.model.name} ({model.count_params():,} params)")

    # Loss
    loss_fn = build_loss(cfg.training, meta["train_positive"], meta["train_negative"])

    # LR schedule
    steps_per_epoch = meta["train_samples"] // cfg.training.batch_size
    lr_schedule = build_lr_schedule(cfg.training, steps_per_epoch)

    # Optimizer
    opt_name = cfg.training.get("optimizer", "adam")
    if opt_name == "adam":
        optimizer = nn.Adam(model.trainable_params(), learning_rate=lr_schedule, weight_decay=cfg.training.weight_decay)
    elif opt_name == "adamw":
        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_schedule, weight_decay=cfg.training.weight_decay)
    else:
        optimizer = nn.SGD(model.trainable_params(), learning_rate=lr_schedule, weight_decay=cfg.training.weight_decay, momentum=0.9)

    # Monitoring
    logger = wandb_logger.WandBLogger(cfg) if cfg.get("wandb", True) else None

    # Callbacks
    callbacks = [MetricLoggerCallback(logger=logger)]

    # Train
    ms_model = Model(model, loss_fn=loss_fn, optimizer=optimizer)
    ms_model.train(
        epoch=cfg.training.epochs,
        train_dataset=train_ds,
        callbacks=callbacks,
        dataset_sink_mode=False,
    )

    # Save final
    output_dir = Path(cfg.output_dir) / cfg.model.name
    output_dir.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(model, str(output_dir / "final.ckpt"))
    print(f"Training complete. Checkpoint saved to {output_dir}")

    if logger:
        logger.finish()


def main():
    cfg, mode, checkpoint = _load_configs()

    if mode == "train":
        train(cfg)
    else:
        print("Evaluation mode — TODO: implement evaluation pipeline")


if __name__ == "__main__":
    main()
