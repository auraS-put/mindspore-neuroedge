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


def _build_optimizer(model, lr_schedule, cfg):
    """Construct optimizer from config."""
    import mindspore.nn as nn

    opt_name = cfg.get("optimizer", "adam")
    wd = cfg.weight_decay
    if opt_name == "adamw":
        return nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_schedule, weight_decay=wd)
    elif opt_name == "adam":
        return nn.Adam(model.trainable_params(), learning_rate=lr_schedule, weight_decay=wd)
    else:
        return nn.SGD(model.trainable_params(), learning_rate=lr_schedule, weight_decay=wd, momentum=0.9)


def _train_loop(model, loss_fn, optimizer, train_ds, cfg, *, val_iterator=None):
    """Manual training loop with gradient clipping and optional early stopping.

    Uses ``mindspore.value_and_grad`` for a fully-custom per-step loop so
    that ``nn.ClipByGlobalNorm`` can be applied before the parameter update.

    Parameters
    ----------
    model : nn.Cell
    loss_fn : nn.Cell
    optimizer : nn.Optimizer
    train_ds : MindSpore GeneratorDataset (batched)
    cfg : training sub-config (OmegaConf)
    val_iterator : callable, optional
        Zero-argument callable that returns a fresh batch iterator yielding
        ``(Tensor, Tensor)`` for validation.  Used for early stopping.

    Returns
    -------
    best_epoch : int
    """
    import mindspore as ms
    import mindspore.ops as ops

    from auras.training.evaluator import evaluate_epoch

    clip_norm = cfg.get("gradient_clip_norm", 1.0)

    def forward_fn(data, label):
        return loss_fn(model(data), label)

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    patience = cfg.early_stopping.patience
    monitor_mode = cfg.early_stopping.get("mode", "max")
    best_val = -float("inf") if monitor_mode == "max" else float("inf")
    no_improve = 0
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.set_train(True)
        epoch_loss = 0.0
        n_batches = 0
        for X, label in train_ds.create_tuple_iterator():
            loss, grads = grad_fn(X, label)
            grads = ops.clip_by_global_norm(grads, clip_norm)
            optimizer(grads)
            epoch_loss += float(loss.asnumpy())
            n_batches += 1

        # Early stopping on validation recall
        if val_iterator is not None:
            result = evaluate_epoch(model, val_iterator())
            val_recall = result.segment.recall
            improved = val_recall > best_val if monitor_mode == "max" else val_recall < best_val
            if improved:
                best_val = val_recall
                no_improve = 0
                best_epoch = epoch + 1
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch + 1} (best={best_val:.4f} @ epoch {best_epoch})")
                break

    return best_epoch


def train(cfg):
    """Full training pipeline with gradient clipping."""
    import mindspore as ms

    from auras.models.factory import create_model
    from auras.training.losses import build_loss
    from auras.training.lr_schedulers import build_lr_schedule

    seed_everything(cfg.seed)
    ms.set_context(mode=ms.PYNATIVE_MODE)

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

    # LR schedule + optimizer
    steps_per_epoch = max(meta["train_samples"] // cfg.training.batch_size, 1)
    lr_schedule = build_lr_schedule(cfg.training, steps_per_epoch)
    optimizer = _build_optimizer(model, lr_schedule, cfg.training)

    # Val iterator factory for early stopping
    npz_path = Path(cfg.data.processed_dir) / f"{cfg.data.name}.npz"
    val_idx_arr = None  # saved inside _build_datasets — re-derive if needed

    _train_loop(model, loss_fn, optimizer, train_ds, cfg.training)

    # Save final checkpoint
    output_dir = Path(cfg.output_dir) / cfg.model.name
    output_dir.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(model, str(output_dir / "final.ckpt"))
    print(f"Training complete. Checkpoint saved to {output_dir}")


def loso_train(cfg) -> "LOSOResult":
    """Leave-One-Subject-Out training and evaluation.

    Trains a separate model for every held-out subject and evaluates it,
    then returns a :class:`~auras.training.evaluator.LOSOResult` with
    per-fold and aggregated metrics.

    Paper 02 (Busia et al. — EEGformer): LOSO is the gold-standard
    cross-patient evaluation; avoids window-level data leakage between
    train and test subjects present in simple random splits.

    Parameters
    ----------
    cfg : OmegaConf
        Full experiment config with ``data``, ``model``, ``training`` keys.

    Returns
    -------
    LOSOResult
    """
    import mindspore as ms
    from sklearn.model_selection import train_test_split

    from auras.data.dataset import build_mindspore_dataset
    from auras.experiment.cross_validation import loso_splits
    from auras.models.factory import create_model
    from auras.training.evaluator import LOSOResult, evaluate_epoch
    from auras.training.losses import build_loss
    from auras.training.lr_schedulers import build_lr_schedule

    seed_everything(cfg.seed)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    npz_path = Path(cfg.data.processed_dir) / f"{cfg.data.name}.npz"
    raw = np.load(npz_path)
    y = raw["y"]
    subjects = raw["subjects"] if "subjects" in raw else np.zeros(len(y), dtype="U10")

    loso_result = LOSOResult()
    window_len = float(cfg.data.get("window_len_s", 4.0))
    n_channels = len(cfg.data.channels.selected)
    batch_size = cfg.training.batch_size

    for fold_i, (train_idx, test_idx) in enumerate(loso_splits(subjects, y)):
        held_out = str(subjects[test_idx[0]])
        print(f"\nLOSO fold {fold_i + 1}: held-out={held_out}  "
              f"train={len(train_idx)}  test={len(test_idx)}")

        y_train = y[train_idx]
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        if n_pos == 0 or n_neg == 0:
            print(f"  Skipping fold — degenerate class distribution.")
            continue

        # Inner train/val split (for early stopping only)
        val_frac = cfg.data.split.get("val_size", 0.1)
        if len(train_idx) > 20:
            tr_idx, val_idx = train_test_split(
                train_idx, test_size=val_frac, stratify=y_train, random_state=cfg.seed
            )
        else:
            tr_idx, val_idx = train_idx, train_idx

        tr_ds = build_mindspore_dataset(npz_path, tr_idx, batch_size, shuffle=True)

        def make_val_iter(vi=val_idx):
            return build_mindspore_dataset(npz_path, vi, batch_size, shuffle=False).create_tuple_iterator()

        # Build fresh model + optimizer each fold
        model = create_model(cfg.model, num_channels=n_channels)
        loss_fn = build_loss(cfg.training, n_pos, n_neg)
        steps_per_epoch = max(len(tr_idx) // batch_size, 1)
        lr_schedule = build_lr_schedule(cfg.training, steps_per_epoch)
        optimizer = _build_optimizer(model, lr_schedule, cfg.training)

        _train_loop(model, loss_fn, optimizer, tr_ds, cfg.training, val_iterator=make_val_iter)

        # Evaluate on held-out subject
        test_ds = build_mindspore_dataset(npz_path, test_idx, batch_size, shuffle=False)
        fold_result = evaluate_epoch(model, test_ds.create_tuple_iterator(), window_length_s=window_len)
        loso_result.fold_results.append(fold_result)
        loso_result.subject_ids.append(held_out)

        print(f"  recall={fold_result.segment.recall:.3f}  "
              f"specificity={fold_result.segment.specificity:.3f}  "
              f"fp/h={fold_result.fp_per_hour:.2f}")

    if loso_result.fold_results:
        print(f"\n{loso_result.summary()}")

    return loso_result


def main():
    cfg, mode, checkpoint = _load_configs()

    if mode == "train":
        train(cfg)
    else:
        print("Evaluation mode — TODO: implement evaluation pipeline")


if __name__ == "__main__":
    main()

