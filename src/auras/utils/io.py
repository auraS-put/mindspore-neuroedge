"""Checkpoint I/O helpers for MindSpore models."""

from __future__ import annotations

from pathlib import Path


def save_checkpoint(model, path: str | Path) -> Path:
    """Save model parameters to a MindSpore checkpoint."""
    import mindspore as ms

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(model, str(path))
    return path


def load_checkpoint(model, path: str | Path):
    """Load parameters into a model from a checkpoint file."""
    import mindspore as ms

    param_dict = ms.load_checkpoint(str(path))
    ms.load_param_into_net(model, param_dict)
    return model
