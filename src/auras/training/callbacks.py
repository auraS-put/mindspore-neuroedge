"""Custom MindSpore callbacks for training monitoring.

Includes:
- MetricLoggerCallback: logs metrics to monitoring backends each epoch
- EarlyStoppingCallback: stops training when metric plateaus
- CheckpointCallback: saves top-K checkpoints by monitored metric
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import mindspore as ms
from mindspore.train.callback import Callback


class MetricLoggerCallback(Callback):
    """Log training metrics to monitoring backends (WandB, ModelArts, console).

    Parameters
    ----------
    logger : BaseLogger
        An instance from ``auras.monitoring`` that implements ``.log_metrics()``.
    """

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        if loss is not None:
            loss_val = float(loss.asnumpy()) if hasattr(loss, "asnumpy") else float(loss)
            if self.logger:
                self.logger.log_metrics({"train/loss": loss_val}, step=epoch)


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    monitor : str
        Metric name to watch (e.g. ``'val_recall'``).
    mode : str
        ``'max'`` or ``'min'`` — whether higher or lower is better.
    """

    def __init__(self, patience: int = 15, monitor: str = "val_recall", mode: str = "max"):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0

    def check(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class BestCheckpointCallback(Callback):
    """Save checkpoints ranked by a monitored metric.

    Parameters
    ----------
    save_dir : str | Path
        Directory to save ``.ckpt`` files.
    monitor : str
        Metric name to optimize.
    mode : str
        ``'max'`` or ``'min'``.
    save_top_k : int
        Keep only the top-K checkpoints.
    """

    def __init__(
        self,
        save_dir: str | Path = "experiments/runs",
        monitor: str = "val_recall",
        mode: str = "max",
        save_top_k: int = 3,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self._records: list = []  # (metric_value, path)

    def save_if_best(self, model, metric_value: float, epoch: int) -> Optional[Path]:
        """Save checkpoint if it ranks in top-K. Returns path or None."""
        ckpt_path = self.save_dir / f"epoch_{epoch:04d}_{self.monitor}_{metric_value:.4f}.ckpt"

        self._records.append((metric_value, ckpt_path))
        reverse = self.mode == "max"
        self._records.sort(key=lambda r: r[0], reverse=reverse)

        if len(self._records) > self.save_top_k:
            _, old_path = self._records.pop()
            if old_path.exists():
                old_path.unlink()

        if ckpt_path in [r[1] for r in self._records]:
            ms.save_checkpoint(model, str(ckpt_path))
            return ckpt_path
        return None
