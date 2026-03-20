"""Weights & Biases integration for experiment tracking.

Wraps wandb.init / wandb.log behind the :class:`BaseLogger` interface
so the training loop stays backend-agnostic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from auras.monitoring.base_logger import BaseLogger


class WandBLogger(BaseLogger):
    """Log metrics, configs, and artifacts to Weights & Biases.

    Parameters
    ----------
    cfg : OmegaConf
        Full experiment config (gets logged as the run config).
    project : str, optional
        WandB project name. Defaults to env ``WANDB_PROJECT`` or ``auraS``.
    """

    def __init__(self, cfg=None, project: str | None = None):
        try:
            import wandb
        except ImportError:
            raise ImportError("Install wandb: pip install wandb")

        self._wandb = wandb
        project = project or os.environ.get("WANDB_PROJECT", "auraS")

        config_dict = {}
        if cfg is not None:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(cfg, resolve=True)

        self._run = wandb.init(
            project=project,
            config=config_dict,
            reinit=True,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        self._wandb.config.update(config, allow_val_change=True)

    def log_table(self, key: str, columns, data) -> None:
        """Log a table (e.g. per-subject results)."""
        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({key: table})

    def finish(self) -> None:
        self._run.finish()
