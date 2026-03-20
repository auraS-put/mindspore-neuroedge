"""Learning rate schedulers for MindSpore training.

Wraps common schedules in a uniform interface.
"""

from __future__ import annotations

import math

import numpy as np
import mindspore.nn as nn


def build_lr_schedule(cfg, steps_per_epoch: int) -> list:
    """Build a learning rate schedule as a list of per-step LR values.

    MindSpore expects a list or ``nn.learning_rate_schedule`` —
    we produce a flat list for maximum flexibility.

    Parameters
    ----------
    cfg : training config (OmegaConf)
    steps_per_epoch : int
        Number of training steps per epoch.

    Returns
    -------
    lr_schedule : list of float
        LR value for each global step.
    """
    base_lr = cfg.learning_rate
    epochs = cfg.epochs
    warmup = cfg.scheduler.get("warmup_epochs", 0)
    min_lr = cfg.scheduler.get("min_lr", 1e-6)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    schedule_name = cfg.scheduler.name

    lr_values = []
    for step in range(total_steps):
        if step < warmup_steps:
            # Linear warmup
            lr = base_lr * (step + 1) / warmup_steps
        elif schedule_name == "cosine_annealing":
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        elif schedule_name == "step":
            decay_rate = cfg.scheduler.get("decay_rate", 0.1)
            decay_epochs = cfg.scheduler.get("decay_epochs", [30, 60, 90])
            epoch = step // steps_per_epoch
            lr = base_lr
            for de in decay_epochs:
                if epoch >= de:
                    lr *= decay_rate
        else:
            lr = base_lr

        lr_values.append(lr)

    return lr_values
