"""Loss functions for imbalanced seizure detection.

Provides Weighted Cross-Entropy and Focal Loss, both critical
for handling the extreme class imbalance in EEG datasets.
"""

from __future__ import annotations

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class WeightedCrossEntropyLoss(nn.Cell):
    """Cross-entropy with per-class weights for imbalanced datasets.

    Parameters
    ----------
    class_weights : Tensor
        1-D tensor of shape ``(num_classes,)`` with per-class weights.
        Use higher weight for the minority (seizure) class.
    """

    def __init__(self, class_weights: Tensor):
        super().__init__()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.class_weights = class_weights

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        loss = self.loss_fn(logits, labels)
        weights = self.class_weights[labels]
        return (loss * weights).mean()


class FocalLoss(nn.Cell):
    """Focal Loss (Lin et al., 2017) — down-weighs well-classified samples.

    Particularly effective when positive class is extremely rare.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher gamma → more focus on hard examples.
    class_weights : Tensor, optional
        Per-class balancing weights (alpha in the original paper).
    """

    def __init__(self, gamma: float = 2.0, class_weights: Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        probs = ops.softmax(logits, axis=-1)
        # Gather probability of the true class
        batch_idx = ops.arange(labels.shape[0])
        p_t = probs[batch_idx, labels]
        focal_weight = (1.0 - p_t) ** self.gamma
        ce = -ops.log(p_t + 1e-8)

        loss = focal_weight * ce

        if self.class_weights is not None:
            loss = loss * self.class_weights[labels]

        return loss.mean()


def build_loss(cfg, num_positive: int, num_negative: int):
    """Factory: build loss function from training config.

    Parameters
    ----------
    cfg : training config (OmegaConf)
    num_positive : int
        Number of seizure samples.
    num_negative : int
        Number of non-seizure samples.
    """
    import mindspore
    import numpy as np

    # Compute inverse-frequency weights
    total = num_positive + num_negative
    w_neg = total / (2 * num_negative)
    w_pos = total / (2 * num_positive)
    class_weights = mindspore.Tensor(np.array([w_neg, w_pos], dtype=np.float32))

    name = cfg.loss.name
    if name == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights)
    elif name == "focal":
        return FocalLoss(gamma=cfg.loss.focal_gamma, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss: {name}")
