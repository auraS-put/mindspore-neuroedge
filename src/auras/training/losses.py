"""Loss functions for imbalanced seizure detection.

Provides:
- WeightedCrossEntropyLoss: inverse-frequency class weighting
- FocalLoss: down-weighs easy samples (Lin et al., 2017)
- SSWCELoss: Sensitivity-Specificity Weighted CE
  Paper 01 (Ingolfsson et al. — BrainFuseNet)
- LabelSmoothingCE: cross-entropy with soft labels
  Paper 18 (Mehrabi et al. — ConvSNN)
"""

from __future__ import annotations

import mindspore
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


class SSWCELoss(nn.Cell):
    """Sensitivity-Specificity Weighted Cross-Entropy Loss.

    Combines weighted cross-entropy with soft-confusion-matrix penalties:

        Loss = WCE + α*(1 − Specificity) + β*(1 − Sensitivity)

    Using differentiable (soft) estimates of TP/FP/TN/FN computed from
    predicted probabilities rather than hard thresholded predictions.

    Paper 01 (Ingolfsson et al. — BrainFuseNet): achieves best FP/h metric
    by directly penalising both false positives (specificity term) and
    false negatives (sensitivity term).

    Parameters
    ----------
    alpha : float
        Weight on the (1 − Specificity) penalty.  Default 0.5.
    beta : float
        Weight on the (1 − Sensitivity) penalty.  Default 0.5.
    class_weights : Tensor, optional
        Per-class inverse-frequency weights for the CE backbone.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        # --- base cross-entropy ---
        ce = self.ce_loss(logits, labels)
        if self.class_weights is not None:
            ce = (ce * self.class_weights[labels]).mean()
        else:
            ce = ce.mean()

        # --- soft confusion matrix (differentiable) ---
        probs = ops.softmax(logits, axis=-1)
        p1 = probs[:, 1]                            # P(seizure), shape (B,)
        y = labels.astype(mindspore.float32)        # 0.0 or 1.0, shape (B,)
        not_y = 1.0 - y

        tp = (p1 * y).sum()
        fp = (p1 * not_y).sum()
        tn = ((1.0 - p1) * not_y).sum()
        fn = ((1.0 - p1) * y).sum()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        return ce + self.alpha * (1.0 - specificity) + self.beta * (1.0 - sensitivity)


class LabelSmoothingCE(nn.Cell):
    """Cross-entropy with label smoothing to prevent over-confident predictions.

    Converts hard one-hot targets to smoothed distributions:

        soft_y = (1 − ε) * onehot(y) + ε / K

    Paper 18 (Mehrabi et al. — ConvSNN): ε=0.1 combined with AdamW and
    OneCycleLR yields the best holistic result on the ConvSNN architecture.

    Parameters
    ----------
    epsilon : float
        Smoothing factor.  0.0 → standard cross-entropy.  Default 0.1.
    num_classes : int
        Number of output classes.  Default 2.
    class_weights : Tensor, optional
        Per-class inverse-frequency weights applied after smoothing.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_classes: int = 2,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.class_weights = class_weights
        self._on = mindspore.Tensor(1.0, mindspore.float32)
        self._off = mindspore.Tensor(0.0, mindspore.float32)

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        log_probs = ops.log_softmax(logits, axis=-1)        # (B, K)

        # Build soft targets
        one_hot = ops.one_hot(labels, self.num_classes, self._on, self._off)  # (B, K)
        smooth = one_hot * (1.0 - self.epsilon) + self.epsilon / self.num_classes

        # Per-sample CE: -sum(smooth * log_probs, axis=-1)
        loss = -(smooth * log_probs).sum(axis=-1)   # (B,)

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
    elif name == "sswce":
        alpha = cfg.loss.get("sswce_alpha", 0.5)
        beta = cfg.loss.get("sswce_beta", 0.5)
        return SSWCELoss(alpha=alpha, beta=beta, class_weights=class_weights)
    elif name == "label_smoothing":
        epsilon = cfg.loss.get("label_smoothing", 0.1)
        return LabelSmoothingCE(epsilon=epsilon, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss: {name}")
