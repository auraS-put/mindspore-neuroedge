"""Abstract base model for all auraS architectures.

Every model must subclass :class:`BaseSeizureModel` and implement
``construct()`` following the MindSpore ``nn.Cell`` contract.
"""

from __future__ import annotations

from abc import abstractmethod

import mindspore.nn as nn
from mindspore import Tensor


class BaseSeizureModel(nn.Cell):
    """Base class providing a consistent interface.

    Subclasses receive input of shape ``(B, C, T)`` — batch of
    multi-channel EEG windows — and return logits ``(B, num_classes)``.
    """

    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def construct(self, x: Tensor) -> Tensor:
        """Forward pass. Input: (B, C, T) → Output: (B, num_classes)."""
        ...

    def count_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.size for p in self.trainable_params())
