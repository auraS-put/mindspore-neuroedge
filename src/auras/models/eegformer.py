"""EEGformer (A5) — lightweight pure-Transformer for EEG (Paper 02 (Busia et al. — EEGformer)).

Adapted directly from the 50.6 K-param architecture in Paper 02 (Busia et al. — EEGformer).
A non-overlapping patch embedding projects the raw EEG into a
short token sequence, which is processed by a single Transformer
encoder block with learnable positional encoding.

Paper 02 (Busia et al. — EEGformer) settings:
  embed_dim=128, num_heads=8, patch_size=5, mlp_ratio=2, dropout=0.1
  Training: Adam LR=5e-4, batch=16, two-phase training

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→128, k=patch_size, s=patch_size)  ← non-overlapping patches
  Learnable positional encoding (max_seq, 1, 128)
  1× _TransformerBlock1D(128, 8 heads, mlp_ratio=2)
  Mean-pool over patches → Dropout → Dense(128→num_classes)
"""

from __future__ import annotations

import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor

from auras.models.base import BaseSeizureModel
from auras.models.mobilevit_1d import _TransformerBlock1D


class EEGformer(BaseSeizureModel):
    """Patch-based Transformer for EEG, ~50 K params (Paper 02 (Busia et al. — EEGformer))."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        patch_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        max_seq: int = 512,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        # Non-overlapping patch embedding (stride == kernel → no overlap)
        self.patch_embed = nn.Conv1d(
            num_channels, embed_dim, patch_size,
            stride=patch_size, pad_mode="valid",
        )

        # Learnable positional encoding broadcast over batch
        self.pos_enc = Parameter(
            mindspore.Tensor(
                np.zeros((max_seq, 1, embed_dim), dtype=np.float32)
            )
        )

        self.transformer = _TransformerBlock1D(
            dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(embed_dim, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)           # (B, embed_dim, n_patches)
        x = x.transpose(2, 0, 1)          # (n_patches, B, embed_dim)
        seq_len = x.shape[0]
        x = x + self.pos_enc[:seq_len]    # add positional encoding
        x = self.transformer(x)           # (n_patches, B, embed_dim)
        x = x.mean(axis=0)                # (B, embed_dim) — patch mean-pool
        return self.classifier(x)
