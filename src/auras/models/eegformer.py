"""EEGformer (A5) — lightweight pure-Transformer for EEG (Paper 02 (Busia et al. — EEGformer)).

Adapted directly from the 50.6 K-param architecture in Paper 02 (Busia et al. — EEGformer).
Paper 02 uses TWO non-overlapping Conv1D layers for patch embedding (both k=5, both
stride=k so no overlap) before the single Transformer encoder block.

Paper 02 (Busia et al. — EEGformer) settings:
  embed_dim=128, num_heads=8, patch_size=5, mlp_ratio=2, dropout=0.1
  Training: Adam LR=5e-5, batch=16
  Two-phase: 100 epochs subject-independent → 50 epochs subject-specific fine-tuning
  Input: 4 × 2048 (4 channels × 8 s × 256 Hz); 8 s windows outperform 4 s

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→128, k=patch_size, s=patch_size)  ← patch layer 1 (non-overlapping)
  Conv1d(128→128, k=patch_size, s=patch_size) ← patch layer 2 (Paper 02 uses 2 layers)
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

        # Two non-overlapping patch embedding layers (Paper 02 uses 2× Conv1D, both k=5, stride=k)
        self.patch_embed1 = nn.Conv1d(
            num_channels, embed_dim, patch_size,
            stride=patch_size, pad_mode="valid",
        )
        self.patch_embed2 = nn.Conv1d(
            embed_dim, embed_dim, patch_size,
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
        x = self.patch_embed1(x)           # (B, embed_dim, n_patches_1)
        x = self.patch_embed2(x)           # (B, embed_dim, n_patches_2)
        x = x.transpose(2, 0, 1)          # (n_patches, B, embed_dim)
        seq_len = x.shape[0]
        x = x + self.pos_enc[:seq_len]    # add positional encoding
        x = self.transformer(x)           # (n_patches, B, embed_dim)
        x = x.mean(axis=0)                # (B, embed_dim) — patch mean-pool
        return self.classifier(x)
