"""CNN-Informer (A6) — CNN stem + Informer ProbSparse encoder (Paper 10 (Li et al. — CNN-Informer)).

Combines a 3-layer CNN feature extractor with stacked Informer encoder
layers to achieve O(L log L) complexity attention .  The self-attention
distilling mechanism progressively halves the sequence at each layer.

Paper 10 (Li et al. — CNN-Informer) settings (CNN-Informer):
  d_model=64, n_heads=4, e_layers=3, prob_factor=3 (c=3 optimal)
  Training: Adam LR=1e-4, batch=32, 4s windows, ELU activations

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→32, k=3, ELU)
  Conv1d(32→32, k=3, ELU)
  Conv1d(32→d_model, k=3, ELU)              ← 3-layer CNN stem
  3× InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
      each layer: ProbSparse attn + FFN + AttentionDistilling (T→T//2)
  Transpose → AdaptiveAvgPool1d(1) → Dropout → Dense(d_model→num_classes)
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel
from auras.models.modules import InformerEncoderLayer


class CNNInformer(BaseSeizureModel):
    """Informer-based seizure predictor with CNN front-end (~200–400 K params)."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        prob_factor: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        # CNN stem — 3 layers with ELU, all pad="same" to preserve T
        self.cnn = nn.SequentialCell(
            nn.Conv1d(num_channels, 32, 3, pad_mode="same"),
            nn.ELU(),
            nn.Conv1d(32, 32, 3, pad_mode="same"),
            nn.ELU(),
            nn.Conv1d(32, d_model, 3, pad_mode="same"),
            nn.ELU(),
        )

        # Informer encoder stack (each layer halves T via distilling)
        self.encoder = nn.SequentialCell(
            *[
                InformerEncoderLayer(d_model, n_heads, d_ff, prob_factor, dropout)
                for _ in range(e_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(d_model, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.cnn(x)                    # (B, d_model, T)
        x = x.transpose(0, 2, 1)           # (B, T, d_model)
        x = self.encoder(x)                # (B, T/2^e_layers, d_model)
        x = x.transpose(0, 2, 1)           # (B, d_model, T')
        x = self.pool(x).squeeze(-1)       # (B, d_model)
        return self.classifier(x)
