"""MobileViT adapted for 1-D EEG classification.

Hybrid CNN + Transformer: local MobileNet blocks extract short-range
features, while Transformer blocks capture global temporal dependencies.
Input: (B, C, T) → Output: (B, num_classes)
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel


class _TransformerBlock1D(nn.Cell):
    """Standard Transformer encoder block (self-attention + FFN)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([dim])
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm([dim])
        self.ffn = nn.SequentialCell(
            nn.Dense(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate=False),
            nn.Dropout(p=dropout),
            nn.Dense(int(dim * mlp_ratio), dim),
            nn.Dropout(p=dropout),
        )

    def construct(self, x: Tensor) -> Tensor:
        # x: (seq_len, B, dim)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _MobileViTBlock1D(nn.Cell):
    """Local CNN features → unfold to patches → Transformer → fold back."""

    def __init__(self, in_ch: int, dim: int, depth: int, num_heads: int, patch_size: int, **kw):
        super().__init__()
        self.patch_size = patch_size
        self.local_rep = nn.SequentialCell(
            nn.Conv1d(in_ch, in_ch, 3, pad_mode="same", group=in_ch),
            nn.Conv1d(in_ch, dim, 1, pad_mode="valid"),
        )
        self.transformers = nn.SequentialCell(
            *[_TransformerBlock1D(dim, num_heads, **kw) for _ in range(depth)]
        )
        self.proj = nn.Conv1d(dim, in_ch, 1, pad_mode="valid")

    def construct(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        local = self.local_rep(x)  # (B, dim, T)
        # Unfold into patches: (B, dim, n_patches, patch_size) → (n_patches*patch_size, B, dim)
        dim = local.shape[1]
        n_patches = T // self.patch_size
        t_crop = n_patches * self.patch_size
        local = local[:, :, :t_crop].view(B, dim, n_patches, self.patch_size)
        local = local.transpose(0, 2, 3, 1).view(B * n_patches, self.patch_size, dim)
        local = local.transpose(1, 0, 2)  # (patch_size, B*n_patches, dim)

        local = self.transformers(local)

        local = local.transpose(1, 0, 2).view(B, n_patches, self.patch_size, dim)
        local = local.transpose(0, 3, 1, 2).view(B, dim, t_crop)
        return x[:, :, :t_crop] + self.proj(local)


class MobileViT_1D(BaseSeizureModel):
    """MobileViT for 1-D time-series: CNN stem + MobileViT blocks + classifier."""

    def __init__(
        self,
        num_channels: int = 4,
        patch_size: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        transformer_depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        self.stem = nn.SequentialCell(
            nn.Conv1d(num_channels, 32, 3, stride=2, pad_mode="same"),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, 3, stride=2, pad_mode="same"),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )

        self.mvit_block = _MobileViTBlock1D(
            in_ch=64,
            dim=embed_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.SequentialCell(
            nn.Dense(64, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.mvit_block(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
