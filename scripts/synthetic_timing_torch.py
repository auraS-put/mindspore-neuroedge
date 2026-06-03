#!/usr/bin/env python
"""PyTorch GPU timing benchmark — self-contained for Colab/Kaggle.

Mirrors the architecture of synthetic_timing.py (MindSpore) but uses PyTorch
which works out-of-the-box on Colab (T4 GPU) and Kaggle (P100 GPU).

Same model architectures, same tensor shapes, same batch sizes.
Results give accurate GPU timing estimates for planning MindSpore training
on ModelArts (same underlying CUDA/cuDNN kernels).

Usage (on Colab/Kaggle with GPU runtime):
    python synthetic_timing_torch.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn


def setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Results won't represent GPU timing.")
    print(f"PyTorch: {torch.__version__}, Device: {device}")
    return device


# ─── Model definitions (same architectures as MindSpore versions) ────────

class CnnBilstmAttn(nn.Module):
    """Paper 18 — DS-Conv + BiLSTM + Transformer attention. ~210K params."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 7, padding=3, groups=64),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2, groups=64),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.lstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        self.attn = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (B, 64, 64) → (B, 64, 64)
        x, _ = self.lstm(x)    # (B, 64, 128)
        x = self.attn(x)       # (B, 64, 128)
        x = x.mean(dim=1)      # (B, 128)
        return self.head(x)


class EEGFormer(nn.Module):
    """Paper 02 — Patch Transformer. ~280K params."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv1d(4, 64, 5, stride=5),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=5),
            nn.GELU(),
        )
        self.pos_embed = nn.Embedding(512, 128)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.patch_embed(x)       # (B, 128, T)
        x = x.transpose(1, 2)         # (B, T, 128)
        B, T, D = x.shape
        pos = self.pos_embed(torch.arange(T, device=x.device))
        x = x + pos.unsqueeze(0)
        x = self.encoder(x)
        x = x.mean(dim=1)             # (B, 128)
        return self.head(x)


class CnnInformer(nn.Module):
    """Paper 10 — CNN + ProbSparse Informer. ~196K params."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(128)
        # 3 encoder layers with distilling
        self.enc1 = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.pool3 = nn.MaxPool1d(2)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.cnn(x)              # (B, 64, 1024)
        x = self.pool(x)             # (B, 64, 128)
        x = x.transpose(1, 2)        # (B, 128, 64)
        x = self.enc1(x)
        x = self.pool1(x.transpose(1, 2)).transpose(1, 2)  # (B, 64, 64)
        x = self.enc2(x)
        x = self.pool2(x.transpose(1, 2)).transpose(1, 2)  # (B, 32, 64)
        x = self.enc3(x)
        x = self.pool3(x.transpose(1, 2)).transpose(1, 2)  # (B, 16, 64)
        x = x.mean(dim=1)            # (B, 64)
        return self.head(x)


class PyramidalCnnBilstm(nn.Module):
    """Paper 04 — Pyramidal stride CNN + BiLSTM. ~33K params."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 8, 7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 5, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, 3, stride=4, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, 32, bidirectional=True, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x)


class CamCnnBilstm(nn.Module):
    """Paper 19/20 — Channel Attention + CNN + BiLSTM. ~45K params."""
    def __init__(self):
        super().__init__()
        self.ca_fc = nn.Linear(4, 4)
        self.ca_sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d(4, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.lstm = nn.LSTM(64, 40, bidirectional=True, batch_first=True)
        self.head = nn.Linear(80, 2)

    def forward(self, x):
        gap = x.mean(dim=-1)  # (B, 4)
        w = self.ca_sigmoid(self.ca_fc(gap)).unsqueeze(-1)
        x = x * w
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x)


# ─── Timing logic ─────────────────────────────────────────────────────

MODELS = {
    "cnn_bilstm_attn":      (CnnBilstmAttn, 64, 100),
    "eegformer":            (EEGFormer, 16, 150),
    "cnn_informer":         (CnnInformer, 32, 100),
    "pyramidal_cnn_bilstm": (PyramidalCnnBilstm, 32, 300),
    "cam_cnn_bilstm":       (CamCnnBilstm, 32, 40),
}

FULL_DATASET_WINDOWS = 451344


def time_model(name: str, model_cls, batch_size: int, device: str, n_steps: int = 200):
    """Time forward+backward for n_steps and return avg time/step."""
    print(f"\n  Building {name} (batch={batch_size})...")
    model = model_cls().to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x_batch = torch.randn(batch_size, 4, 1024, device=device)
    y_batch = torch.randint(0, 2, (batch_size,), device=device)

    # Warmup
    print(f"  Warmup (10 steps)...")
    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

    # Sync before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    print(f"  Timing ({n_steps} steps)...")
    t0 = time.time()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    time_per_step = elapsed / n_steps

    print(f"  Done: {elapsed:.2f}s total, {time_per_step*1000:.2f} ms/step")
    return {
        "time_per_step_ms": round(time_per_step * 1000, 2),
        "params": n_params,
        "batch_size": batch_size,
    }


def main():
    device = setup()

    print(f"\n{'='*60}")
    print(f"  Synthetic Timing Benchmark (PyTorch)")
    print(f"  Device: {device}")
    print(f"  Full dataset: {FULL_DATASET_WINDOWS:,} windows")
    print(f"{'='*60}")

    results = {}
    for name, (model_cls, batch_size, full_epochs) in MODELS.items():
        try:
            r = time_model(name, model_cls, batch_size, device, n_steps=200)
            steps_per_epoch = FULL_DATASET_WINDOWS // batch_size
            total_steps = steps_per_epoch * full_epochs
            total_hours = (r["time_per_step_ms"] / 1000) * total_steps / 3600

            r.update({
                "steps_per_epoch": steps_per_epoch,
                "full_epochs": full_epochs,
                "total_steps": total_steps,
                "estimated_hours": round(total_hours, 2),
            })
            results[name] = r
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    # Summary table
    print(f"\n{'='*70}")
    print(f"  TIMING RESULTS — {device}")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Params':>8} {'ms/step':>8} {'steps/ep':>9} {'epochs':>7} {'Est.hours':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*10}")
    total_h = 0
    for name, r in results.items():
        if "error" not in r:
            print(f"  {name:<25} {r['params']:>8,} {r['time_per_step_ms']:>8.2f}"
                  f" {r['steps_per_epoch']:>9,} {r['full_epochs']:>7}"
                  f" {r['estimated_hours']:>10.1f}")
            total_h += r["estimated_hours"]
        else:
            print(f"  {name:<25} {'ERROR':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*10}")
    print(f"  {'TOTAL (sequential)':<53} {total_h:>10.1f}h")
    max_h = max((r.get('estimated_hours', 0) for r in results.values()), default=0)
    print(f"  {'TOTAL (5 parallel jobs)':<53} {max_h:>10.1f}h")
    print(f"{'='*70}\n")

    # JSON output
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
