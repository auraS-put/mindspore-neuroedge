"""Time-series data augmentations for EEG windows.

All transforms operate on numpy arrays of shape ``(C, T)`` and return
the same shape — suitable for on-the-fly augmentation in the DataLoader.
"""

from __future__ import annotations

import numpy as np


class Compose:
    """Chain multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y: np.ndarray):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class GaussianNoise:
    """Additive Gaussian noise."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray):
        if np.random.random() < self.p:
            x = x + np.random.normal(0, self.std, size=x.shape).astype(x.dtype)
        return x, y


class TimeShift:
    """Random circular shift along the time axis."""

    def __init__(self, max_shift: int = 50, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray):
        if np.random.random() < self.p:
            shift = np.random.randint(-self.max_shift, self.max_shift)
            x = np.roll(x, shift, axis=-1)
        return x, y


class AmplitudeScale:
    """Random scaling of signal amplitude."""

    def __init__(self, low: float = 0.8, high: float = 1.2, p: float = 0.5):
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray):
        if np.random.random() < self.p:
            scale = np.random.uniform(self.low, self.high)
            x = x * scale
        return x, y


class ChannelDropout:
    """Randomly zero-out one channel to simulate electrode disconnect."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray):
        if np.random.random() < self.p:
            ch = np.random.randint(0, x.shape[0])
            x[ch, :] = 0.0
        return x, y
