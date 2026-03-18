from __future__ import annotations


def is_risk(prob: float, threshold: float = 0.7) -> bool:
    return prob >= threshold
