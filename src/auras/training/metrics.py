from __future__ import annotations


def recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return float(tp) / denom if denom else 0.0


def precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return float(tp) / denom if denom else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    denom = p + r
    return 2.0 * p * r / denom if denom else 0.0
