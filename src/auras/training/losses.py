from typing import Sequence


def weighted_loss_config(weights: Sequence[float]) -> dict:
    """Placeholder for weighted cross-entropy configuration."""
    if len(weights) != 2:
        raise ValueError("Expected binary class weights")
    return {"type": "weighted_cross_entropy", "weights": list(weights)}
