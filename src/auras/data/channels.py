from typing import List


CLINICAL_TO_WEARABLE_PRIORITY = [
    "T7",
    "T8",
    "F7",
    "F8",
    "P7",
    "P8",
]


def select_channels(available: List[str], max_channels: int = 6) -> List[str]:
    """Pick a stable subset approximating wearable placements."""
    selected = [ch for ch in CLINICAL_TO_WEARABLE_PRIORITY if ch in available]
    return selected[:max_channels]
