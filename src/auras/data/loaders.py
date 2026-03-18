from pathlib import Path
from typing import Dict, Iterable


def discover_records(root: str) -> Iterable[Path]:
    """Return candidate record files under a dataset root."""
    path = Path(root)
    if not path.exists():
        return []
    return sorted(path.rglob("*.edf"))


def build_record_index(dataset_name: str, root: str) -> Dict[str, str]:
    """Create a lightweight index used by future dataset-specific loaders."""
    records = discover_records(root)
    return {f"{dataset_name}_{i:06d}": str(p) for i, p in enumerate(records)}
