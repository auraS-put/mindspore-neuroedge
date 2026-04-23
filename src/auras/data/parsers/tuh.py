"""TUH Seizure Corpus annotation parser.

The TUH EEG Seizure Corpus uses ``.tse`` (time-series event) files
alongside ``.edf`` recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# TODO: Implement when TUH dataset access is obtained.
#       Expected format: .tse files with onset offset label probability


@dataclass
class SeizureInterval:
    recording: str
    onset_sec: float
    offset_sec: float


def parse_tse_file(path: Path) -> List[SeizureInterval]:
    """Parse a single TUH .tse annotation file. Placeholder."""
    raise NotImplementedError("TUH parser not yet implemented")


def load_all_seizures(dataset_root: Path) -> Dict[str, List[SeizureInterval]]:
    """Load seizure annotations for all TUH subjects."""
    raise NotImplementedError("TUH parser not yet implemented")
