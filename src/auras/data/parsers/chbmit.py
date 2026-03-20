"""CHB-MIT seizure annotation parser.

Parses ``*-summary.txt`` files from the CHB-MIT Scalp EEG database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# TODO: Implement when CHB-MIT dataset is integrated.
#       Expected format: chbXX-summary.txt with File Name / Seizure Start / End.


@dataclass
class SeizureInterval:
    recording: str
    onset_sec: float
    offset_sec: float


def parse_summary_file(path: Path) -> List[SeizureInterval]:
    """Parse a CHB-MIT summary file. Placeholder."""
    raise NotImplementedError("CHB-MIT parser not yet implemented")


def load_all_seizures(dataset_root: Path) -> Dict[str, List[SeizureInterval]]:
    """Load seizure annotations for all CHB-MIT subjects."""
    raise NotImplementedError("CHB-MIT parser not yet implemented")
