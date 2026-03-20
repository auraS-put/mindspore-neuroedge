"""Siena Scalp EEG seizure annotation parser.

Reads ``Seizures-list-PNXX.txt`` files and extracts seizure onset/offset
times per recording, returning them as intervals in seconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SeizureInterval:
    """A single seizure event within a recording."""

    recording: str
    onset_sec: float
    offset_sec: float


def _time_to_seconds(time_str: str) -> float:
    """Convert HH.MM.SS or HH:MM:SS to total seconds."""
    parts = re.split(r"[.:]", time_str.strip())
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(time_str)


def parse_seizure_file(path: Path) -> List[SeizureInterval]:
    """Parse a single ``Seizures-list-PNXX.txt`` file.

    Returns a list of :class:`SeizureInterval` objects.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    intervals: List[SeizureInterval] = []

    current_recording = None
    onset = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match "File name: PN00-1.edf"
        fname_match = re.match(r"File\s+name:\s*(.+)", line, re.IGNORECASE)
        if fname_match:
            current_recording = fname_match.group(1).strip()
            continue

        # Match "Registration start time: HH.MM.SS"
        # (we don't need the absolute time, seizure times are relative)

        # Match "Seizure start time: HH.MM.SS" or "Start time: ..."
        onset_match = re.match(
            r"(?:Seizure\s+)?[Ss]tart\s+time:\s*(\S+)", line, re.IGNORECASE
        )
        if onset_match:
            onset = _time_to_seconds(onset_match.group(1))
            continue

        # Match "Seizure end time: HH.MM.SS" or "End time: ..."
        offset_match = re.match(
            r"(?:Seizure\s+)?[Ee]nd\s+time:\s*(\S+)", line, re.IGNORECASE
        )
        if offset_match and onset is not None:
            offset = _time_to_seconds(offset_match.group(1))
            intervals.append(
                SeizureInterval(
                    recording=current_recording or path.stem,
                    onset_sec=onset,
                    offset_sec=offset,
                )
            )
            onset = None
            continue

    return intervals


def load_all_seizures(dataset_root: Path) -> Dict[str, List[SeizureInterval]]:
    """Load seizure annotations for all subjects under *dataset_root*.

    Returns:
        dict mapping ``subject_id`` → list of :class:`SeizureInterval`.
    """
    seizure_files = sorted(dataset_root.rglob("Seizures-list-*.txt"))
    result: Dict[str, List[SeizureInterval]] = {}
    for sf in seizure_files:
        subject_id = sf.parent.name  # e.g. "PN00"
        result[subject_id] = parse_seizure_file(sf)
    return result
