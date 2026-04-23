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

    Siena annotation files store **wall-clock** times (HH.MM.SS).  The EDF
    file, however, starts at time 0.  We therefore convert each seizure onset/
    offset to *seconds relative to the recording start* using the
    ``Registration start time`` field, handling midnight roll-over:

        relative = wall_clock_onset - registration_start
        if relative < 0:          # seizure is after midnight
            relative += 86400
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    intervals: List[SeizureInterval] = []

    current_recording = None
    reg_start: float | None = None   # wall-clock seconds of registration start
    onset_wall: float | None = None  # wall-clock seconds of seizure onset

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match "File name: PN00-1.edf"
        fname_match = re.match(r"File\s+name:\s*(.+)", line, re.IGNORECASE)
        if fname_match:
            raw_name = fname_match.group(1).strip()
            # Fix known Siena annotation typo: "PNO6" (letter O) → "PN06" (zero)
            raw_name = re.sub(r"\bPNO(\d)", r"PN0\1", raw_name)
            current_recording = raw_name
            reg_start = None      # reset per recording block
            onset_wall = None
            continue

        # Match "Registration start time: HH.MM.SS"
        reg_match = re.match(r"Registration\s+start\s+time:\s*(\S+)", line, re.IGNORECASE)
        if reg_match:
            try:
                reg_start = _time_to_seconds(reg_match.group(1))
            except (ValueError, IndexError):
                reg_start = None
            continue

        # Match "Seizure start time: HH.MM.SS" (or "Start time: …")
        onset_match = re.match(
            r"(?:Seizure\s+)?[Ss]tart\s+time:\s*(\S+)", line, re.IGNORECASE
        )
        if onset_match:
            try:
                onset_wall = _time_to_seconds(onset_match.group(1))
            except (ValueError, IndexError):
                onset_wall = None
            continue

        # Match "Seizure end time: HH.MM.SS" (or "End time: …")
        offset_match = re.match(
            r"(?:Seizure\s+)?[Ee]nd\s+time:\s*(\S+)", line, re.IGNORECASE
        )
        if offset_match and onset_wall is not None:
            try:
                offset_wall = _time_to_seconds(offset_match.group(1))
            except (ValueError, IndexError):
                onset_wall = None
                continue

            # Convert wall-clock → relative-to-recording-start
            if reg_start is not None:
                onset_rel = onset_wall - reg_start
                if onset_rel < 0:       # seizure crosses midnight
                    onset_rel += 86400
                offset_rel = offset_wall - reg_start
                if offset_rel < onset_rel:  # offset also crosses midnight
                    offset_rel += 86400
            else:
                # Fallback: no registration start found — treat as relative
                onset_rel = onset_wall
                offset_rel = offset_wall

            intervals.append(
                SeizureInterval(
                    recording=current_recording or path.stem,
                    onset_sec=onset_rel,
                    offset_sec=offset_rel,
                )
            )
            onset_wall = None
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
