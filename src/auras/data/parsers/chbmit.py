"""CHB-MIT Scalp EEG seizure annotation parser.

Parses ``chbXX-summary.txt`` files from the `CHB-MIT Scalp EEG Database
<https://physionet.org/content/chbmit/1.0.0/>`_.

Summary file format (per subject directory)::

    File Name: chb01_01.edf
    File Start Time: 11:42:54
    File End Time: 12:42:54
    Number of Seizures in File: 0

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

    File Name: chb02_16+.edf
    ...
    Number of Seizures in File: 4
    Seizure 1 Start Time: 130 seconds
    Seizure 1 End Time: 212 seconds
    Seizure 2 Start Time: 2972 seconds
    Seizure 2 End Time: 3053 seconds

Usage::

    from auras.data.parsers.chbmit import load_all_seizures
    seizures = load_all_seizures(Path("data/raw/chbmit"))
    # {  "chb01": [SeizureInterval("chb01_03.edf", 2996, 3036)], ... }
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SeizureInterval:
    """A single seizure annotation within a specific EDF recording.

    Attributes
    ----------
    recording  : EDF file stem (e.g. ``"chb01_03"`` without ``.edf``)
    onset_sec  : seizure onset in **seconds from EDF file start**
    offset_sec : seizure offset in **seconds from EDF file start**
    file_start : EDF recording start as ``(HH, MM, SS)`` tuple, or None if unknown
    file_end   : EDF recording end as ``(HH, MM, SS)`` tuple, or None if unknown
    """

    recording: str
    onset_sec: float
    offset_sec: float
    file_start: Optional[Tuple[int, int, int]] = None
    file_end: Optional[Tuple[int, int, int]] = None

    @property
    def duration_sec(self) -> float:
        return self.offset_sec - self.onset_sec


def _parse_hhmmss(time_str: str) -> Tuple[int, int, int]:
    """Parse ``HH:MM:SS`` into an (h, m, s) tuple."""
    parts = re.split(r"[:.]", time_str.strip())
    return int(parts[0]), int(parts[1]), int(parts[2])


def _hhmmss_to_seconds(h: int, m: int, s: int) -> float:
    return h * 3600 + m * 60 + s


def parse_summary_file(path: Path) -> List[SeizureInterval]:
    """Parse a single ``chbXX-summary.txt`` annotation file.

    Handles all known edge-cases in the CHB-MIT corpus:
    * Multiple seizures per file (``Seizure N Start Time: …``)
    * Single seizure per file (``Seizure Start Time: …``)
    * Files with no seizures (``Number of Seizures in File: 0``)
    * Malformed or missing time fields

    Parameters
    ----------
    path : path to the ``*-summary.txt`` file

    Returns
    -------
    List of :class:`SeizureInterval`, sorted by (recording, onset).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    intervals: List[SeizureInterval] = []

    # ---- track current file block ----------------------------------------
    current_rec: Optional[str] = None
    current_start: Optional[Tuple[int, int, int]] = None
    current_end: Optional[Tuple[int, int, int]] = None
    n_seizures: int = 0
    current_seizure_onset: Optional[float] = None  # for single-seizure files

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # --- File Name ---
        m = re.match(r"File\s+Name\s*:\s*(.+\.edf)", line, re.IGNORECASE)
        if m:
            current_rec = Path(m.group(1).strip()).stem  # strip .edf
            current_start = None
            current_end = None
            n_seizures = 0
            current_seizure_onset = None
            continue

        # --- File Start Time ---
        m = re.match(r"File\s+Start\s+Time\s*:\s*(\S+)", line, re.IGNORECASE)
        if m:
            try:
                current_start = _parse_hhmmss(m.group(1))
            except (ValueError, IndexError):
                pass
            continue

        # --- File End Time ---
        m = re.match(r"File\s+End\s+Time\s*:\s*(\S+)", line, re.IGNORECASE)
        if m:
            try:
                current_end = _parse_hhmmss(m.group(1))
            except (ValueError, IndexError):
                pass
            continue

        # --- Number of Seizures ---
        m = re.match(r"Number\s+of\s+Seizures\s+in\s+File\s*:\s*(\d+)", line, re.IGNORECASE)
        if m:
            n_seizures = int(m.group(1))
            continue

        if n_seizures == 0 or current_rec is None:
            continue

        # --- Seizure {N} Start Time ---  (single or numbered)
        m_start = re.match(
            r"Seizure(?:\s+\d+)?\s+Start\s+Time\s*:\s*(\d+)\s*(?:seconds)?",
            line, re.IGNORECASE
        )
        if m_start:
            current_seizure_onset = float(m_start.group(1))
            continue

        # --- Seizure {N} End Time ---
        m_end = re.match(
            r"Seizure(?:\s+\d+)?\s+End\s+Time\s*:\s*(\d+)\s*(?:seconds)?",
            line, re.IGNORECASE
        )
        if m_end and current_seizure_onset is not None:
            offset = float(m_end.group(1))
            intervals.append(SeizureInterval(
                recording=current_rec,
                onset_sec=current_seizure_onset,
                offset_sec=offset,
                file_start=current_start,
                file_end=current_end,
            ))
            current_seizure_onset = None
            continue

    return sorted(intervals, key=lambda s: (s.recording, s.onset_sec))


def load_all_seizures(dataset_root: Path) -> Dict[str, List[SeizureInterval]]:
    """Load seizure annotations for all CHB-MIT subjects under *dataset_root*.

    Scans for ``chbXX-summary.txt`` files (one per subject directory).

    Parameters
    ----------
    dataset_root : root of the CHB-MIT dataset
                   (should contain ``chb01/``, ``chb02/``, … sub-directories)

    Returns
    -------
    dict mapping subject_id → list of :class:`SeizureInterval`
    e.g. ``{"chb01": [...], "chb02": [...], ...}``
    """
    dataset_root = Path(dataset_root)
    result: Dict[str, List[SeizureInterval]] = {}

    summary_files = sorted(dataset_root.rglob("*-summary.txt"))
    if not summary_files:
        raise FileNotFoundError(
            f"No CHB-MIT summary files found under {dataset_root}. "
            "Expected files named 'chbXX-summary.txt'."
        )

    for sf in summary_files:
        # subject ID = parent directory name (chb01, chb02, …)
        subject_id = sf.parent.name
        result[subject_id] = parse_summary_file(sf)

    return result


def subject_seizure_count(dataset_root: Path) -> Dict[str, int]:
    """Return a dict mapping subject_id → total seizure count."""
    return {sid: len(szs) for sid, szs in load_all_seizures(dataset_root).items()}

