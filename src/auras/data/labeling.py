"""Window labeling functions for seizure detection and prediction.

These functions compute per-window labels from seizure annotations.
They are used by ``scripts/prepare_dataset.py`` and are tested independently.

Two modes
---------
detection
    y = 1 if the window overlaps an ictal interval by at least
    ``overlap_fraction`` of the window duration, else 0.
    (Paper 18 recommendation: overlap_fraction ≥ 0.20)

prediction
    y = 1  → window is in the preictal zone:
              [seizure_onset − sph_sec − sop_sec,  seizure_onset − sph_sec]
    y = 0  → window is interictal (outside all exclusion zones)
    y = -1 → excluded (ictal / SPH zone / post-ictal gap); dropped from dataset.
"""

from __future__ import annotations

from typing import Any, List, Protocol


class _HasAnnotation(Protocol):
    recording: str
    onset_sec: float
    offset_sec: float


def label_detection(
    starts_sec: "np.ndarray",  # type: ignore[name-defined]
    win_sec: float,
    seizures: List[Any],
    edf_stem: str,
    overlap_fraction: float = 0.0,
) -> "np.ndarray":
    """Binary seizure detection labels.

    Parameters
    ----------
    starts_sec       : (N,) float array — window start times in seconds
    win_sec          : window duration in seconds
    seizures         : list of seizure annotation objects with attributes
                       ``recording``, ``onset_sec``, ``offset_sec``
    edf_stem         : EDF file stem to filter relevant seizures
    overlap_fraction : minimum fraction of window that must overlap an ictal
                       interval to assign label 1 (Paper 18: ≥ 0.20)

    Returns
    -------
    (N,) int32 label array — values in {0, 1}
    """
    import numpy as np

    n = len(starts_sec)
    labels = np.zeros(n, dtype=np.int32)
    for i, win_start in enumerate(starts_sec):
        win_end = win_start + win_sec
        for sz in seizures:
            if edf_stem not in sz.recording:
                continue
            overlap_start = max(win_start, sz.onset_sec)
            overlap_end = min(win_end, sz.offset_sec)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap / win_sec >= overlap_fraction:
                labels[i] = 1
                break
    return labels


def label_prediction(
    starts_sec: "np.ndarray",  # type: ignore[name-defined]
    win_sec: float,
    seizures: List[Any],
    edf_stem: str,
    sop_sec: float,
    sph_sec: float,
    postictal_gap_sec: float,
) -> "np.ndarray":
    """Preictal/interictal labels for seizure prediction.

    Preictal zone per seizure:
        ``[onset − sph_sec − sop_sec,  onset − sph_sec]``

    Excluded zones → label **-1**, will be dropped from the dataset:
        * ictal:       ``[onset, offset]``
        * SPH zone:    ``[onset − sph_sec, onset]``
        * post-ictal:  ``[offset, offset + postictal_gap_sec]``

    Parameters
    ----------
    starts_sec         : (N,) float array — window start times in seconds
    win_sec            : window duration in seconds
    seizures           : list of seizure annotations for this recording
    edf_stem           : EDF file stem to filter relevant seizures
    sop_sec            : seizure occurrence period in seconds (e.g. 15×60)
    sph_sec            : seizure prediction horizon in seconds (e.g. 5×60)
    postictal_gap_sec  : post-ictal exclusion gap in seconds (e.g. 15×60)

    Returns
    -------
    (N,) int32 label array — values in {-1 (excluded), 0 (interictal), 1 (preictal)}
    """
    import numpy as np

    n = len(starts_sec)
    labels = np.zeros(n, dtype=np.int32)

    rec_seizures = [sz for sz in seizures if edf_stem in sz.recording]

    for i, win_start in enumerate(starts_sec):
        win_end = win_start + win_sec
        label = 0

        for sz in rec_seizures:
            onset = sz.onset_sec
            offset = sz.offset_sec

            # ---- exclusion zones (checked first; any match → -1, break) ----
            # ictal
            if win_start < offset and win_end > onset:
                label = -1
                break
            # SPH zone: [onset - sph_sec, onset]
            if win_start < onset and win_end > (onset - sph_sec):
                label = -1
                break
            # post-ictal gap
            if win_start < (offset + postictal_gap_sec) and win_end > offset:
                label = -1
                break

            # ---- preictal zone: [onset - sph_sec - sop_sec, onset - sph_sec] ----
            preictal_start = onset - sph_sec - sop_sec
            preictal_end = onset - sph_sec
            if win_start >= preictal_start and win_end <= preictal_end:
                label = 1
                # Don't break: another seizure might exclude this window

        labels[i] = label

    return labels
