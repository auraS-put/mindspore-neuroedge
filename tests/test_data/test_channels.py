"""Tests for channel selection logic."""

from auras.data.channels import select_channels, CLINICAL_TO_WEARABLE_PRIORITY


def test_select_channels_all_available():
    available = ["Fp1", "F7", "T7", "P7", "O1", "F8", "T8", "P8", "Fp2"]
    selected = select_channels(available, max_channels=4)
    assert selected == ["T7", "T8", "F7", "F8"]


def test_select_channels_partial():
    available = ["T7", "P3", "O1", "F8"]
    selected = select_channels(available, max_channels=4)
    assert selected == ["T7", "F8"]


def test_select_channels_empty():
    selected = select_channels(["Cz", "Pz"], max_channels=4)
    assert selected == []
