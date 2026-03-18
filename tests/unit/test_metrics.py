from auras.training.metrics import f1_score, precision, recall


def test_recall_basic():
    assert recall(8, 2) == 0.8


def test_precision_basic():
    assert precision(8, 2) == 0.8


def test_f1_basic():
    assert round(f1_score(8, 2, 2), 6) == 0.8
