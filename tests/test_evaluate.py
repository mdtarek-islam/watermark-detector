import pytest

from watermark_detector.evaluate import binary_metrics


def test_metrics_from_predictions():
    m = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.7, 0.1])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["n"] == 4


def test_no_positive_predictions_gives_zero_precision():
    m = binary_metrics([1, 0], [0.1, 0.2])
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_roc_auc_perfect_and_tied():
    from watermark_detector.evaluate import roc_auc

    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.2, 0.8]) == pytest.approx(0.75)
