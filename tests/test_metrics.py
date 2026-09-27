"""Pure metrics: the capture curve, weighted review budgets and the thresholded metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mule_pattern_learner.metrics import (
    capture_curve,
    proxy_metrics,
    weighted_metrics,
)


def test_the_capture_curve_has_one_point_per_block_of_tied_scores() -> None:
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.5, 0.5, 0.1])
    reviewed, found = capture_curve(y, score, np.array([1.0, 2.0, 4.0, 8.0]))
    assert reviewed.tolist() == [0.0, 1.0, 7.0, 15.0]
    assert found.tolist() == [0.0, 1.0, 5.0, 5.0]


def test_unit_weights_give_the_unweighted_top_fraction_metrics() -> None:
    rng = np.random.default_rng(3)
    # 1, 5 and 10% of 200 accounts are whole numbers: 2, 10 and 20.
    y = (rng.random(200) < 0.1).astype(np.int64)
    score = rng.random(200)
    weighted = weighted_metrics(y, score, np.ones(200), 0.5)
    unweighted = proxy_metrics(y, score, 0.5)
    for key in ("precision_at_1pct", "recall_at_1pct", "precision_at_5pct", "recall_at_5pct"):
        assert weighted[key] == pytest.approx(unweighted[key])
    for key in ("average_precision", "roc_auc", "precision", "recall", "f1"):
        assert weighted[key] == pytest.approx(unweighted[key])
    # The thresholded metrics are one formula: unit weights give the counts' ratios.
    predicted = score >= 0.5
    tp = int((predicted & (y == 1)).sum())
    assert unweighted["precision"] == tp / predicted.sum() and unweighted["recall"] == tp / y.sum()
    top = np.argsort(-score)[:20]
    assert weighted["precision_at_10pct"] == pytest.approx(y[top].sum() / 20)
    assert weighted["recall_at_10pct"] == pytest.approx(y[top].sum() / y.sum())


def test_the_proxy_budgets_share_tied_scores_whatever_the_row_order() -> None:
    # 20 accounts, 4 mules; the top two tie, one of them a mule. The budgets are 0.2, 1
    # and 2 accounts: the first two lie inside the tied block and take half a mule per
    # account, the third holds the block.
    y = np.array([1, 0, 1, 1, 1] + [0] * 15)
    score = np.array([0.9, 0.9, 0.5, 0.4, 0.3] + [0.1] * 15)
    expected = {
        "precision_at_1pct": 0.1 / 0.2,
        "recall_at_1pct": 0.1 / 4,
        "precision_at_5pct": 0.5 / 1,
        "recall_at_5pct": 0.5 / 4,
        "precision_at_10pct": 1 / 2,
        "recall_at_10pct": 1 / 4,
    }
    for order in (np.arange(20), np.r_[1, 0, 2:20]):
        metrics = proxy_metrics(y[order], score[order], 0.5)
        assert {k: metrics[k] for k in expected} == pytest.approx(expected)
