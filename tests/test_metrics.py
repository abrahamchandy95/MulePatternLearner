"""Pure metrics: the capture curve, weighted review budgets and grouped intervals."""

from __future__ import annotations

import numpy as np
import pytest

from mule_pattern_learner.metrics import (
    bootstrap_interval,
    capture_curve,
    evaluate,
    weighted_metrics,
)


def test_bootstrap_interval_resamples_whole_groups() -> None:
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.4, 0.2, 0.7])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    low, high = bootstrap_interval(y, scores, groups, seed=7, draws=100) or (None, None)
    assert low is not None and high is not None and 0.0 <= low <= high <= 1.0
    # Same seed, same interval; one group or no positive has no interval.
    assert bootstrap_interval(y, scores, groups, seed=7, draws=100) == [low, high]
    assert bootstrap_interval(y, scores, np.zeros(8, dtype=np.int64)) is None
    assert bootstrap_interval(np.zeros(8, dtype=np.int64), scores, groups) is None


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
    unweighted = evaluate(y, score, 0.5)
    for key in ("precision_at_1pct", "recall_at_1pct", "precision_at_5pct", "recall_at_5pct"):
        assert weighted[key] == pytest.approx(unweighted[key])
    for key in ("average_precision", "roc_auc", "precision", "recall", "f1"):
        assert weighted[key] == pytest.approx(unweighted[key])
    top = np.argsort(-score)[:20]
    assert weighted["precision_at_10pct"] == pytest.approx(y[top].sum() / 20)
    assert weighted["recall_at_10pct"] == pytest.approx(y[top].sum() / y.sum())
