"""The validation recall that chooses the epoch and the prior, worked out by hand."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from mule_pattern_learner.training.selection import Recall, validation_recall


def _date(
    day: int, labels: list[int], weights: list[float], scores: list[float]
) -> list[Sequence[float]]:
    return [[day] * len(labels), labels, weights, scores]


def _recall(*dates: list[Sequence[float]]) -> Recall:
    as_of, label, weight, score = (np.concatenate(column) for column in zip(*dates, strict=True))
    return validation_recall(as_of.astype(np.int64), label.astype(np.int64), weight, score)


def test_each_dates_recall_in_the_top_1_and_5_percent_is_averaged_over_dates() -> None:
    # Date 1: the mule scores highest of 100 accounts, so it is in both budgets.
    first = _date(1, [1] + [0] * 99, [1.0] * 100, [0.9] + [0.1] * 99)
    # Date 2: one sampled account standing for 3 scores above the mule and 9 standing for
    # 96 below it: the top 1 account holds no mule, the top 5 hold it.
    second = _date(2, [0, 1] + [0] * 9, [3.0, 1.0] + [96 / 9] * 9, [0.9, 0.8] + [0.1] * 9)
    # Date 3 has no revealed mule, so it has no recall and is left out.
    third = _date(3, [0, 0], [50.0, 50.0], [0.2, 0.1])
    recall = _recall(first, second, third)
    assert (recall.top_1pct, recall.top_5pct) == pytest.approx((0.5, 1.0))
    assert recall.score == pytest.approx(0.75)


def test_a_date_with_several_mules_recalls_the_share_of_them_in_the_budget() -> None:
    date = _date(1, [1, 1] + [0] * 98, [1.0] * 100, [0.9, 0.0] + [0.5] * 98)
    recall = _recall(date)
    assert (recall.top_1pct, recall.top_5pct) == pytest.approx((0.5, 0.5))


def test_validation_without_a_revealed_mule_cannot_choose() -> None:
    with pytest.raises(ValueError, match="No validation date"):
        _recall(_date(1, [0, 0], [1.0, 1.0], [0.5, 0.4]))
