"""What the XGBoost comparisons need besides XGBoost (comparisons.rounds, docs/design.md,
"Comparisons"): each label's weights rescaled to equal totals, the round counts, and the choice
of a round count by the selection score, a larger count only when strictly better. Nothing here
runs XGBoost."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pytest

from mule_pattern_learner.comparisons.rounds import (
    MAX_ROUNDS,
    ROUND_STEP,
    check_rounds,
    choose_rounds,
    equal_totals,
)


def test_each_labels_weights_total_half_the_examples() -> None:
    labels = np.array([1, 0, 0, 0])
    np.testing.assert_allclose(
        equal_totals(labels, np.array([2.0, 1.0, 1.0, 2.0])), [2, 0.5, 0.5, 1]
    )
    with pytest.raises(ValueError, match="no label 1"):
        equal_totals(np.zeros(3, dtype=np.int64), np.ones(3))
    with pytest.raises(ValueError, match="no label 0"):
        equal_totals(np.ones(3, dtype=np.int64), np.ones(3))


def test_a_round_count_is_a_multiple_of_the_step_up_to_the_trees_trained() -> None:
    assert (MAX_ROUNDS, ROUND_STEP) == (1_000, 10)
    check_rounds(10)
    check_rounds(1_000)
    check_rounds(50, trained=50)
    for rounds, trained in ((0, 1_000), (25, 1_000), (1_010, 1_000), (60, 50), (-10, 1_000)):
        with pytest.raises(ValueError, match="multiple of 10"):
            check_rounds(rounds, trained)


def _keys() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_ms": np.full(100, 1_000, dtype=np.int64),
            "account_id": [f"a{index:03d}" for index in range(100)],
            "positive": [index == 0 for index in range(100)],
            "ranked": True,
            "weight": 1.0,
        }
    )


def test_the_round_count_is_the_first_with_the_best_score() -> None:
    """The scores of the counts tried rise to their best at 40 rounds and tie it at 60: 40 is
    kept, since a larger count must be strictly better."""
    # The positive's place at each count: 50th, 10th, 3rd, 1st, 2nd, 1st, then last.
    places = [50, 10, 3, 1, 2, 1] + [100] * 94

    def planted() -> Iterator[tuple[int, NDArray[np.float64]]]:
        for step, place in enumerate(places):
            total = -np.arange(100, dtype=np.float64)
            total[0] = -(place - 1) + 0.5
            yield (step + 1) * ROUND_STEP, total

    rounds, found, history = choose_rounds(planted(), _keys())
    assert rounds == 40 and found.score == pytest.approx(1.0)
    assert [count for count, _ in history] == list(range(10, 1_001, 10))
    assert history[5] == (60, pytest.approx(1.0))


def test_a_choice_needs_a_count_tried_and_scores_every_key() -> None:
    with pytest.raises(ValueError, match="No round count was tried"):
        choose_rounds(iter(()), _keys())
    wrong: Any = iter([(10, np.zeros(99))])
    with pytest.raises(ValueError, match="99 logits for 100 selection keys"):
        choose_rounds(wrong, _keys())
