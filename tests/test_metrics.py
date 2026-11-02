"""The weekly rankings of the test world: each week's budgets, rows read against a ranking,
and the chance of a flag in any week and of a first flag.

The small cases are worked by hand in their comments.
"""

from __future__ import annotations

import numpy as np
import pytest

from mule_pattern_learner import metrics

# The weekly rankings of the test world. Week 1: a mule of weight 1 at 0.9, a non-mule of
# weight 2 and a mule of weight 4 tied at 0.5, a non-mule of weight 8 at 0.1 (15 in all).
# Week 2: a non-mule of weight 1 at 3.0 and a mule of weight 1 at 2.0. A score of 5.0 on
# Week 2 would top Week 1, but scores are compared within a week only.
WEEK = np.array([1, 1, 1, 1, 2, 2])
LOGIT = np.array([0.9, 0.5, 0.5, 0.1, 3.0, 2.0])
MULE = np.array([True, False, True, False, False, True])
SAMPLED = np.array([1.0, 2.0, 4.0, 8.0, 1.0, 1.0])
BUDGETS_TESTED = (0.1, 0.5)


def test_each_weeks_budgets_find_shares_of_its_blocks() -> None:
    ranking = metrics.WeekRanking(WEEK, LOGIT)
    shares = ranking.shares(SAMPLED, BUDGETS_TESTED)
    # Week 1's 10% is 1.5 of 15: the first account whole, then half an account of the tied
    # block of 6, a twelfth of each of its accounts. Its 50% is 7.5: the first account and the
    # block whole, then 0.5 of the 8 below, a sixteenth. Week 2's 10% is 0.2 of 2, a fifth
    # of its first account, and its 50% that account whole.
    np.testing.assert_allclose(
        shares, [[1, 1], [1 / 12, 1], [1 / 12, 1], [0, 1 / 16], [0.2, 1], [0, 0]]
    )
    found, positives, sizes = ranking.budget_found(shares, MULE, SAMPLED, BUDGETS_TESTED)
    # Week 1 finds 1 + 4/12 of its mules' 5 at 10% and all 5 at 50%; Week 2 none of 1.
    np.testing.assert_allclose(found, [[4 / 3, 5], [0, 0]])
    np.testing.assert_allclose(sizes, [[1.5, 7.5], [0.2, 1]])
    assert positives.tolist() == [5, 1]
    assert ranking.weeks.tolist() == [1, 2]
    # Each row's place: the weight above its block and half its block, over its week's.
    assert ranking.rank_fraction(SAMPLED).tolist() == pytest.approx(
        [0.5 / 15, 4 / 15, 4 / 15, 11 / 15, 0.25, 0.75]
    )


def test_rows_outside_a_ranking_are_read_against_it() -> None:
    ranking = metrics.WeekRanking(WEEK, LOGIT)
    shares, fractions = ranking.against(
        SAMPLED, np.array([1, 1, 1, 2]), np.array([0.5, 0.7, 0.05, 5.0]), BUDGETS_TESTED
    )
    # A score tied with Week 1's block takes its share; one between the first account and
    # the block lies wholly inside 10%; one under everything is never flagged; on Week 2 a
    # score of 5.0 tops its week.
    np.testing.assert_allclose(shares, [[1 / 12, 1], [1, 1], [0, 0], [1, 1]])
    assert fractions.tolist() == pytest.approx([4 / 15, 1 / 15, 1, 0])
    with pytest.raises(ValueError, match="No ranked row"):
        ranking.against(SAMPLED, np.array([3]), np.array([1.0]), BUDGETS_TESTED)


def test_a_flag_in_any_week_and_the_first_flag() -> None:
    # Two rows of half a chance each are flagged at least once with chance 3/4; a whole share
    # makes a flag certain.
    found = metrics.ever_flagged(
        np.array([0.5, 0.5, 1.0, 0.0]), np.array([0, 0, 1, 1], dtype=np.intp), 2
    )
    assert found.tolist() == pytest.approx([0.75, 1.0])
    # First flagged 7 days after onset with chance 1/2, 14 days after with 1/2 x 1/2: given a
    # flag, (7 x 0.5 + 14 x 0.25) / 0.75 days.
    ever, days = metrics.first_flag(np.array([14.0, 7.0, 21.0]), np.array([0.5, 0.5, 0.0]))
    assert ever == pytest.approx(0.75) and days == pytest.approx(7 / 0.75)
    ever, days = metrics.first_flag(np.array([7.0]), np.array([0.0]))
    assert ever == 0 and np.isnan(days)
