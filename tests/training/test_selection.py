"""The selection score on hand-made rankings, worked out by hand (docs/design.md, "Training and
selection"): pooled recall in the top 1% and 5% of each Monday, ties shared, identified mules'
inactive Mondays out of the ranking, the fallback's weights, and the paired interval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.training.selection import (
    BUDGETS,
    Found,
    found,
    paired_interval,
    ranked_keys,
    selection_score,
)

M1, M2 = 1_000, 2_000


def _keys(
    t_ms: int, n: int, positives: dict[int, float] | None = None, weight: float = 1.0
) -> pd.DataFrame:
    """n accounts of the Monday t, a000 to a<n - 1>, positive at the rows positives names,
    each weighing weight."""
    return pd.DataFrame(
        {
            "t_ms": np.full(n, t_ms, dtype=np.int64),
            "account_id": [f"a{index:03d}" for index in range(n)],
            "positive": [index in (positives or {}) for index in range(n)],
            "ranked": [True] * n,
            "weight": np.full(n, weight),
        }
    )


def test_the_budgets_are_the_top_1_and_5_percent() -> None:
    assert BUDGETS == (0.01, 0.05)


def _ranked(n: int, places: dict[int, int]) -> np.ndarray:
    """Scores of n accounts by row: the rows of places at their place (1 the highest), and the
    other rows in the remaining places in row order."""
    free = iter(sorted(set(range(1, n + 1)) - set(places.values())))
    return np.array(
        [float(n - places[row] if row in places else n - next(free)) for row in range(n)]
    )


def test_pooled_recall_over_mondays_and_the_score_is_their_mean() -> None:
    """Two Mondays of 200 accounts: the top 1% is 2 accounts, the top 5% 10. On the first, the
    positives rank 1st, 5th and 20th; on the second, the one positive ranks 3rd. Pooled over 4
    positives: 1 found in the top 1% and 3 in the top 5%."""
    keys = ranked_keys(pd.concat([_keys(M2, 200, {7: 1}), _keys(M1, 200, {0: 1, 1: 1, 2: 1})]))
    scores = np.concatenate([_ranked(200, {0: 1, 1: 5, 2: 20}), _ranked(200, {7: 3})])
    result = found(keys, scores)
    assert result.accounts == ("a000", "a001", "a002", "a007")
    np.testing.assert_array_equal(result.shares, [[1, 1], [0, 1], [0, 0], [0, 1]])
    np.testing.assert_allclose(result.recall(), [1 / 4, 3 / 4])
    assert result.score == pytest.approx(0.5)
    assert selection_score(keys, scores) == result.score


def test_a_monday_without_a_positive_counts_nothing() -> None:
    keys = ranked_keys(pd.concat([_keys(M1, 100, {0: 1}), _keys(M2, 100)]))
    scores = np.where(keys["positive"], 10.0, 0.0) - np.arange(200) * 1e-3
    assert found(keys, scores).score == pytest.approx(1.0)


def test_a_tie_that_straddles_a_budget_finds_a_share_of_each_tied_account() -> None:
    """100 accounts: the top 1% is 1 account and the top 5% is 5. A positive ties with three
    others for places 1 to 4, so the top 1% finds a quarter of it and the top 5% all of it; a
    second positive ties with three others for places 5 to 8, so the top 5% finds a quarter of
    it and the top 1% none."""
    keys = ranked_keys(_keys(M1, 100, {0: 1, 4: 1}))
    scores = -np.arange(100, dtype=np.float64)
    scores[:4] = 50.0
    scores[4:8] = 40.0
    result = found(keys, scores)
    np.testing.assert_allclose(result.shares, [[0.25, 1.0], [0.0, 0.25]])


def test_logits_are_ranked_in_float64() -> None:
    """Two logits that a float32 sigmoid would tie still rank apart."""
    keys = ranked_keys(_keys(M1, 100, {1: 1}))
    scores = np.full(100, -50.0)
    scores[0], scores[1] = 30.0, 30.000001
    assert found(keys, scores).shares[0, 0] == 1.0


def test_inactive_identified_mules_are_left_out_of_the_ranking() -> None:
    """101 accounts with an identified mule on an inactive Monday: left out, the 100 others rank
    and the positive is 1st of them, inside the top 1% of 1 account."""
    keys = _keys(M1, 101, {1: 1})
    keys.loc[0, "ranked"] = False
    held = ranked_keys(keys)
    assert len(held) == 100 and "a000" not in set(held["account_id"])
    scores = -np.arange(100, dtype=np.float64)
    assert found(held, scores).score == pytest.approx(1.0)
    keys.loc[1, "ranked"] = False
    with pytest.raises(ValueError, match="positive"):
        ranked_keys(keys)


def test_the_fallback_weights_set_the_budget() -> None:
    """The positive and 20 sampled accounts weighing 10 each: 201 accounts stood for, so the top
    1% is 2.01 and the top 5% 10.05. Ranked 2nd, behind a sampled account standing for 10, the
    positive lies past the top 1% and 0.05 of it inside the top 5%; ranked 1st, inside both."""
    keys = _keys(M1, 21, {20: 1}, weight=10.0)
    keys.loc[20, "weight"] = 1.0
    keys = ranked_keys(keys)
    scores = -np.arange(21, dtype=np.float64)
    scores[20] = -0.5
    np.testing.assert_allclose(found(keys, scores).shares, [[0.0, 0.05]])
    scores[20] = 1.0
    assert found(keys, scores).shares.tolist() == [[1.0, 1.0]]


def test_found_refuses_a_world_without_positives_and_bad_logits() -> None:
    keys = ranked_keys(_keys(M1, 10))
    with pytest.raises(ValueError, match="no positive"):
        found(keys, np.zeros(10))
    keys = ranked_keys(_keys(M1, 10, {0: 1}))
    with pytest.raises(ValueError, match="9 logits for 10"):
        found(keys, np.zeros(9))
    with pytest.raises(ValueError, match="not finite"):
        found(keys, np.full(10, np.nan))
    with pytest.raises(ValueError, match="lack the columns"):
        ranked_keys(keys.drop(columns="ranked"))


def _run(shares: Sequence[Sequence[float]], accounts: tuple[str, ...]) -> Found:
    return Found(accounts, np.ones(len(accounts)), np.array(shares, dtype=np.float64))


def test_the_paired_interval_of_equal_runs_is_zero() -> None:
    accounts = ("m1", "m1", "m2", "m3")
    run = _run([[1, 1], [0, 1], [0, 0], [1, 1]], accounts)
    assert paired_interval([run], [run]) == (0.0, 0.0)


def test_the_paired_interval_resamples_mule_accounts() -> None:
    """One run finds every positive, the other none: every replicate's difference is 1, since
    each replicate draws mule accounts, never positives one by one."""
    accounts = tuple(f"m{index}" for index in range(30))
    found_all = _run([[1, 1]] * 30, accounts)
    found_none = _run([[0, 0]] * 30, accounts)
    assert paired_interval([found_all], [found_none]) == (1.0, 1.0)
    # Runs that differ on half the mules: an interval around 0.5 that excludes zero.
    half = _run([[1, 1]] * 15 + [[0, 0]] * 15, accounts)
    low, high = paired_interval([half], [found_none])
    assert 0.2 < low < 0.5 < high < 0.8
    # Seeds average within each side.
    low, high = paired_interval([found_all, found_none], [half])
    assert low <= 0 <= high
    with pytest.raises(ValueError, match="different positives"):
        paired_interval([found_all], [_run([[1, 1]] * 29, accounts[:29])])
