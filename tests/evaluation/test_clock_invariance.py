"""The clock invariance test (docs/design.md, the Clock guard): the audit sample stored with the
start of each row's week, and the test that scores it at both times: it passes when scores do
not depend on the time, fails when they do, fails with fewer mule clusters than the minimum
whatever its interval says, recomputes each true mule's label at the week start, and ranks only
the rows in the same class at both times, counting the others as left out."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.answer_key import AnswerKey
from mule_pattern_learner.evaluation.clock_invariance import (
    CLOCK_INVARIANCE,
    COLUMNS,
    Compared,
    Invariance,
    invariance,
    week_start_keys,
    with_week_starts,
)
from mule_pattern_learner.evaluation.sample import COLUMNS as SAMPLE_COLUMNS
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth

W = 7 * DAY_MS
WEEKS = 20
# Twenty weeks from T, each scored 3 days and a few hours into it.
STARTS = tuple(T + week * W for week in range(WEEKS))
TIMES = tuple(start + 3 * DAY_MS + week * 3_600_000 for week, start in enumerate(STARTS))
AUDIT = replace(DEFAULT_CONFIG.audit, replicates=100)
Score = Callable[[pd.DataFrame], pd.DataFrame]


def _noise(*parts: object) -> float:
    """A number in [0, 1) fixed by its parts."""
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _world(rings: int, others: int, seed: int = 0) -> tuple[AnswerKey, pd.DataFrame]:
    """An answer key and an audit sample of rings of two mules and others non-mules in each of
    the twenty weeks: every mule active from before its week's start to after its scoring time,
    each non-mule weighing 100 (so that a week's top 1% holds about the mules above its first
    non-mule), and every account scorable at the start of its week."""
    truths: list[AccountTruth] = []
    rows: list[tuple[int, str, bool, int, float]] = []
    for week, (start, scoring) in enumerate(zip(STARTS, TIMES, strict=True)):
        for ring in range(rings):
            for member in range(2):
                account = f"m{seed}_{week}_{ring}_{member}"
                truths.append(
                    AccountTruth(
                        account,
                        1,
                        week * 1_000 + ring,
                        start - 10 * DAY_MS,
                        start + 20 * DAY_MS,
                        0,
                        "deposit",
                        True,
                    )
                )
                rows.append((scoring, account, True, week * 1_000 + ring, 1.0))
        for index in range(others):
            account = f"n{seed}_{week}_{index}"
            truths.append(AccountTruth(account, 0, -1, 0, 0, 0, "deposit", True))
            rows.append((scoring, account, False, -1, 100.0))
    sample = pd.DataFrame(rows, columns=list(SAMPLE_COLUMNS[:-1]))
    sample["cluster"] = [
        f"ring:{ring}" if positive else f"account:{account}"
        for account, positive, ring in zip(
            sample["account_id"], sample["positive"], sample["ring"], strict=True
        )
    ]
    return AnswerKey(tuple(truths)), sample.sort_values(["t_ms", "account_id"], ignore_index=True)


def _stored(sample: pd.DataFrame, unscorable: frozenset[str] = frozenset()) -> pd.DataFrame:
    """The sample as mule audit stores it: every account scorable at its week's start but the
    unscorable ones."""
    starts = dict(zip(TIMES, STARTS, strict=True))
    scorable = {start: [a for a in sample["account_id"] if a not in unscorable] for start in STARTS}
    return with_week_starts(sample, starts, scorable)


def _scores(
    separation: float, shift: Callable[[str, int], float] = lambda account, t_ms: 0.0
) -> tuple[Score, list[pd.DataFrame]]:
    """A scorer that gives every mule logit separation more than a non-mule's plus noise of its
    account, and shift(account, t_ms) beside, and the keys it was asked for."""
    asked: list[pd.DataFrame] = []

    def score(keys: pd.DataFrame) -> pd.DataFrame:
        asked.append(keys.copy())
        logits = [
            _noise("account", account) * 0.5
            + (separation if account.startswith("m") else 0.0)
            + shift(account, int(t_ms))
            for t_ms, account in zip(keys["t_ms"], keys["account_id"], strict=True)
        ]
        return keys.loc[:, ["t_ms", "account_id"]].assign(logit=logits)

    return score, asked


def test_the_sample_is_stored_with_the_start_of_each_rows_week() -> None:
    _, sample = _world(2, 3)
    stored = _stored(sample, frozenset({"n0_4_1"}))
    assert tuple(stored.columns) == COLUMNS
    starts = dict(zip(TIMES, STARTS, strict=True))
    assert stored["week_ms"].tolist() == [starts[t_ms] for t_ms in stored["t_ms"]]
    assert stored["at_week_start"].tolist() == [a != "n0_4_1" for a in stored["account_id"]]
    keys = week_start_keys(stored)
    # Each kept account-week at its week's start, by week and account; the unscorable one out.
    assert list(keys.columns) == ["t_ms", "account_id"] and len(keys) == len(sample) - 1
    assert keys["t_ms"].is_monotonic_increasing and set(keys["t_ms"]) == set(STARTS)
    assert "n0_4_1" not in set(keys["account_id"])


def test_a_row_of_an_account_unscorable_at_its_week_start_is_in_the_sample_alone() -> None:
    _, sample = _world(2, 3)
    starts = dict(zip(TIMES, STARTS, strict=True))
    stored = with_week_starts(sample, starts, {start: [] for start in STARTS})
    assert not stored["at_week_start"].any() and week_start_keys(stored).empty


def test_scores_that_do_not_depend_on_the_time_pass() -> None:
    key, sample = _world(10, 20)
    score, asked = _scores(separation=0.3)
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert found.passed and found.recall.difference == 0.0 and found.roc_auc.difference == 0.0
    assert found.recall.interval == (0.0, 0.0)
    assert found.recall.scoring_time == found.recall.week_start > 0.5
    assert found.margin == AUDIT.clock_invariance_margin == 0.05
    # Every mule is active at both times, so every row is compared: 20 weeks of 10 rings of two
    # mules and 20 non-mules.
    assert (found.rows, found.mule_rows) == (800, 400)
    assert (found.clusters, found.mule_clusters) == (WEEKS * 10 + WEEKS * 20, WEEKS * 10)
    assert (found.mule_rows_left_out, found.nonmule_rows_left_out) == (0, 0)
    # The sample was scored twice: at its scoring times, and at the start of each of its weeks.
    scoring, starts = asked
    assert set(scoring["t_ms"]) == set(TIMES) and set(starts["t_ms"]) == set(STARTS)


def test_scores_that_depend_on_the_time_fail() -> None:
    key, sample = _world(10, 20)
    # At the start of its week a mule is scored below every ordinary account: the model reads
    # something that the time of the week gives away.
    start_set = set(STARTS)

    def down(account: str, t_ms: int) -> float:
        return -1.0 if account.startswith("m") and t_ms in start_set else 0.0

    score, _ = _scores(0.3, down)
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert not found.passed
    assert found.recall.difference > 0.5 and found.roc_auc.difference > 0.5
    assert found.recall.interval is not None and found.recall.interval[0] > 0.5
    assert found.recall.week_start < found.recall.scoring_time


def _drift(size: float) -> Callable[[str, int], float]:
    """A shift that moves each account's logit between the two times by up to size, in no
    direction in particular."""
    return lambda account, t_ms: size * (_noise("time", account, t_ms) - 0.5)


def test_a_small_drift_inside_the_margin_passes() -> None:
    key, sample = _world(10, 20)
    score, _ = _scores(0.3, _drift(0.1))
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert found.recall.interval is not None
    assert -found.margin <= found.recall.interval[0] < found.recall.interval[1] <= found.margin
    assert found.recall.difference != 0 and found.passed


def test_a_test_with_too_few_mule_clusters_fails_however_narrow_its_interval() -> None:
    # Scores that do not depend on the time, so that the difference is exactly 0 on every
    # resample and the interval is the point [0, 0], inside the margin. Twenty rings, one a week,
    # are fewer mule clusters than the minimum of 30.
    key, sample = _world(1, 20)
    score, _ = _scores(separation=0.3)
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert AUDIT.clock_min_mule_clusters == 30
    assert found.recall.interval == (0.0, 0.0) and found.mule_clusters == WEEKS == 20
    assert not found.passed
    # The minimum is where the line falls: 20 clusters pass a minimum of 20, not one of 21.
    for least, passes in ((20, True), (21, False)):
        audit = replace(AUDIT, clock_min_mule_clusters=least)
        found = invariance(_stored(sample), key, TIMES, STARTS, score, audit, seed=0)
        assert found.recall.interval == (0.0, 0.0) and found.passed is passes
    # Two rings a week are 40 mule clusters, enough for the default.
    key, sample = _world(2, 20)
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert found.mule_clusters == 2 * WEEKS == 40 and found.passed


def test_a_test_with_too_few_mules_to_bound_the_recall_fails() -> None:
    # A drift, with four mules in two rings of two weeks' samples: the recalls move by whole
    # mules, and the interval is wider than the margin.
    key, sample = _world(1, 20)
    sample = sample[sample["t_ms"].isin(TIMES[:2])]
    score, _ = _scores(0.3, _drift(0.1))
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert (found.mule_rows, found.mule_clusters) == (4, 2)
    assert found.recall.interval is not None
    assert found.recall.interval[0] < -found.margin and not found.passed
    # With no mule, recall is undefined at both times: no interval, and not passed.
    score, _ = _scores(0.3)
    found = invariance(
        _stored(sample[~sample["positive"]]), key, TIMES, STARTS, score, AUDIT, seed=0
    )
    assert found.recall.interval is None and not found.passed
    assert (found.mule_rows, found.mule_clusters, found.clusters) == (0, 0, 40)


def _onset_at(key: AnswerKey, account: str, onset_ms: int) -> AnswerKey:
    """The answer key with the onset of one mule account moved."""
    return AnswerKey(
        tuple(
            replace(truth, mule_onset_ms=onset_ms) if truth.account_id == account else truth
            for truth in key.truths
        )
    )


@pytest.mark.parametrize(
    ("onset_ms", "compared"),
    [
        # Active at both times: its onset is before the start of its week.
        (STARTS[5] - 1, True),
        # At the start of its week it has not started (onset < t is strict): its early-warning
        # time, so it is in neither class there, though it is active at the scoring time.
        (STARTS[5], False),
        (STARTS[5] + DAY_MS, False),
        (TIMES[5] - 1, False),
        # Not active at its scoring time either: the onset is not before it.
        (TIMES[5], False),
    ],
)
def test_a_mule_that_started_after_the_week_start_is_left_out_of_both_rankings(
    onset_ms: int, compared: bool
) -> None:
    key, sample = _world(10, 20)
    late = "m0_5_0_0"
    moved = _onset_at(key, late, onset_ms)

    def down(account: str, t_ms: int) -> float:
        # Scored at its scoring time alone it would rank below every ordinary account.
        return -1.0 if account == late and t_ms == TIMES[5] else 0.0

    score, asked = _scores(0.3, down)
    found = invariance(_stored(sample), moved, TIMES, STARTS, score, AUDIT, seed=0)
    left_out = 0 if compared else 1
    assert (found.rows, found.mule_rows) == (800 - left_out, 400 - left_out)
    assert (found.mule_rows_left_out, found.nonmule_rows_left_out) == (left_out, 0)
    # The other member of its ring keeps the cluster in the sample.
    assert found.mule_clusters == 200
    assert all((late in set(keys["account_id"])) is compared for keys in asked)
    if not compared:
        # The test is the one of a sample without the row.
        rest = sample[sample["account_id"] != late]
        without = invariance(_stored(rest), moved, TIMES, STARTS, score, AUDIT, seed=0)
        assert (found.recall, found.roc_auc) == (without.recall, without.roc_auc)
        assert found.passed and found.recall.difference == 0.0


def test_a_mule_not_active_at_its_scoring_time_is_left_out_though_active_at_the_week_start() -> (
    None
):
    key, sample = _world(10, 20)
    stale = "m0_5_0_0"
    # Its last payment is 6 days before the start of its week: inside the 7 days before the
    # week start, but before the 7 days and the previous scoring time that the window of its
    # scoring time reaches back to, so that it is after its activity there.
    truths = tuple(
        replace(truth, mule_last_ms=STARTS[5] - 6 * DAY_MS) if truth.account_id == stale else truth
        for truth in key.truths
    )
    score, asked = _scores(0.3)
    found = invariance(_stored(sample), AnswerKey(truths), TIMES, STARTS, score, AUDIT, seed=0)
    assert (found.rows, found.mule_rows) == (799, 399)
    assert (found.mule_rows_left_out, found.nonmule_rows_left_out) == (1, 0)
    assert all(stale not in set(keys["account_id"]) for keys in asked)


def test_the_clusters_are_those_of_the_rows_compared() -> None:
    key, sample = _world(10, 20)
    # Both mules of the first ring of week 5 start after its start, and a non-mule of week 7 is
    # not scorable at the start of its week.
    for account in ("m0_5_0_0", "m0_5_0_1"):
        key = _onset_at(key, account, STARTS[5] + DAY_MS)
    gone = frozenset({"n0_7_3"})
    score, _ = _scores(0.3)
    found = invariance(_stored(sample, gone), key, TIMES, STARTS, score, AUDIT, seed=0)
    assert (found.rows, found.mule_rows) == (800 - 3, 400 - 2)
    assert (found.mule_rows_left_out, found.nonmule_rows_left_out) == (2, 1)
    assert (found.clusters, found.mule_clusters) == (600 - 2, 200 - 1)


def test_a_row_not_scorable_at_its_week_start_is_left_out_of_both_times() -> None:
    key, sample = _world(10, 20)
    mules = frozenset(sample.loc[sample["positive"], "account_id"].head(5))
    others = frozenset(sample.loc[~sample["positive"], "account_id"].head(3))
    score, asked = _scores(0.3)
    found = invariance(_stored(sample, mules | others), key, TIMES, STARTS, score, AUDIT, seed=0)
    scoring, starts = asked
    # Neither time is asked for them, and both are asked for the same account-weeks.
    assert len(scoring) == len(starts) == len(sample) - 8
    assert not (mules | others) & (set(scoring["account_id"]) | set(starts["account_id"]))
    assert sorted(scoring["account_id"]) == sorted(starts["account_id"])
    assert (found.rows, found.mule_rows) == (792, 395)
    assert (found.mule_rows_left_out, found.nonmule_rows_left_out) == (5, 3)
    assert found.passed and found.recall.difference == 0.0


def test_the_table_of_the_test_is_one_row() -> None:
    key, sample = _world(10, 20)
    score, _ = _scores(0.3)
    found = invariance(_stored(sample), key, TIMES, STARTS, score, AUDIT, seed=0)
    frame = found.frame()
    assert len(frame) == 1 and CLOCK_INVARIANCE == "clock_invariance"
    assert list(frame.columns) == [
        "recall_at_1pct_scoring_time",
        "recall_at_1pct_week_start",
        "recall_at_1pct_difference",
        "recall_at_1pct_low",
        "recall_at_1pct_high",
        "roc_auc_scoring_time",
        "roc_auc_week_start",
        "roc_auc_difference",
        "roc_auc_low",
        "roc_auc_high",
        "margin",
        "passed",
        "rows",
        "mule_rows",
        "clusters",
        "mule_clusters",
        "mule_rows_left_out",
        "nonmule_rows_left_out",
    ]
    assert frame.iloc[0][list(frame.columns[-6:])].tolist() == [800, 400, 600, 200, 0, 0]
    nothing = Invariance(
        Compared(np.nan, np.nan, np.nan, None),
        Compared(np.nan, np.nan, np.nan, None),
        0.05,
        False,
        0,
        0,
        0,
        0,
        3,
        4,
    )
    row = nothing.frame().iloc[0]
    assert np.isnan(row["recall_at_1pct_low"]) and not row["passed"]
    assert (row["mule_rows_left_out"], row["nonmule_rows_left_out"]) == (3, 4)


def test_a_stored_sample_has_the_columns_the_audit_writes() -> None:
    key, sample = _world(2, 3)
    score, _ = _scores(0.3)
    with pytest.raises(ValueError, match="The stored sample has the columns"):
        invariance(sample, key, TIMES, STARTS, score, AUDIT, seed=0)
