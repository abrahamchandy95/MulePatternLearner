"""The answer key: read a page at a time through the evaluation executor, which runs nothing
else, and what a true mule is at a scoring time (docs/design.md, "read_answer_key" and
"Metrics")."""

from __future__ import annotations

from datetime import UTC, datetime
import random

import numpy as np
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.answer_key import (
    ACTIVE,
    AFTER,
    BEFORE,
    EARLY_WARNING,
    NON_MULE,
    WEEK_MS,
    early_warning_at,
    early_warning_rows,
    read_answer_key,
    status,
    statuses,
)
from mule_pattern_learner.testing.calendar import Spy, early_warning_times, positive_times
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import Mule, T
from mule_pattern_learner.tigergraph.executor import QueryNotAllowed
from mule_pattern_learner.tigergraph.queries import READ_ANSWER_KEY, READ_LABELS
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth

D = 86_400_000
# Scoring times of two kinds: a Monday every week around T, and four 7, 9 and 7 days apart, the
# middle gap longer than a week, as a world's times can be.
MONDAYS = tuple(T + week * WEEK_MS for week in range(-3, 4))
UNEVEN = (T - 3 * D, T + 4 * D, T + 13 * D, T + 20 * D)


def test_every_page_is_read_through_the_evaluation_executor(graph: ReferenceGraph) -> None:
    spy = Spy(graph)
    key = read_answer_key(spy, page=10)
    # 36 internal accounts (the 34 scorable, a closed one and the loan ledger), in pages of
    # 10: the fourth comes back short.
    assert len(key.truths) == 36 and spy.names() == [READ_ANSWER_KEY] * 4
    assert [truth.account_id for truth in key.truths] == sorted(key.by_id)
    assert {truth.account_id for truth in key.mules} == {
        "a_mule",
        "a_collector",
        "a_mule2",
        "a_mule3",
        "a_hr1",
        "a_hr2",
        "a_hr3",
        "a_drop",
        "a_late",
    }
    assert key.by_id["a_hr2"].mule_ring_id == 11 and key.by_id["a_drop"].mule_ring_id == -1


def test_the_evaluation_executor_runs_the_answer_key_alone(graph: ReferenceGraph) -> None:
    from mule_pattern_learner.tigergraph.executor import EvaluationExecutor

    with pytest.raises(QueryNotAllowed):
        EvaluationExecutor(Spy(graph)).run(READ_LABELS, {"after_id": "", "page": 10})


def _mule(onset: int, last: int) -> AccountTruth:
    return AccountTruth("a_m", 1, 3, onset, last, 0, "deposit", True)


def test_a_mules_status_at_each_scoring_time_and_its_bounds() -> None:
    mule = _mule(T - D, T + 3 * D)
    # Active from the time after its onset to the time at most 7 days after its last payment.
    assert status(mule, T, MONDAYS) == ACTIVE and status(mule, T + WEEK_MS, MONDAYS) == ACTIVE
    assert status(mule, T + 2 * WEEK_MS, MONDAYS) == AFTER
    # Its early-warning time is the last scoring time at or before its onset.
    assert status(mule, T - WEEK_MS, MONDAYS) == EARLY_WARNING
    assert status(mule, T - 2 * WEEK_MS, MONDAYS) == BEFORE
    # An onset exactly at t is not in view: t is its early-warning time, not the time before.
    at = _mule(T, T + D)
    assert status(at, T, MONDAYS) == EARLY_WARNING and status(at, T + WEEK_MS, MONDAYS) == ACTIVE
    assert status(at, T - WEEK_MS, MONDAYS) == BEFORE
    # A last payment exactly 7 days before t still makes it active.
    edge = _mule(T - 9 * D, T - WEEK_MS)
    assert status(edge, T, MONDAYS) == ACTIVE and status(edge, T + D, MONDAYS) == AFTER
    non_mule = AccountTruth("a_n", 0, -1, 0, 0, 0, "deposit", True)
    assert status(non_mule, T, MONDAYS) == NON_MULE


def test_a_mule_with_no_scoring_time_in_its_window_is_active_at_the_one_that_ends_the_gap() -> None:
    # Its payments, T + 5 days and an hour later, fall in the 9 days between T + 4 days and
    # T + 13 days, so the window of T + 13 days is 9 days: active there, and after its activity at
    # T + 20 days, whose window is 7 days.
    quiet = _mule(T + 5 * D, T + 5 * D + 3_600_000)
    assert [status(quiet, t_ms, UNEVEN) for t_ms in UNEVEN] == [
        BEFORE,
        EARLY_WARNING,
        ACTIVE,
        AFTER,
    ]
    # A payment exactly at T + 4 days counts at T + 13 days too, so it is active at both.
    busy = _mule(T - 1, T + 4 * D)
    assert [status(busy, t_ms, UNEVEN) for t_ms in UNEVEN] == [
        EARLY_WARNING,
        ACTIVE,
        ACTIVE,
        AFTER,
    ]
    # An onset exactly at a time is its early-warning time, not an active one: it is active at
    # the next, T + 13 days.
    at_time = _mule(T + 4 * D, T + 4 * D)
    assert [status(at_time, t_ms, UNEVEN) for t_ms in UNEVEN] == [
        BEFORE,
        EARLY_WARNING,
        ACTIVE,
        AFTER,
    ]
    # An onset 12 days after T, 8 days after the time before it and 1 day before the next, has
    # its early-warning time at T + 4 days: a week before the onset would hold none.
    gap = _mule(T + 12 * D, T + 12 * D + 3_600_000)
    assert [status(gap, t_ms, UNEVEN) for t_ms in UNEVEN] == [
        BEFORE,
        EARLY_WARNING,
        ACTIVE,
        AFTER,
    ]
    # A mule whose onset follows every scoring time is active at none, and its early-warning time
    # is the last scoring time, a week from it.
    late = _mule(T + 21 * D, T + 22 * D)
    assert [status(late, t_ms, UNEVEN) for t_ms in UNEVEN] == [
        BEFORE,
        BEFORE,
        BEFORE,
        EARLY_WARNING,
    ]
    # And beyond that week it has none.
    far = _mule(T + 27 * D, T + 28 * D)
    assert [status(far, t_ms, UNEVEN) for t_ms in UNEVEN] == [BEFORE] * 4


def _draws(rng: random.Random, times: list[int], count: int = 300) -> list[tuple[int, int, int]]:
    """Count mules and times (onset, last, t) of a world with the scoring times given, drawn from
    times where the rule ties or flips: each scoring time, a week from it, and a millisecond
    either side, beside some of no pattern."""
    near = [t_ms + shift for t_ms in times for shift in (0, WEEK_MS, -WEEK_MS)]
    pool = [t_ms + step for t_ms in near for step in (-1, 0, 1)]
    pool += [rng.randrange(times[0] - 30 * D, times[-1] + 30 * D) for _ in range(20)]
    draws: list[tuple[int, int, int]] = []
    for _ in range(count):
        onset, last = sorted((rng.choice(pool), rng.choice(pool)))
        draws.append((onset, last, rng.choice(pool)))
    return draws


def _grid(rng: random.Random) -> list[int]:
    """The scoring times of a random world: one to six, a day to two weeks apart, a week apart
    now and then."""
    found = [T + rng.randrange(WEEK_MS)]
    for _ in range(rng.randrange(0, 6)):
        gap = WEEK_MS if rng.random() < 0.25 else rng.randrange(D, 14 * D)
        found.append(found[-1] + gap)
    return found


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()) * 1_000


def test_statuses_agrees_with_status() -> None:
    truths = [
        _mule(T - D, T + 3 * D),
        _mule(T, T + D),
        _mule(T - 9 * D, T - WEEK_MS),
        _mule(T + 5 * D, T + 5 * D + 3_600_000),
        _mule(T - 1, T + 4 * D),
        _mule(T + 21 * D, T + 22 * D),
        AccountTruth("a_n", 0, -1, 0, 0, 0, "deposit", True),
    ]
    grids = (MONDAYS, UNEVEN, DEFAULT_CONFIG.calendar.scoring_times(1001), ())
    for grid in grids:
        times = [T + week * WEEK_MS for week in range(-2, 3)] + list(grid[:10])
        pairs = [(truth, t_ms) for truth in truths for t_ms in times]
        found = statuses(
            np.array([t_ms for _, t_ms in pairs], dtype=np.int64),
            np.array([truth.is_mule for truth, _ in pairs], dtype=np.int64),
            np.array([truth.mule_onset_ms for truth, _ in pairs], dtype=np.int64),
            np.array([truth.mule_last_ms for truth, _ in pairs], dtype=np.int64),
            grid,
        )
        assert found.tolist() == [status(truth, t_ms, grid) for truth, t_ms in pairs]


def test_statuses_and_early_warning_rows_agree_with_the_scalar_rule_on_random_draws() -> None:
    rng = random.Random(11)
    for _ in range(150):
        times = _grid(rng) if rng.random() < 0.95 else []
        draws = _draws(rng, times or [T])
        onset = np.array([o for o, _, _ in draws], dtype=np.int64)
        last = np.array([x for _, x, _ in draws], dtype=np.int64)
        at = np.array([t for _, _, t in draws], dtype=np.int64)
        mule = np.ones(len(draws), dtype=np.int64)
        mule[::7] = 0
        found = statuses(at, mule, onset, last, times)
        truths = [
            AccountTruth("a_m", int(m), 3, int(o), int(x), 0, "deposit", True)
            for m, o, x in zip(mule, onset, last, strict=True)
        ]
        assert found.tolist() == [
            status(truth, int(t), times) for truth, t in zip(truths, at, strict=True)
        ]
        assert early_warning_rows(onset, at, times).tolist() == [
            early_warning_at(int(o), int(t), times) for o, t in zip(onset, at, strict=True)
        ]


def test_on_a_grid_of_mondays_the_statuses_are_the_old_monday_rule() -> None:
    # Scoring times 7 days apart: active when onset < t <= last + 7 days, at its early warning
    # when onset - 7 days < t <= onset, before when t <= onset, after otherwise, over many draws.
    rng = random.Random(12)
    for grid in (DEFAULT_CONFIG.calendar.week_starts(), MONDAYS):
        times = list(grid)
        for onset, last, _ in _draws(rng, times, 1_000):
            truth = _mule(onset, last)
            for t_ms in times:
                if onset < t_ms <= last + WEEK_MS:
                    old = ACTIVE
                elif onset - WEEK_MS < t_ms <= onset:
                    old = EARLY_WARNING
                else:
                    old = BEFORE if t_ms <= onset else AFTER
                assert status(truth, t_ms, times) == old


def test_a_mule_has_one_early_warning_time_when_its_onset_is_inside_the_scored_year() -> None:
    # Exactly one when T[0] <= onset < T[-1] + 7 days; none before the first scoring time or from
    # a week after the last. An onset exactly at a scoring time makes that time its early
    # warning, and the one before it is not.
    rng = random.Random(13)
    for seed in (1001, 1002):
        times = DEFAULT_CONFIG.calendar.scoring_times(seed)
        for onset, _, _ in _draws(rng, list(times), 3_000):
            found = early_warning_times(Mule(onset, onset), times)
            statuses_at = [status(_mule(onset, onset), t_ms, times) for t_ms in times]
            assert [
                t_ms
                for t_ms, state in zip(times, statuses_at, strict=True)
                if state == EARLY_WARNING
            ] == list(found)
            assert len(found) == (1 if times[0] <= onset < times[-1] + WEEK_MS else 0)
        for k, t_ms in enumerate(times):
            assert early_warning_times(Mule(t_ms, t_ms), times) == (t_ms,)
            assert status(_mule(t_ms, t_ms), t_ms, times) == EARLY_WARNING
            if k:
                assert status(_mule(t_ms, t_ms), times[k - 1], times) == BEFORE
        assert early_warning_times(Mule(times[0] - 1, times[0] - 1), times) == ()
        last = times[-1]
        assert early_warning_times(Mule(last + WEEK_MS - 1, last + WEEK_MS), times) == (last,)
        assert early_warning_times(Mule(last + WEEK_MS, last + WEEK_MS), times) == ()


def test_the_early_warning_of_a_mule_between_two_scoring_times_far_apart() -> None:
    # Seed 1001 has scoring times on Monday 1 July 09:50 and Saturday 13 July 22:22. A mule with
    # onset Tuesday 9 July 09:50 is between them; a week before the onset holds none of its
    # times, and the last scoring time at or before it, 1 July, is its early-warning time.
    times = DEFAULT_CONFIG.calendar.scoring_times(1001)
    july = _mule(_utc(2024, 7, 9, 9, 50), _utc(2024, 7, 10, 9, 50))
    assert times[13] < july.mule_onset_ms - WEEK_MS and july.mule_onset_ms < times[14]
    assert [t_ms for t_ms in times if status(july, t_ms, times) == EARLY_WARNING] == [times[13]]
    assert [t_ms for t_ms in times if status(july, t_ms, times) == ACTIVE] == [times[14]]


def test_a_mules_statuses_follow_activity_then_after_it_along_the_scoring_times() -> None:
    # Along the real scoring times, in order: before, at most one early-warning time, active
    # times, after, and no after time before an active one.
    rng = random.Random(14)
    for seed in (1001, 1002):
        times = DEFAULT_CONFIG.calendar.scoring_times(seed)
        for onset, last, _ in _draws(rng, list(times), 3_000):
            truth = _mule(onset, last)
            found = [status(truth, t_ms, times) for t_ms in times]
            order = [(BEFORE, EARLY_WARNING, ACTIVE, AFTER).index(state) for state in found]
            assert order == sorted(order)


def test_the_status_follows_the_rule_read_literally_in_every_world_of_the_calendar() -> None:
    mules = [Mule(T - D, T + 24 * WEEK_MS), Mule(T + 8 * WEEK_MS + 2 * D, T + 8 * WEEK_MS + 3 * D)]
    mules.append(Mule(T + 30 * WEEK_MS, T + 30 * WEEK_MS))
    mules.append(Mule(1_706_745_600_000, 1_707_350_400_000))  # February 2024, before the year
    rng = random.Random(15)
    for seed in (1001, 1002, 1003, 2001, 3001):
        times = DEFAULT_CONFIG.calendar.scoring_times(seed)
        random_mules = [Mule(o, x) for o, x, _ in _draws(rng, list(times))]
        for mule in [*mules, *random_mules]:
            truth = _mule(mule.onset_ms, mule.last_ms)
            found = tuple(t_ms for t_ms in times if status(truth, t_ms, times) == ACTIVE)
            assert found == positive_times(mule, times)
            warned = tuple(t_ms for t_ms in times if status(truth, t_ms, times) == EARLY_WARNING)
            assert warned == early_warning_times(mule, times)
            # After activity is every time after the onset at which it is not active.
            after = [t_ms for t_ms in times if status(truth, t_ms, times) == AFTER]
            assert after == [t_ms for t_ms in times if t_ms > mule.onset_ms and t_ms not in found]
        # The calendar world's mules: February's has no active time, the rest at least one.
        assert [bool(positive_times(mule, times)) for mule in mules] == [True, True, True, False]
