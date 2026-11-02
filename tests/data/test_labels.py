"""The labels: every identified mule of read_labels, their identity, when a mule is active at a
scoring time, and the identified mules never scored at a scoring time of their activity
(docs/design.md, "Labels")."""

from __future__ import annotations

from datetime import UTC, datetime
import random

import numpy as np
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data.labels import (
    active,
    active_at,
    active_mules,
    active_rows,
    label_identity,
    read_identified,
    unscored,
    window_start,
    window_starts,
)
from mule_pattern_learner.testing.calendar import (
    EARLY,
    EDGE,
    GONE,
    JOINED,
    LONG,
    SHORT,
    Spy,
    calendar_world,
    positive_times,
)
from mule_pattern_learner.testing.world import DAY_MS, Mule, T
from mule_pattern_learner.tigergraph.queries.read_labels import IdentifiedMule
from mule_pattern_learner.tigergraph.queries.rows import QueryRefused

W = 7 * DAY_MS
IDENTIFIED = {
    "a_long": LONG,
    "a_short": SHORT,
    "a_edge": EDGE,
    "a_gone": GONE,
    "a_early": EARLY,
}


def test_every_identified_mule_is_read_and_no_other() -> None:
    spy = Spy(calendar_world().graph())
    labels = read_identified(spy, page=2)
    expected = tuple(
        IdentifiedMule(account, mule.onset_ms, mule.last_ms)
        for account, mule in sorted(IDENTIFIED.items())
    )
    # a_hidden is a mule no one identified, so read_labels never prints it.
    assert labels.mules == expected
    assert labels.accounts == frozenset(IDENTIFIED)
    # Five mules in pages of 2: 2, 2 and 1.
    assert spy.names() == ["read_labels"] * 3
    assert read_identified(spy) == labels


def test_the_label_identity_is_the_hash_of_the_rows() -> None:
    mules = [IdentifiedMule("a_1", 10, 20), IdentifiedMule("a_2", 30, 30)]
    found = label_identity(mules)
    assert len(found) == 32 and found == label_identity(list(reversed(mules)))
    assert label_identity([IdentifiedMule("a_1", 10, 21), mules[1]]) != found
    assert label_identity(mules[:1]) != found
    # Golden: blake2b, 16 bytes, of [["a_1",10,20],["a_2",30,30]].
    assert found == "dc34298aea9b0ac7dfbedb645a833490"


def test_a_mule_without_a_valid_activity_period_is_refused() -> None:
    w = calendar_world()
    w.customer("bad", JOINED, mule=Mule(T, T - 1, identified=True))
    with pytest.raises(QueryRefused, match="invalid_activity"):
        read_identified(Spy(w.graph()))


# Scoring times 7, 9 and 7 days apart, as a world's can be, the middle gap longer than a week.
TIMES = (T - 3 * DAY_MS, T + 4 * DAY_MS, T + 13 * DAY_MS, T + 20 * DAY_MS)
SEEDS = (1001, 1002)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()) * 1_000


def _anchors(rng: random.Random, times: list[int]) -> list[int]:
    """Times where the rule's comparisons tie or flip: every scoring time, a week from each, and
    a millisecond either side of those, with some times of no pattern."""
    near = [t_ms + shift for t_ms in times for shift in (0, W, -W)]
    around = [t_ms + step for t_ms in near for step in (-1, 0, 1)]
    return around + [
        rng.randrange(times[0] - 30 * DAY_MS, times[-1] + 30 * DAY_MS) for _ in range(20)
    ]


def _grid(rng: random.Random) -> list[int]:
    """Scoring times of a random world: one to six, a day to two weeks apart, a week apart now
    and then."""
    found = [T + rng.randrange(W)]
    for _ in range(rng.randrange(0, 6)):
        found.append(found[-1] + (W if rng.random() < 0.25 else rng.randrange(DAY_MS, 14 * DAY_MS)))
    return found


def test_a_mule_is_active_from_after_its_onset_to_a_week_after_its_last_payment() -> None:
    mule = IdentifiedMule("a_1", T, T + 3 * DAY_MS)
    # An onset exactly at t is not in view.
    assert not active(mule, T, (T,))
    assert active(mule, T + 1, TIMES) and active(mule, T + W, TIMES)
    # Its last payment plus 7 days exactly at t is active; a millisecond later is not. The
    # previous scoring time, T + 4 days, is 6 days before T + 10 days, so the window stays 7 days.
    assert active(mule, T + 3 * DAY_MS + W, TIMES)
    assert not active(mule, T + 3 * DAY_MS + W + 1, TIMES)
    assert window_start(T + 3 * DAY_MS + W, TIMES) == T + 3 * DAY_MS
    mules = [mule, IdentifiedMule("a_2", T + W, T + W)]
    assert active_mules(mules, T + W, TIMES) == [mule]


def test_the_window_reaches_back_to_the_previous_scoring_time_when_that_is_over_a_week() -> None:
    # No scoring time before t: 7 days. The previous one 9 days back: 9 days. 7 days back: 7
    # days. And t need not be a scoring time.
    assert window_start(TIMES[0], TIMES) == TIMES[0] - W
    assert window_start(TIMES[1], TIMES) == TIMES[1] - W
    assert window_start(TIMES[2], TIMES) == TIMES[1]
    assert window_start(TIMES[3], TIMES) == TIMES[2]
    assert window_start(TIMES[2] + 1, TIMES) == TIMES[2] + 1 - W
    assert window_start(T, TIMES) == T - W and window_start(T, ()) == T - W


def test_a_payment_exactly_at_the_previous_scoring_time_counts_at_t() -> None:
    earlier, later = T, T + 12 * DAY_MS
    times = (earlier, later)
    # A payment exactly at the previous scoring time was not in view there, so it counts at the
    # next one, 12 days on. A millisecond earlier it was in view at the first and does not count
    # at the second.
    at = IdentifiedMule("a_at", earlier - DAY_MS, earlier)
    before = IdentifiedMule("a_before", earlier - DAY_MS, earlier - 1)
    assert active(at, later, times)
    assert active(before, earlier, times) and not active(before, later, times)


def test_a_short_gap_keeps_the_week_and_a_long_gap_stretches_it() -> None:
    first, soon = T, T + DAY_MS
    short = (first, soon)
    # The last payment 5 days before the second time, 4 days before the first: active at both.
    five = IdentifiedMule("a_5", T - 4 * DAY_MS, soon - 5 * DAY_MS)
    assert all(active(five, t_ms, short) for t_ms in short)
    # A gap of 1 day does not shorten the week: a payment a week and a millisecond before the
    # second time is active at the first and not at the second.
    week = IdentifiedMule("a_week", T - 7 * DAY_MS, soon - W - 1)
    assert active(week, first, short) and not active(week, soon, short)
    # A gap of 12 days stretches it: a payment 10 days before the second time counts at the
    # second, as it would not under a week alone, and one of 13 days before it does not.
    late = T + 12 * DAY_MS
    long = (first, late)
    ten = IdentifiedMule("a_10", T - DAY_MS, late - 10 * DAY_MS)
    assert active(ten, late, long) and not active(ten, late, (late,))
    assert not active(IdentifiedMule("a_13", T - DAY_MS, late - 13 * DAY_MS), late, long)


def test_a_mule_active_through_a_gap_is_active_at_the_scoring_time_that_ends_it() -> None:
    # Its payments, T + 5 and T + 5 days + 1 hour, fall between T + 4 days and T + 13 days, which
    # 9 days apart leave a window of 9 days, so it is active at T + 13 days and no later: at
    # T + 20 days the window is 7 days again.
    quiet = IdentifiedMule("a_quiet", T + 5 * DAY_MS, T + 5 * DAY_MS + 3_600_000)
    assert [t_ms for t_ms in TIMES if active(quiet, t_ms, TIMES)] == [T + 13 * DAY_MS]
    # A payment exactly at T + 4 days counts at T + 13 days as well as at T + 4 days.
    busy = IdentifiedMule("a_busy", T - 1, T + 4 * DAY_MS)
    assert [t_ms for t_ms in TIMES if active(busy, t_ms, TIMES)] == [
        T + 4 * DAY_MS,
        T + 13 * DAY_MS,
    ]
    # An onset exactly at a scoring time is not in view there, so its first scoring time is the
    # next one.
    at_time = IdentifiedMule("a_at_time", T + 4 * DAY_MS, T + 4 * DAY_MS)
    assert [t_ms for t_ms in TIMES if active(at_time, t_ms, TIMES)] == [T + 13 * DAY_MS]
    # A last payment exactly 7 days before a scoring time, whose gap is 7 days, reaches it.
    reaches = IdentifiedMule("a_reaches", T + 5 * DAY_MS, T + 13 * DAY_MS)
    assert [t_ms for t_ms in TIMES if active(reaches, t_ms, TIMES)] == [
        T + 13 * DAY_MS,
        T + 20 * DAY_MS,
    ]


def test_a_mule_with_no_scoring_time_after_its_onset_is_active_at_none() -> None:
    late = IdentifiedMule("a_late", T + 21 * DAY_MS, T + 22 * DAY_MS)
    assert not any(active(late, t_ms, TIMES) for t_ms in TIMES)
    # No scoring times at all: the week alone.
    assert active_at(10, 20, 20 + W, ()) and not active_at(10, 20, 21 + W, ())


def test_the_rule_is_the_old_monday_rule_on_a_grid_of_weeks() -> None:
    # On the week starts, Mondays 7 days apart, the window is 7 days: onset < t <= last + 7
    # days, over many draws, with onsets and last payments exactly on the starts and a week from
    # them as well as between.
    rng = random.Random(4)
    for seed_weeks in (
        DEFAULT_CONFIG.calendar.week_starts(),
        tuple(T + k * W for k in range(-2, 9)),
    ):
        grid = list(seed_weeks)
        pool = _anchors(rng, grid)
        for _ in range(3_000):
            onset, last = sorted((rng.choice(pool), rng.choice(pool)))
            for t_ms in grid:
                assert active_at(onset, last, t_ms, grid) == (onset < t_ms <= last + W)


def test_the_scalar_and_the_vectorised_rule_agree_on_random_draws() -> None:
    rng = random.Random(5)
    for _ in range(200):
        times = _grid(rng) if rng.random() < 0.95 else []
        pool = _anchors(rng, times or [T])
        draws = [
            (*sorted((rng.choice(pool), rng.choice(pool))), rng.choice(pool)) for _ in range(60)
        ]
        onset = np.array([o for o, _, _ in draws], dtype=np.int64)
        last = np.array([x for _, x, _ in draws], dtype=np.int64)
        at = np.array([t for _, _, t in draws], dtype=np.int64)
        assert window_starts(at, times).tolist() == [window_start(int(t), times) for t in at]
        assert active_rows(onset, last, at, times).tolist() == [
            active_at(int(o), int(x), int(t), times)
            for o, x, t in zip(onset, last, at, strict=True)
        ]


def test_the_rule_is_the_one_the_calendar_worlds_oracle_reads_literally() -> None:
    # On random scoring times, and on those of several worlds, every mule is active at the times
    # its oracle names, which is written apart from this module.
    rng = random.Random(6)
    for _ in range(200):
        times = _grid(rng)
        pool = _anchors(rng, times)
        for _ in range(30):
            onset, last = sorted((rng.choice(pool), rng.choice(pool)))
            found = tuple(t_ms for t_ms in times if active_at(onset, last, t_ms, times))
            assert found == positive_times(Mule(onset, last), times)
    for seed in (1001, 1002, 1003, 2001, 3001):
        times = DEFAULT_CONFIG.calendar.scoring_times(seed)
        for mule in (LONG, SHORT, EDGE, EARLY, GONE):
            found = IdentifiedMule("a_x", mule.onset_ms, mule.last_ms)
            positive = tuple(t_ms for t_ms in times if active(found, t_ms, times))
            assert positive == positive_times(mule, times)
        assert len(positive_times(SHORT, times)) >= 1 and len(positive_times(EDGE, times)) >= 1
        # a_early's last payment is more than 7 days before the first scoring time.
        assert positive_times(EARLY, times) == ()


def test_a_mules_active_times_are_consecutive_scoring_times() -> None:
    # The window's start, the smaller of t - 7 days and the previous scoring time, never
    # decreases along the times, so a mule never goes quiet and comes back.
    rng = random.Random(7)
    for seed in SEEDS:
        times = DEFAULT_CONFIG.calendar.scoring_times(seed)
        pool = _anchors(rng, list(times))
        for _ in range(3_000):
            onset, last = sorted((rng.choice(pool), rng.choice(pool)))
            held = [k for k, t_ms in enumerate(times) if active_at(onset, last, t_ms, times)]
            if held:
                assert held == list(range(held[0], held[0] + len(held)))
        starts = window_starts(np.array(times, dtype=np.int64), times)
        assert (np.diff(starts) >= 0).all()


def test_every_mule_that_started_and_was_active_in_the_last_week_before_the_first_time_has_one() -> (
    None
):
    # A mule with onset < T[-1] and last >= T[0] - 7 days is active at one scoring time at least,
    # since every payment counts at the first scoring time after it; one whose last payment is
    # before T[0] - 7 days is active at none, the first time having no previous one.
    rng = random.Random(8)
    for seed in SEEDS:
        times = DEFAULT_CONFIG.calendar.scoring_times(seed)
        pool = _anchors(rng, list(times)) + [
            times[0] - W,
            times[0] - W - 1,
            times[-1] - 1,
            times[-1],
        ]
        for _ in range(5_000):
            onset, last = sorted((rng.choice(pool), rng.choice(pool)))
            held = [t_ms for t_ms in times if active_at(onset, last, t_ms, times)]
            if onset < times[-1] and last >= times[0] - W:
                assert held
            if last < times[0] - W:
                assert not held


def test_the_two_mules_of_seed_1001_the_plan_names() -> None:
    times = DEFAULT_CONFIG.calendar.scoring_times(1001)
    # Active 7 to 22 January 2024, 75 days before the first scoring time, Saturday 6 April
    # 18:44: no active time, where the first rule gave it the first scoring time.
    january = IdentifiedMule("a_january", _utc(2024, 1, 7), _utc(2024, 1, 22))
    assert not any(active(january, t_ms, times) for t_ms in times)
    assert unscored([january], times, {}).no_active_week == ("a_january",)
    assert unscored([january], times, {}).never_scorable == ()
    # Onset Tuesday 9 July 2024 09:50 and last the next day: between the scoring times of Monday
    # 1 July 09:50 and Saturday 13 July 22:22, so 13 July is its only active time. Its last
    # payment plus 7 days (17 July) reaches that time too, so this half pins the active time,
    # not the longer window (test_a_short_mule_after_a_long_gap_has_one_positive_time does);
    # its early-warning time, 1 July, is pinned in tests/evaluation/test_answer_key.py.
    july = IdentifiedMule("a_july", _utc(2024, 7, 9, 9, 50), _utc(2024, 7, 10, 9, 50))
    assert times[13] < july.mule_onset_ms < times[14]
    assert [t_ms for t_ms in times if active(july, t_ms, times)] == [times[14]]


def test_a_short_mule_after_a_long_gap_has_one_positive_time() -> None:
    # A real world's times can be 12 days apart: a mule that lives a day, right after one
    # scoring time, is active at the one that ends the gap, since the window reaches back to
    # the previous scoring time.
    times = DEFAULT_CONFIG.calendar.scoring_times(1001)
    gap = next(k for k in range(len(times) - 1) if times[k + 1] - times[k] > 10 * DAY_MS)
    onset = times[gap] + DAY_MS
    mule = IdentifiedMule("a_gap", onset, onset + DAY_MS)
    assert onset + DAY_MS + W < times[gap + 1]
    assert [t_ms for t_ms in times if active(mule, t_ms, times)] == [times[gap + 1]]


def test_mules_never_scored_at_a_scoring_time_of_their_activity_are_reported() -> None:
    times = DEFAULT_CONFIG.calendar.scoring_times(1001)
    mules = [
        IdentifiedMule(account, mule.onset_ms, mule.last_ms)
        for account, mule in sorted(IDENTIFIED.items())
    ]
    # Positives as mule prepare counts them: scorable ones only, so a_gone has none. a_early's
    # last payment is before the first scoring time less 7 days, so it is active at none.
    found = unscored(mules, times, {"a_long": 26, "a_short": 1, "a_edge": 1})
    assert found.no_active_week == ("a_early",)
    assert found.never_scorable == ("a_gone",)
    # Without the positives of a_edge, which is active at some times, it was scorable at none,
    # as a_gone is; a mule whose onset follows every scoring time is active at none.
    assert unscored(mules, times, {"a_long": 26, "a_short": 1}).never_scorable == (
        "a_edge",
        "a_gone",
    )
    last = IdentifiedMule("a_last", times[-1] + 1, times[-1] + DAY_MS)
    assert unscored([last], times, {}).no_active_week == ("a_last",)
    assert sum(active(mules[2], t_ms, times) for t_ms in times) == len(positive_times(GONE, times))
