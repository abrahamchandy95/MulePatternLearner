"""The clock gain test (docs/design.md, the Clock guard): the controls and the clock values of an
example, read from its rows; the test passes on a world whose events' clock times say nothing
about is_mule beyond what the controls hold, fails on a world whose mules pay at 07:00 on days
like everybody's, and fails with fewer mule clusters than the minimum whatever its interval
says; and its interval is of the gain on shared resamples."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, AuditConfig
from mule_pattern_learner.evaluation import clock_gain
from mule_pattern_learner.evaluation.clock_gain import (
    CONTROLS,
    EVENTS,
    FAMILIES,
    MISSING,
    OTHER,
    OWN,
    PER_EVENT,
    WIDTH,
    Gain,
    clock_values,
    control_values,
    durations,
    folds,
    gain,
    node_counts,
)
from mule_pattern_learner.subgraph.build import read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T

AUDIT = replace(DEFAULT_CONFIG.audit, replicates=100)


def test_the_durations_of_an_example_by_family(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: ["a_hr1"]}, parity=parity)
    (rows,) = read_rows(cache, [(T, "a_hr1")])
    # a_hr1 paid a_hr2 from the ring's device 4 days 12 hours before T, its one payment before
    # T; it received none. Its customer's link to the device started 20 days before T, with no
    # earlier payment of its own, so neither that start nor its first use there is known.
    ago = 4 * DAY_MS + 12 * HOUR_MS
    # Beside it, in its subgraph, a_hr2 paid a_hr3 at the same time: a payment of its own that
    # a_hr1 is not at either end of, a_hr2's last payment received (from a_hr1) and sent, and
    # a_hr3's last received. The three ring customers are enrolled on the device, p_hr1 and p_hr2
    # having used it from an account 4 days 12 hours ago (p_hr3 never): three enrolment edges
    # that have a last use, as have a_hr2's owner-enrolment edge and its seen edge, while a_hr3
    # has an owner-enrolment edge with none and no seen edge, and no start is known anywhere.
    found = durations(rows)
    assert found == {
        "age": [ago],
        "since_last_in": [],
        "since_last_out": [ago],
        "since_start": [],
        "since_last_seen": [ago],
        "since_first_use": [],
        "since_last_use": [ago],
        "other_age": [ago],
        "other_since_last_in": [ago, ago],
        "other_since_last_out": [ago],
        "other_since_start": [],
        "other_since_last_seen": [ago, ago, ago],
        "other_since_first_use": [],
        "other_since_last_use": [ago],
    }
    # Its subgraph holds the three ring accounts and their three customers, the one device they
    # share, and no IP; the controls count each family's events and floor each age to whole
    # days: 4 days 12 hours is 4.
    assert node_counts(rows) == [3, 3, 1, 0]
    values = control_values(found, node_counts(rows))
    assert values.shape == (CONTROLS,) == (130,)
    counts = [len(found[family]) for family in FAMILIES]
    assert values[: len(FAMILIES)].tolist() == counts == [1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 0, 3, 0, 1]
    assert values[len(FAMILIES) : len(FAMILIES) + 4].tolist() == [3, 3, 1, 0]
    days = values[len(FAMILIES) + 4 :].reshape(len(FAMILIES), EVENTS)
    assert days[FAMILIES.index("age")].tolist() == [4] + [MISSING] * (EVENTS - 1)
    assert days[FAMILIES.index("other_since_last_seen")].tolist() == [4, 4, 4] + [MISSING] * 5
    assert (days[FAMILIES.index("since_start")] == MISSING).all()


def test_clock_values_by_hand() -> None:
    # t is Monday 2024-07-01 00:00 UTC. An age of 25 hours 30 minutes puts the payment at
    # Saturday 2024-06-29 22:30:00: 0 ms past the minute, 30 minutes past the hour, 22 h 30 min
    # into the day, and 2 days 22 h 30 min after the Thursday 00:00 that starts each week of
    # epoch time; the 29th of the month.
    found = {family: [] for family in FAMILIES} | {"age": [91_800_000, 200 * HOUR_MS]}
    values = clock_values(T, found).reshape(len(FAMILIES), EVENTS, 5)
    assert values[0, 0].tolist() == [0, 1_800_000, 81_000_000, 2 * DAY_MS + 81_000_000, 29]
    # 200 hours before T is 2024-06-22 16:00, a Saturday.
    assert values[0, 1].tolist() == [0, 0, 16 * HOUR_MS, 2 * DAY_MS + 16 * HOUR_MS, 22]
    assert (values[0, 2:] == MISSING).all() and (values[1:] == MISSING).all()
    assert values.size == WIDTH == 560


def test_the_newest_events_of_a_family_are_kept_in_order() -> None:
    ages = [hours * HOUR_MS for hours in (50, 3, 9, 1, 7, 30, 2, 80, 11, 5)]
    values = clock_values(T, {family: [] for family in FAMILIES} | {"since_last_use": ages})
    kept = values.reshape(len(FAMILIES), EVENTS, 5)[FAMILIES.index("since_last_use"), :, 2]
    # The eight shortest ages, hours before midnight: 1, 2, 3, 5, 7, 9, 11 and 30 hours.
    assert kept.tolist() == [(24 - h % 24) % 24 * HOUR_MS for h in (1, 2, 3, 5, 7, 9, 11, 30)]


def test_controls_hold_counts_and_whole_days_and_no_clock_time() -> None:
    ages = [hours * HOUR_MS for hours in (50, 3, 9, 1, 7, 30, 2, 80, 11, 5)]
    found = {family: [] for family in FAMILIES} | {"since_last_use": ages}
    values = control_values(found, [4, 3, 2, 1])
    # Ten events with a duration, though only the eight newest are kept; four node counts.
    assert values[: len(FAMILIES)].tolist() == [0] * 6 + [10] + [0] * 7
    assert values[len(FAMILIES) : len(FAMILIES) + 4].tolist() == [4, 3, 2, 1]
    days = values[len(FAMILIES) + 4 :].reshape(len(FAMILIES), EVENTS)
    # 1, 2, 3, 5, 7, 9, 11 and 30 hours are 0, 0, 0, 0, 0, 0, 0 and 1 days.
    assert days[FAMILIES.index("since_last_use")].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    # Moving every event by a few hours inside its day changes no control, and changes the clock
    # values.
    moved = {family: [age + HOUR_MS // 2 for age in held] for family, held in found.items()}
    assert np.array_equal(control_values(moved, [4, 3, 2, 1]), values)
    assert not np.array_equal(clock_values(T, moved), clock_values(T, found))


def test_the_folds_keep_a_cluster_together() -> None:
    keys = [f"ring:{index % 40}" for index in range(400)]
    found = folds(keys)
    assert set(found.tolist()) == {0, 1}
    by_key = {key: set(found[[i for i, k in enumerate(keys) if k == key]].tolist()) for key in keys}
    assert all(len(held) == 1 for held in by_key.values())


def _ages(rng: np.random.Generator, count: int, at_seven: bool) -> list[int]:
    """The ages of count events on random days of the 30 before T, at a random time of day, or
    when at_seven is set between 07:00 and 07:59, so that the days are the same either way."""
    days = rng.integers(1, 31, count)
    clock = (
        7 * HOUR_MS + rng.integers(0, HOUR_MS, count)
        if at_seven
        else rng.integers(1, 24 * HOUR_MS, count)
    )
    return (days * DAY_MS - clock).tolist()


@cache
def _world(
    rings: int, others: int, clock: bool, seed: int, clocked: str = "age"
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], list[str]
]:
    """A gain test's input: two mules in each ring and others non-mules, each with a count of
    events of the family clocked names (by default its own payments, 5 to 8 for a mule and 3 to 8
    for a non-mule, so that the controls separate them a little) and a last outflow, on random
    days of the 30 before t. A non-mule's events fall at random times of day; a mule's, when
    clock is set, at 07:00 to 07:59 in that family, which the controls do not show."""
    rng = np.random.default_rng(seed)
    controls: list[NDArray[np.float64]] = []
    values: list[NDArray[np.float64]] = []
    positive: list[bool] = []
    keys: list[str] = []
    for index in range(2 * rings + others):
        mule = index < 2 * rings
        count = int(rng.integers(5, 9)) if mule else int(rng.integers(3, 9))
        found = {family: [] for family in FAMILIES}
        found["age"] = _ages(rng, count, False)
        found["since_last_out"] = _ages(rng, 1, False)
        found[clocked] = _ages(rng, count, mule and clock)
        controls.append(control_values(found, [3, 3, 1, 1]))
        values.append(clock_values(T, found))
        positive.append(mule)
        keys.append(f"ring:{index // 2}" if mule else f"account:{index}")
    return (
        np.array(controls),
        np.array(values),
        np.array(positive),
        np.ones(len(positive)),
        keys,
    )


def test_the_test_passes_on_a_world_whose_clock_times_say_nothing_beyond_the_controls() -> None:
    controls, values, positive, weight, keys = _world(3_000, 24_000, clock=False, seed=1)
    found = gain(controls, values, positive, weight, keys, AUDIT, seed=0)
    # The controls separate the mules a little, and the clock values add nothing.
    assert 0.6 < found.roc_auc_controls < 0.75
    assert found.interval is not None and found.interval[1] < AUDIT.clock_gain_upper
    assert found.passed and abs(found.gain) < AUDIT.clock_gain_upper
    assert (found.mule_rows, found.clusters, found.mule_clusters) == (6_000, 27_000, 3_000)


def test_the_test_fails_on_a_world_whose_mules_pay_at_seven_on_days_like_everybodys() -> None:
    controls, values, positive, weight, keys = _world(300, 3_000, clock=True, seed=1)
    found = gain(controls, values, positive, weight, keys, AUDIT, seed=0)
    # The controls hold the days, which are alike for mules and others, and counts that separate
    # them a little; the clock values hold the time of day, which separates them wholly.
    assert found.roc_auc_controls < 0.75 and found.roc_auc_with_clock > 0.95
    assert found.gain > 0.2 and not found.passed
    assert found.interval is not None and found.interval[0] > AUDIT.clock_gain_upper


def _clean(rings: int, others: int, seed: int, audit: AuditConfig = AUDIT) -> Gain:
    """The gain test on a clock-free world of rings and others non-mules."""
    controls, values, positive, weight, keys = _world(rings, others, clock=False, seed=seed)
    return gain(controls, values, positive, weight, keys, audit, seed=0)


@pytest.mark.parametrize("seed", range(6))
def test_a_test_with_fewer_mule_clusters_than_the_minimum_fails(seed: int) -> None:
    # Three rings and ten non-mules of a clock-free world, on every seed tried.
    found = _clean(3, 10, seed)
    assert found.mule_clusters == 3 < AUDIT.clock_min_mule_clusters == 30
    assert not found.passed


def test_the_minimum_is_what_fails_a_world_whose_interval_would_pass() -> None:
    # Seed 1 of the same world passes without a minimum: with a handful of mules the interval
    # is wide or lucky, and a gain below 0 on every resample is no evidence of none.
    lax = _clean(3, 10, 1, replace(AUDIT, clock_min_mule_clusters=1))
    assert lax.passed and lax.interval is not None and lax.interval[1] < AUDIT.clock_gain_upper
    found = _clean(3, 10, 1)
    assert found.interval == lax.interval and not found.passed


@pytest.mark.parametrize(
    ("rings", "least", "passes"), [(29, 30, False), (30, 30, True), (29, 29, True), (30, 31, False)]
)
def test_the_test_passes_from_the_minimum_of_mule_clusters_up(
    monkeypatch: pytest.MonkeyPatch, rings: int, least: int, passes: bool
) -> None:
    controls, values, positive, weight, keys = _world(rings, 300, clock=False, seed=1)
    plain = np.random.default_rng(0).normal(size=len(positive)) + 0.3 * positive
    scores = iter([plain, plain])

    def fixed(
        values: NDArray[np.float64],
        positive: NDArray[np.bool_],
        fold: NDArray[np.int64],
        seed: int,
    ) -> NDArray[np.float64]:
        return next(scores)

    monkeypatch.setattr(clock_gain.process, "fold_scores", fixed)
    # The same scores twice: the gain is 0 on every resample, the interval is the point [0, 0],
    # below the bound, and only the number of mule clusters decides.
    audit = replace(AUDIT, clock_min_mule_clusters=least)
    found = gain(controls, values, positive, weight, keys, audit, seed=0)
    assert found.interval == (0.0, 0.0) and found.mule_clusters == rings
    assert found.passed is passes


def test_the_table_of_the_test_is_one_row() -> None:
    frame = _clean(40, 100, seed=1).frame()
    columns = [
        "roc_auc_controls",
        "roc_auc_with_clock",
        "gain",
        "low",
        "high",
        "passed",
        "mule_rows",
        "clusters",
        "mule_clusters",
    ]
    assert list(frame.columns) == columns and len(frame) == 1
    assert frame.iloc[0][["mule_rows", "clusters", "mule_clusters"]].tolist() == [80, 140, 40]
    # A test with no interval has none in its table, and keeps its counts.
    nothing = Gain(np.nan, np.nan, np.nan, None, False, 6, 13, 3).frame().iloc[0]
    assert np.isnan(nothing["low"]) and np.isnan(nothing["high"]) and not nothing["passed"]
    assert nothing[["mule_rows", "clusters", "mule_clusters"]].tolist() == [6, 13, 3]
    # With every mule in one fold, the other has none to learn from, and there is no interval at
    # all, whatever the minimum.
    found = _clean(1, 300, seed=2, audit=replace(AUDIT, clock_min_mule_clusters=1))
    assert found.interval is None and not found.passed and np.isnan(found.gain)
    assert found.mule_clusters == 1


@pytest.mark.parametrize(
    "family", [f"{OTHER}age", f"{OTHER}since_last_seen", f"{OTHER}since_last_use"]
)
def test_the_test_fails_on_a_clock_habit_that_only_the_neighbours_edges_show(family: str) -> None:
    # The mules' own payments, last outflow and edges are at random times; the habit is on a
    # neighbour's payments, a co-user's enrolment edges or another account's seen edges, which
    # the model reads through message passing.
    controls, values, positive, weight, keys = _world(
        300, 3_000, clock=True, seed=1, clocked=family
    )
    found = gain(controls, values, positive, weight, keys, AUDIT, seed=0)
    assert found.roc_auc_with_clock > 0.95 and not found.passed
    # Read as the scored account's own families alone, the same world, with the power of a
    # larger one, would have passed.
    controls, values, positive, weight, keys = _world(
        3_000, 24_000, clock=True, seed=1, clocked=family
    )
    blind = values.copy()
    blind[:, len(OWN) * EVENTS * PER_EVENT :] = MISSING
    assert gain(controls, blind, positive, weight, keys, AUDIT, seed=0).passed


def test_the_interval_is_of_the_gain_on_resamples_shared_by_both_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls, values, positive, weight, keys = _world(300, 3_000, clock=False, seed=1)
    rng = np.random.default_rng(0)
    plain = rng.normal(size=len(positive)) + 0.3 * positive
    scores = iter([plain, plain])

    def fixed(
        values: NDArray[np.float64],
        positive: NDArray[np.bool_],
        fold: NDArray[np.int64],
        seed: int,
    ) -> NDArray[np.float64]:
        return next(scores)

    monkeypatch.setattr(clock_gain.process, "fold_scores", fixed)
    # The same scores twice: the gain is 0 on every resample, which two independent resamples
    # would not give, and the interval is a point.
    found = gain(controls, values, positive, weight, keys, AUDIT, seed=0)
    assert found.gain == 0.0 and found.interval == (0.0, 0.0) and found.passed
    assert found.roc_auc_controls == found.roc_auc_with_clock > 0.55
    # The clock values lift every mule's score a little: no pair of a mule and a non-mule is
    # ranked the other way round by it, so the gain is above 0 on every resample, and the
    # interval, which would span 0 were the two resampled apart, does not.
    scores = iter([plain, plain + 0.02 * positive])
    found = gain(controls, values, positive, weight, keys, AUDIT, seed=0)
    assert found.gain > 0 and found.interval is not None and found.interval[0] >= 0


def test_the_controls_and_the_clock_values_need_their_widths() -> None:
    controls, values, positive, weight, keys = _world(4, 300, clock=False, seed=2)
    with pytest.raises(ValueError, match="The controls are"):
        gain(controls[:, :-1], values, positive, weight, keys, AUDIT, seed=0)
    with pytest.raises(ValueError, match="The clock values are"):
        gain(controls, values[:, :-1], positive, weight, keys, AUDIT, seed=0)
