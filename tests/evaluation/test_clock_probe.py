"""The clock probe (docs/design.md, the Clock guard): the clock times of durations measured to t,
read from an example's rows, predict is_mule no better than chance on a clock-free world, and
the probe fails on a world whose mules pay at 07:00 on the 1st, and on one with too few mules."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.clock_probe import (
    EVENTS,
    FAMILIES,
    MISSING,
    WIDTH,
    clock_values,
    durations,
    probe,
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
    assert durations(rows) == {
        "age": [ago],
        "since_last_in": [],
        "since_last_out": [ago],
        "since_start": [],
        "since_last_seen": [ago],
        "since_first_use": [],
        "since_last_use": [ago],
    }


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
    assert values.size == WIDTH == 280


def test_the_newest_events_of_a_family_are_kept_in_order() -> None:
    ages = [hours * HOUR_MS for hours in (50, 3, 9, 1, 7, 30, 2, 80, 11, 5)]
    values = clock_values(T, {family: [] for family in FAMILIES} | {"since_last_use": ages})
    kept = values.reshape(len(FAMILIES), EVENTS, 5)[-1, :, 2]
    # The eight shortest ages, hours before midnight: 1, 2, 3, 5, 7, 9, 11 and 30 hours.
    assert kept.tolist() == [(24 - h % 24) % 24 * HOUR_MS for h in (1, 2, 3, 5, 7, 9, 11, 30)]


def _world(
    rings: int, others: int, clock: bool, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64], list[str]]:
    """A probe's input: two mules in each ring and others non-mules, each with six payments
    and a last outflow before t. A non-mule's events fall at random times in the 30 days
    before t; a mule's, when clock is set, at 07:00 to 07:59 on the 1st of a month, else at
    random times too."""
    rng = np.random.default_rng(seed)
    firsts = [
        int(datetime(2024, month, 1, 7, tzinfo=UTC).timestamp() * 1000) for month in (4, 5, 6)
    ]
    rows: list[NDArray[np.float64]] = []
    positive: list[bool] = []
    keys: list[str] = []
    for index in range(2 * rings + others):
        mule = index < 2 * rings
        if mule and clock:
            events = np.array(firsts)[rng.integers(0, 3, 7)] + rng.integers(0, HOUR_MS, 7)
            ages = (T - events).tolist()
        else:
            ages = rng.integers(HOUR_MS, 30 * DAY_MS, 7).tolist()
        found = {family: [] for family in FAMILIES}
        found["age"] = ages[:6]
        found["since_last_out"] = ages[6:]
        rows.append(clock_values(T, found))
        positive.append(mule)
        keys.append(f"ring:{index // 2}" if mule else f"account:{index}")
    return np.array(rows), np.array(positive), np.ones(len(rows)), keys


def test_the_probe_passes_on_a_clock_free_world() -> None:
    values, positive, weight, keys = _world(300, 3_000, clock=False, seed=1)
    found = probe(values, positive, weight, keys, AUDIT, seed=0)
    assert found.interval is not None and found.interval[1] < 0.55 and found.passed
    assert (found.mule_rows, found.clusters) == (600, 3_300)


def test_the_probe_fails_on_a_world_whose_mules_pay_at_seven_on_the_first() -> None:
    values, positive, weight, keys = _world(300, 3_000, clock=True, seed=1)
    found = probe(values, positive, weight, keys, AUDIT, seed=0)
    assert found.roc_auc > 0.95 and not found.passed


def test_a_probe_with_too_few_mules_fails() -> None:
    # Four rings of a clock-free world: the ROC AUC is near 0.5, but the interval is wide.
    values, positive, weight, keys = _world(4, 300, clock=False, seed=2)
    found = probe(values, positive, weight, keys, AUDIT, seed=0)
    assert found.interval is not None and found.interval[1] >= 0.55 and not found.passed
    # With every mule in one fold, the other has none to learn from, and there is no
    # interval at all.
    values, positive, weight, keys = _world(1, 300, clock=False, seed=2)
    found = probe(values, positive, weight, keys, AUDIT, seed=0)
    assert found.interval is None and not found.passed
