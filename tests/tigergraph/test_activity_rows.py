"""read_activity_change's parameter builder and row parser, on answers written by hand in
TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set as a list
of vertices with their projected fields under "attributes".

The parser takes a row for every account asked for and no other, and refuses the values the
query cannot give.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries import (
    READ_ACTIVITY_CHANGE,
    VERSIONS,
    read_activity_change,
)
from mule_pattern_learner.tigergraph.queries.read_activity_change import ActivityChange
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS, QueryRefused

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday
HOUR_MS = 3_600_000


def test_a_request_takes_a_time_and_its_accounts_each_once() -> None:
    assert read_activity_change.params(T_MS, ["a_02", "a_01"]) == {
        "t_ms": T_MS,
        "account_ids": ["a_02", "a_01"],
    }
    # Any whole second is a time: a Tuesday afternoon is one, a millisecond is not.
    assert read_activity_change.params(T_MS + DAY_MS + 3_661_000, ["a_01"])["t_ms"] == (
        T_MS + DAY_MS + 3_661_000
    )
    with pytest.raises(ValueError, match="whole second"):
        read_activity_change.params(T_MS + 1, ["a_01"])
    with pytest.raises(ValueError, match="repeats an account"):
        read_activity_change.params(T_MS, ["a_01", "a_01"])
    with pytest.raises(ValueError, match="at least one account"):
        read_activity_change.params(T_MS, [])


# A mule account that woke up last week, and a dormant account.
# - a_01 received 5 payments in the last 7 days and none before, 3 of them within an hour of
#   one evening and all 5 within 24 hours; the shortest gap between two of them was 4 minutes
#   (240,000 ms), and the latest came 2 days before t. It sent 4, the latest 1 day and 20
#   hours before t (158,400,000 ms), 2 within an hour and 4 within 24 hours.
# - a_02 received a salary 45 days before t (3,888,000,000 ms) and has sent nothing in 90
#   days: no burst, no gap, no time since its last outflow.
MULE = ActivityChange(
    account_id="a_01",
    in_7d=5,
    in_30d=5,
    in_90d=5,
    out_7d=4,
    out_30d=4,
    out_90d=4,
    burst_in_1h=3,
    burst_in_24h=5,
    burst_out_1h=2,
    burst_out_24h=4,
    shortest_in_gap_present=True,
    shortest_in_gap_ms=240_000,
    since_last_in_present=True,
    since_last_in_ms=2 * DAY_MS,
    since_last_out_present=True,
    since_last_out_ms=DAY_MS + 20 * HOUR_MS,
)
DORMANT = ActivityChange(
    account_id="a_02",
    in_7d=0,
    in_30d=0,
    in_90d=1,
    out_7d=0,
    out_30d=0,
    out_90d=0,
    burst_in_1h=0,
    burst_in_24h=0,
    burst_out_1h=0,
    burst_out_24h=0,
    shortest_in_gap_present=False,
    shortest_in_gap_ms=0,
    since_last_in_present=True,
    since_last_in_ms=45 * DAY_MS,
    since_last_out_present=False,
    since_last_out_ms=0,
)


# Rows at the edges of what the query can give, MULE with its inflows changed.
# - ONE_IN: 1 inflow in 30 days, 2 days ago, so a burst of 1 and no gap.
# - FULL: 256 inflows in the last 30 days, all kept, so a burst of at least 1 and a gap.
ONE_IN = replace(
    MULE,
    in_7d=1,
    in_30d=1,
    in_90d=1,
    burst_in_1h=1,
    burst_in_24h=1,
    shortest_in_gap_present=False,
    shortest_in_gap_ms=0,
)
FULL = replace(
    MULE,
    in_7d=256,
    in_30d=256,
    in_90d=256,
    burst_in_1h=1,
    burst_in_24h=9,
    shortest_in_gap_ms=2 * HOUR_MS,
)


def _row(row: ActivityChange) -> dict[str, Any]:
    return {"v_id": row.account_id, "v_type": "Account", "attributes": vars(row)}


def _answer(rows: list[ActivityChange]) -> list[dict[str, Any]]:
    """An answer as TigerGraph prints it."""
    return [
        {"status": "ok", "version": VERSIONS[READ_ACTIVITY_CHANGE]},
        {"accounts": [_row(row) for row in rows]},
    ]


def test_an_answer_parses_each_account_sorted_by_id() -> None:
    parsed = read_activity_change.parse(_answer([DORMANT, MULE]), ["a_01", "a_02"])
    assert parsed == (MULE, DORMANT)


def test_a_status_row_is_refused() -> None:
    for status in (
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
    ):
        with pytest.raises(QueryRefused, match=status) as refused:
            read_activity_change.parse([{"status": status}], ["a_01"])
        assert refused.value.status == status


@pytest.mark.parametrize(
    ("rows", "asked"),
    [
        ([MULE], ["a_01", "a_02"]),
        ([MULE, DORMANT], ["a_01"]),
        ([MULE, MULE, DORMANT], ["a_01", "a_02"]),
    ],
    ids=["missing", "not asked", "twice"],
)
def test_a_row_for_every_account_asked_and_no_other(
    rows: list[ActivityChange], asked: list[str]
) -> None:
    with pytest.raises(ValueError, match="answered the accounts"):
        read_activity_change.parse(_answer(rows), asked)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(MULE, in_30d=6), "in counts that shrink"),
        (replace(MULE, out_90d=3), "out counts that shrink"),
        (replace(MULE, burst_in_1h=6), "in bursts beyond"),
        (replace(MULE, burst_out_24h=5), "out bursts beyond"),
        (replace(MULE, burst_out_1h=0), "no out burst though it kept payments"),
        (replace(DORMANT, burst_in_24h=1), "in bursts beyond"),
        (replace(DORMANT, since_last_in_present=False, since_last_in_ms=0), "unlike its count"),
        (replace(DORMANT, since_last_out_present=True), "unlike its count"),
        (replace(DORMANT, since_last_out_ms=5), "absent time since the last payment out"),
        (replace(MULE, since_last_in_ms=8 * DAY_MS), "since the last payment in unlike"),
        (replace(DORMANT, since_last_in_ms=6 * DAY_MS), "since the last payment in unlike"),
        (replace(DORMANT, since_last_in_ms=91 * DAY_MS), "since the last payment in unlike"),
        (replace(MULE, since_last_out_ms=0), "since the last payment out unlike"),
        (replace(DORMANT, shortest_in_gap_present=True, shortest_in_gap_ms=1_000), "a gap"),
        (replace(MULE, shortest_in_gap_ms=0), "a gap"),
        (replace(MULE, shortest_in_gap_ms=30 * DAY_MS), "a gap"),
        (replace(MULE, shortest_in_gap_present=False), "an absent gap"),
        (replace(ONE_IN, shortest_in_gap_present=True, shortest_in_gap_ms=240_000), "a gap"),
        (replace(ONE_IN, burst_in_1h=0, burst_in_24h=0), "no in burst though it kept payments"),
        (replace(FULL, burst_in_1h=0, burst_in_24h=0), "no in burst though it kept payments"),
        (replace(ONE_IN, since_last_in_ms=30 * DAY_MS + 1), "since the last payment in unlike"),
        (replace(DORMANT, since_last_in_ms=29 * DAY_MS), "since the last payment in unlike"),
    ],
    ids=[
        "in counts",
        "out counts",
        "in hour over day",
        "out day over kept",
        "no out burst",
        "a burst with nothing kept",
        "no time since an inflow of the lookback",
        "a time since no outflow",
        "an absent time that is not 0",
        "in 7 days but over 7 days ago",
        "within 7 days but no inflow then",
        "before the lookback",
        "at t",
        "a gap with one inflow",
        "a gap of 0",
        "a gap of 30 days",
        "an absent gap that is not 0",
        "a gap with exactly 1 inflow",
        "no burst of exactly 1 inflow",
        "no burst of exactly 256 inflows",
        "an inflow 1 ms before 30 days ago counted in 30 days",
        "no inflow in 30 days but one less than 30 days ago",
    ],
)
def test_a_row_the_query_cannot_give_is_refused(row: ActivityChange, message: str) -> None:
    others = [DORMANT] if row.account_id == "a_01" else [MULE]
    with pytest.raises(ValueError, match=message):
        read_activity_change.parse(_answer([row, *others]), ["a_01", "a_02"])


def test_the_boundaries_of_the_count_windows() -> None:
    """A payment exactly 7 days before t counts over 7 days, so the time since it is exactly
    7 days; one exactly 90 days before is the oldest of the lookback."""
    week = replace(
        DORMANT, in_7d=1, in_30d=1, burst_in_1h=1, burst_in_24h=1, since_last_in_ms=7 * DAY_MS
    )
    assert read_activity_change.parse(_answer([MULE, week]), ["a_01", "a_02"])[1] == week
    oldest = replace(DORMANT, since_last_in_ms=90 * DAY_MS)
    assert read_activity_change.parse(_answer([MULE, oldest]), ["a_01", "a_02"])[1] == oldest


@pytest.mark.parametrize(
    "row",
    [
        # 2 of 10 inflows in the last 7 days, the burst of 6 reaching back past the week: the
        # bursts and the time since read the 30 days, not the 7.
        replace(
            MULE,
            in_7d=2,
            in_30d=10,
            in_90d=10,
            burst_in_1h=6,
            burst_in_24h=6,
            since_last_in_ms=3 * DAY_MS,
        ),
        # The latest inflow 10 days ago: none in 7 days, some in 30.
        replace(
            MULE,
            in_7d=0,
            in_30d=3,
            in_90d=3,
            burst_in_1h=2,
            burst_in_24h=3,
            since_last_in_ms=10 * DAY_MS,
        ),
        # The latest inflow exactly 30 days ago still counts over 30 days.
        replace(ONE_IN, in_7d=0, since_last_in_ms=30 * DAY_MS),
        # A gap needs two inflows in 30 days, and two are enough.
        replace(MULE, in_7d=2, in_30d=2, in_90d=2, burst_in_1h=2, burst_in_24h=2),
        # 256 inflows are all kept, so one of them is a burst; 257 may leave none kept.
        FULL,
        replace(
            FULL,
            in_7d=257,
            in_30d=257,
            in_90d=257,
            burst_in_1h=0,
            burst_in_24h=0,
            shortest_in_gap_present=False,
            shortest_in_gap_ms=0,
        ),
    ],
    ids=[
        "bursts over the 30 days, not the 7",
        "the latest inflow between 7 and 30 days ago",
        "the latest inflow exactly 30 days ago",
        "a gap between exactly 2 inflows",
        "exactly 256 inflows, all kept",
        "exactly 257 inflows, none kept",
    ],
)
def test_a_row_at_the_edge_of_what_the_query_gives_is_accepted(row: ActivityChange) -> None:
    assert read_activity_change.parse(_answer([row, DORMANT]), ["a_01", "a_02"])[0] == row


def test_more_payments_than_the_cap_may_keep_none_in_a_burst() -> None:
    """With more than 256 inflows in 30 days the whole-second rule may drop them all (257 in
    one second leave none after it), so only the cap bounds a burst; under it, every payment
    is kept and counts at least itself."""
    busy = replace(MULE, in_7d=300, in_30d=300, in_90d=300, burst_in_1h=0, burst_in_24h=0)
    assert read_activity_change.parse(_answer([busy, DORMANT]), ["a_01", "a_02"])[0] == busy
    with pytest.raises(ValueError, match="in bursts beyond"):
        read_activity_change.parse(
            _answer([replace(busy, burst_in_24h=257), DORMANT]), ["a_01", "a_02"]
        )
