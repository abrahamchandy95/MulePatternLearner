"""read_money_flow's parameter builder and row parser, on answers written by hand in
TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set as a list
of vertices with their projected fields under "attributes", a printed list of tuples as a list
of objects.

The parser takes a row for every account asked for and no other, and refuses the values the
walk cannot give.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries import READ_MONEY_FLOW, VERSIONS, read_money_flow
from mule_pattern_learner.tigergraph.queries.read_money_flow import MoneyFlow, TrackedInflow
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS, QueryRefused

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday


def test_a_request_takes_a_time_and_its_accounts_each_once() -> None:
    """The accounts go as a list of ids, which the query checks itself: an installed query
    drops an id the graph lacks from a SET<VERTEX> parameter without a word."""
    assert read_money_flow.params(T_MS, ["a_02", "a_01"]) == {
        "t_ms": T_MS,
        "account_ids": ["a_02", "a_01"],
    }
    # Any whole second is a time: a Tuesday afternoon is one, a millisecond is not.
    assert read_money_flow.params(T_MS + DAY_MS + 3_661_000, ["a_01"])["t_ms"] == (
        T_MS + DAY_MS + 3_661_000
    )
    with pytest.raises(ValueError, match="whole second"):
        read_money_flow.params(T_MS + 1, ["a_01"])
    with pytest.raises(ValueError, match="repeats an account"):
        read_money_flow.params(T_MS, ["a_01", "a_01"])
    with pytest.raises(ValueError, match="at least one account"):
        read_money_flow.params(T_MS, [])
    with pytest.raises(ValueError, match="holds account ids"):
        read_money_flow.params(T_MS, ["a_01", ""])


# A mule account that passed on most of two credits fast, and a saver with nothing received in
# its span. a_01 received R = 10,000 cents in two lots: 9,000 left within 15 minutes, by
# transfer (8,000) and card (1,000), so every window from 15 minutes on holds 9,000 / 10,000;
# 1,000 cents are held; the median dwell is 600 s, where 9,000 first reaches half of 10,000.
MULE = MoneyFlow(
    account_id="a_01",
    in_amount_7d=10_000,
    in_amount_30d=10_000,
    in_amount_90d=12_000,
    out_amount_7d=9_000,
    out_amount_30d=9_000,
    out_amount_90d=9_500,
    flow_present=True,
    left_15m=0.9,
    left_1h=0.9,
    left_6h=0.9,
    left_1d=0.9,
    left_3d=0.9,
    left_7d=0.9,
    held_share=0.1,
    median_dwell_present=True,
    median_dwell_ms=600_000,
    left_by_transfer=8_000 / 9_000,
    left_by_card=1_000 / 9_000,
    left_by_cash=0.0,
    scan_steps=3,
)
SAVER = MoneyFlow(
    account_id="a_02",
    in_amount_7d=0,
    in_amount_30d=0,
    in_amount_90d=250_000,
    out_amount_7d=0,
    out_amount_30d=0,
    out_amount_90d=0,
    flow_present=False,
    left_15m=0.0,
    left_1h=0.0,
    left_6h=0.0,
    left_1d=0.0,
    left_3d=0.0,
    left_7d=0.0,
    held_share=0.0,
    median_dwell_present=False,
    median_dwell_ms=0,
    left_by_transfer=0.0,
    left_by_card=0.0,
    left_by_cash=0.0,
    scan_steps=0,
)
# a_01's two lots: one of two payments in one second, which share their lot's values, all of
# whose cents left within 24 hours, half of them 600 s after it arrived; and one of 1,000
# cents, of which none had left by t.
INFLOWS = [
    TrackedInflow("a_01", "p_09", 1.0, True, 600_000),
    TrackedInflow("a_01", "p_03", 1.0, True, 600_000),
    TrackedInflow("a_01", "p_05", 0.0, False, 0),
]


def _row(row: MoneyFlow) -> dict[str, Any]:
    return {"v_id": row.account_id, "v_type": "Account", "attributes": vars(row)}


def _answer(
    rows: list[MoneyFlow], inflows: list[TrackedInflow] | None = None
) -> list[dict[str, Any]]:
    """An answer as TigerGraph prints it."""
    tracked = INFLOWS if inflows is None else inflows
    return [
        {"status": "ok", "version": VERSIONS[READ_MONEY_FLOW]},
        {"accounts": [_row(row) for row in rows]},
        {"tracked_inflows": [vars(inflow) for inflow in tracked]},
    ]


def test_an_answer_parses_each_account_and_tracked_inflow_sorted_by_id() -> None:
    parsed = read_money_flow.parse(_answer([SAVER, MULE]), ["a_01", "a_02"])
    assert parsed.accounts == (MULE, SAVER)
    assert [row.payment_id for row in parsed.tracked_inflows] == ["p_03", "p_05", "p_09"]
    # A share printed as an integer, as TigerGraph prints 0 and 1, is a float.
    answer = _answer([MULE, SAVER])
    answer[1]["accounts"][1]["attributes"]["left_15m"] = 0
    assert read_money_flow.parse(answer, {"a_01", "a_02"}).accounts[1].left_15m == 0.0


def test_a_status_row_is_refused() -> None:
    refusal = {"status": "unsorted_heap", "accounts": ["a_01"]}
    with pytest.raises(QueryRefused) as refused:
        read_money_flow.parse([refusal], ["a_01"])
    assert (refused.value.status, refused.value.row) == ("unsorted_heap", refusal)
    for status in (
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
    ):
        with pytest.raises(QueryRefused, match=status):
            read_money_flow.parse([{"status": status}], ["a_01"])


@pytest.mark.parametrize(
    ("rows", "asked"),
    [
        ([MULE], ["a_01", "a_02"]),
        ([MULE, SAVER], ["a_01"]),
        ([MULE, MULE, SAVER], ["a_01", "a_02"]),
    ],
    ids=["missing", "not asked", "twice"],
)
def test_a_row_for_every_account_asked_and_no_other(
    rows: list[MoneyFlow], asked: list[str]
) -> None:
    with pytest.raises(ValueError, match="answered the accounts"):
        read_money_flow.parse(_answer(rows), asked)


@pytest.mark.parametrize(
    ("inflows", "message"),
    [
        ([*INFLOWS, TrackedInflow("a_09", "p_11", 0.0, False, 0)], "tracked an inflow of a_09"),
        ([*INFLOWS, INFLOWS[0]], "tracked a payment twice"),
        ([], "flow_present True and no tracked inflows"),
        ([*INFLOWS, TrackedInflow("a_02", "p_11", 0.0, False, 0)], "flow_present False and some"),
        ([replace(INFLOWS[0], left_24h=1.5)], "share outside"),
        ([replace(INFLOWS[2], dwell_ms=1_000)], "a dwell it cannot give"),
        ([replace(INFLOWS[0], dwell_ms=600_500)], "a dwell it cannot give"),
    ],
    ids=[
        "an account not asked",
        "a payment twice",
        "flow with no inflow",
        "an inflow with no flow",
        "a share above 1",
        "an absent dwell",
        "part of a second",
    ],
)
def test_tracked_inflows_the_walk_cannot_give_are_refused(
    inflows: list[TrackedInflow], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_money_flow.parse(_answer([MULE, SAVER], inflows), ["a_01", "a_02"])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(MULE, in_amount_30d=12_001), "in amounts that shrink"),
        (replace(MULE, out_amount_7d=9_001), "out amounts that shrink"),
        (replace(MULE, left_by_cash=-0.1), "outside"),
        (replace(MULE, held_share=1.1), "outside"),
        (replace(MULE, left_1h=0.8), "a dwell share that shrinks"),
        (replace(MULE, left_by_cash=0.2), "rail shares above 1"),
        (replace(MULE, left_15m=0.0, left_1h=0.0, left_6h=0.0, left_1d=0.0), "no cents"),
        (replace(MULE, median_dwell_present=False), "an absent median dwell"),
        (replace(MULE, median_dwell_ms=600_001), "no whole second"),
        (replace(SAVER, held_share=0.5), "without cents received"),
        (replace(SAVER, median_dwell_present=True), "without cents received"),
    ],
    ids=[
        "in amounts",
        "out amounts",
        "a negative share",
        "a share above 1",
        "windows",
        "rails",
        "rails of nothing",
        "median",
        "median second",
        "saver share",
        "saver median",
    ],
)
def test_a_row_the_walk_cannot_give_is_refused(row: MoneyFlow, message: str) -> None:
    others = [SAVER] if row.account_id == "a_01" else [MULE]
    inflows = INFLOWS if row.account_id == "a_01" or not row.flow_present else []
    with pytest.raises(ValueError, match=message):
        read_money_flow.parse(_answer([row, *others], inflows), ["a_01", "a_02"])


def test_rail_shares_may_add_up_to_1_with_rounding() -> None:
    # 9, 18 and 1 cents of 28 that left within a day: each share is one division, and the
    # three add up to 1.0000000000000002 in floats.
    row = replace(MULE, left_by_transfer=9 / 28, left_by_card=18 / 28, left_by_cash=1 / 28)
    assert row.left_by_transfer + row.left_by_card + row.left_by_cash > 1.0
    assert read_money_flow.parse(_answer([row, SAVER]), ["a_01", "a_02"]).accounts[0] == row
