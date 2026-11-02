"""The reference answer of read_activity_change, B3, on worlds worked out by hand from
docs/features.md ("Account node", columns 34 to 46): the look-alike fixtures, the windows'
bounds, the whole-second cut of the bursts' lists, the refusals, and the leak tests over B3.

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it. Durations are in milliseconds.
"""

from __future__ import annotations

from collections.abc import Sequence
import json

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import READ_ACTIVITY_CHANGE, read_activity_change
from mule_pattern_learner.tigergraph.queries.read_activity_change import ActivityChange
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS, QueryRefused

D, H, M = DAY_MS, HOUR_MS, fixtures.MINUTE_MS


def _rows(graph: ReferenceGraph, accounts: Sequence[str], t: int = T) -> tuple[ActivityChange, ...]:
    found = graph.run(READ_ACTIVITY_CHANGE, read_activity_change.params(t, accounts))
    return read_activity_change.parse(found, accounts)


def _existing(graph: ReferenceGraph, t: int = T) -> list[str]:
    """Every account that exists at t, which B3 may be asked for."""
    return [a for a in graph.ids["Account"] if graph.accounts[a].opened_ms < t]


def _row(
    account: str,
    received: tuple[int, int, int],
    sent: tuple[int, int, int],
    bursts: tuple[int, int, int, int],
    gap: int | None,
    since_in: int | None,
    since_out: int | None,
) -> ActivityChange:
    """A row from its counts in and out over 7, 30 and 90 days, its bursts in and out in an
    hour and in 24 hours, and its durations in milliseconds, None when absent."""
    return ActivityChange(
        account,
        *received,
        *sent,
        *bursts,
        gap is not None,
        gap or 0,
        since_in is not None,
        since_in or 0,
        since_out is not None,
        since_out or 0,
    )


# Each look-alike's B3 by hand.
LOOK_ALIKES: dict[str, ActivityChange] = {
    # Salaries every 14 days at 09:00 from t - 89 d, the last at t - 5 d; each followed by
    # three outflows 20 min, 27 h and 3 d later, so no two are within 24 hours. The newest
    # outflow is the bill of t - 2 d 09:00.
    "a_split": _row(
        "a_split", (1, 2, 7), (3, 7, 21), (1, 1, 1, 1), 14 * D, 5 * D - 9 * H, 2 * D - 9 * H
    ),
    # Two inflows 20 d 12 h apart, the last at t - 8 d 21:00; outflows at t - 27 d 20:00 and
    # t - 26 d 10:00, 14 hours apart, then t - 13 d and t - 8 d 22:00.
    "a_saver": _row(
        "a_saver",
        (0, 2, 3),
        (0, 4, 6),
        (1, 1, 1, 2),
        20 * D + 12 * H,
        8 * D - 21 * H,
        8 * D - 22 * H,
    ),
    # One refund in 90 days; three card payments, 22 hours and then 7 hours apart.
    "a_sleeper": _row(
        "a_sleeper", (0, 1, 1), (0, 3, 3), (1, 1, 1, 2), None, 12 * D - 8 * H, 11 * D - 20 * H
    ),
    # Rents at t - 4 d 10:00, t - 3 d 19:00 and t - 2 d 08:00: the last two 13 hours apart.
    "a_landlord": _row(
        "a_landlord", (3, 3, 9), (1, 2, 3), (1, 2, 1, 1), 13 * H, 2 * D - 8 * H, D - 15 * H
    ),
    # Two allowances 14 days apart in the 30 days; tuition and cafe 26 hours apart.
    "a_student": _row(
        "a_student", (1, 2, 3), (1, 3, 4), (1, 1, 1, 1), 14 * D, 6 * D - 19 * H, 5 * D - 12 * H
    ),
    # One payroll in 30 days; card at t - 3 d 12:00 and rent 22 hours later.
    "a_newjob": _row(
        "a_newjob", (1, 1, 4), (2, 3, 6), (1, 1, 1, 2), None, 3 * D - 9 * H, 2 * D - 10 * H
    ),
    # The two victims of one second count together in an hour, and give no gap between them:
    # the shortest is from that second to x_vic3's, 54 hours later. The ATM and the card
    # payment are 21 hours apart.
    "a_mule": _row(
        "a_mule",
        (3, 4, 6),
        (2, 4, 6),
        (2, 2, 1, 2),
        54 * H,
        4 * D - 20 * H,
        4 * D - 20 * H - 30 * M,
    ),
    # Gathers at t - 6 d 14:09, t - 5 d 10:00, t - 4 d 20:30 and t - 3 d 16:00: the closest
    # 19 h 30 min apart. Sends 14 hours apart.
    "a_collector": _row(
        "a_collector",
        (4, 4, 4),
        (2, 2, 2),
        (1, 2, 1, 2),
        19 * H + 30 * M,
        3 * D - 16 * H,
        D - 2 * H,
    ),
    "a_newjob_old": _row("a_newjob_old", (0, 0, 0), (0, 0, 0), (0, 0, 0, 0), None, None, None),
}


@pytest.fixture(scope="module")
def catalogue() -> ReferenceGraph:
    return fixtures.catalogue().graph()


@pytest.mark.parametrize("account", LOOK_ALIKES)
def test_each_look_alike_reads_as_worked_out_by_hand(
    catalogue: ReferenceGraph, account: str
) -> None:
    assert _rows(catalogue, [account]) == (LOOK_ALIKES[account],)


def test_a_request_answers_each_account_as_it_answers_it_alone(
    catalogue: ReferenceGraph,
) -> None:
    accounts = list(LOOK_ALIKES)
    assert _rows(catalogue, accounts) == tuple(LOOK_ALIKES[a] for a in sorted(accounts))


def test_the_windows_count_from_t_minus_w_to_just_before_t() -> None:
    w = WorldBuilder()
    w.customer("edge", fixtures.JOINED)
    w.customer("old", fixtures.JOINED)
    w.external("x_edge", fixtures.OPENED)
    for payment_id, ts in (
        ("x_in_90d", T - LOOKBACK_MS),
        ("x_in_before_90d", T - LOOKBACK_MS - 1),
        ("x_in_30d", T - 30 * D),
        ("x_in_30d_30m", T - 30 * D + 30 * M),
        ("x_in_20d", T - 20 * D),
        ("x_in_20d_1h", T - 20 * D + H),
        ("x_in_7d", T - 7 * D),
        ("x_in_t", T),
    ):
        w.pay(payment_id, "x_edge", "a_edge", ts, 10.0, "ach")
    w.pay("x_out_90d", "a_edge", "x_edge", T - LOOKBACK_MS, 10.0, "ach")
    w.pay("x_in_60d", "x_edge", "a_old", T - 60 * D, 10.0, "ach")
    # In: exactly t - 7 d counts over 7 days; t - 30 d, 30 minutes later, t - 20 d, an hour
    # later and t - 7 d over 30 days; t - 90 d too over 90; a millisecond before t - 90 d and
    # t itself never. The two from exactly t - 30 d are in the bursts' list: an hour holds
    # both, and their 30 minutes are the shortest gap; the two of t - 20 d are exactly an
    # hour apart, so no hour holds both. The latest inflow is exactly 7 days before t. Out:
    # one payment, exactly 90 days before t, in no burst's 30 days. a_old's only payment is
    # an inflow 60 days before t: it counts over 90 days alone, is in no burst's list, so
    # gives no gap, and is its latest inflow, 60 days before t, which reads the whole
    # lookback and not only the last 30 days.
    edge = _row("a_edge", (1, 5, 6), (0, 0, 1), (2, 2, 0, 0), 30 * M, 7 * D, LOOKBACK_MS)
    old = _row("a_old", (0, 0, 1), (0, 0, 0), (0, 0, 0, 0), None, 60 * D, None)
    assert _rows(w.graph(), ["a_edge", "a_old"]) == (edge, old)


def test_the_bursts_lists_are_cut_at_a_whole_second() -> None:
    w = WorldBuilder()
    w.customer("burst", fixtures.JOINED)
    w.external("x_burst", fixtures.OPENED)
    for i in range(300):
        minute = 43 if i == 44 else i
        at = T - 10 * D + minute * M
        w.pay(f"x_burst_in{i:03d}", "x_burst", "a_burst", at, 1.0, "ach")
        w.pay(f"x_burst_out{i:03d}", "a_burst", "x_burst", at, 1.0, "ach")
    # 300 inflows and 300 outflows, one of each a minute from t - 10 days, the 45th of each in
    # the minute of the 44th. The newest 256 of each list end in that tie: the newest left out
    # is at minute 43, so both of minute 43 are dropped and minutes 45 to 299 kept. An hour
    # from any of them holds 60, 24 hours all 255, each way; the inflows are a minute apart.
    # The counts read every payment, and so do the times since the last, 299 minutes after
    # the first.
    expected = _row(
        "a_burst",
        (0, 300, 300),
        (0, 300, 300),
        (60, 255, 60, 255),
        M,
        10 * D - 299 * M,
        10 * D - 299 * M,
    )
    assert _rows(w.graph(), ["a_burst"]) == (expected,)


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"t_ms": T + 1, "account_ids": ["a_mule"]}, "invalid_parameters"),
        ({"t_ms": T, "account_ids": []}, "invalid_parameters"),
        ({"t_ms": T, "account_ids": ["a_mule", "a_mule"]}, "duplicate_account"),
        ({"t_ms": T, "account_ids": ["a_nobody"]}, "unknown_account"),
        ({"t_ms": T, "account_ids": ["a_mule", "a_late"]}, "invisible_account"),
        ({"t_ms": T + 1, "account_ids": ["a_mule", "a_mule"]}, "invalid_parameters"),
        ({"t_ms": T, "account_ids": ["a_nobody", "a_nobody"]}, "duplicate_account"),
        ({"t_ms": T, "account_ids": ["a_late", "a_nobody"]}, "unknown_account"),
    ],
    ids=[
        "no whole second",
        "no account",
        "an account twice",
        "an unknown account",
        "opened at t",
        "no whole second before an account twice",
        "an account twice before an unknown one",
        "an unknown account before one opened at t",
    ],
)
def test_the_query_refuses_a_request_it_cannot_answer(
    params: dict[str, object], status: str
) -> None:
    w = fixtures.catalogue()
    w.customer("late", T)
    rows = w.graph().run(READ_ACTIVITY_CHANGE, params)
    assert rows == [{"status": status}]
    with pytest.raises(QueryRefused, match=status):
        read_activity_change.parse(rows, ["a_mule"])


# The leak tests over B3.


def _printed(graph: ReferenceGraph) -> str:
    """The answer for every account that exists at t, as the bytes TigerGraph would send."""
    params = read_activity_change.params(T, _existing(graph))
    return json.dumps(graph.run(READ_ACTIVITY_CHANGE, params))


def test_erasing_every_label_column_changes_no_answer() -> None:
    tables = fixtures.catalogue().tables()
    erased = fixtures.erase_labels(tables)
    assert not erased["Account"].equals(tables["Account"])
    assert _printed(ReferenceGraph(erased)) == _printed(ReferenceGraph(tables))


def test_moving_dates_of_record_before_the_lookback_changes_no_answer() -> None:
    tables = fixtures.catalogue().tables()
    moved = fixtures.move_dates_of_record(tables, T, LOOKBACK_MS)
    assert not moved["Account"].equals(tables["Account"])
    assert _printed(ReferenceGraph(moved)) == _printed(ReferenceGraph(tables))


def test_swapping_the_ids_of_same_second_payments_changes_no_value() -> None:
    w = fixtures.catalogue()
    # Two more inflows of a_mule in one second, and two outflows of one second.
    w.external("x_tie1", fixtures.OPENED)
    w.pay("x_tie_b", "x_tie1", "a_mule", T - 2 * D + 5 * H, 70.0, "ach")
    w.pay("x_tie_a", "x_tie1", "a_mule", T - 2 * D + 5 * H, 30.0, "ach")
    w.pay("x_tie_d", "a_mule", "x_tie1", T - D + 9 * H, 5.0, "ach")
    w.pay("x_tie_c", "a_mule", "x_mule_atm", T - D + 9 * H, 5.0, "cash")
    tables = w.tables()
    swapped, original = fixtures.swap_same_second_ids(tables)
    assert any(new != old for new, old in original.items())
    assert _printed(ReferenceGraph(swapped)) == _printed(ReferenceGraph(tables))


def test_a_self_payment_is_read_nowhere() -> None:
    w = fixtures.catalogue()
    before = _printed(w.graph())
    w.pay("x_mule_self", "a_mule", "a_mule", T - 6 * D + 14 * H + M, 900.0, "internal")
    assert _printed(w.graph()) == before
