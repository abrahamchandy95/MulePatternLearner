"""read_payment_events' parameter builder and row parser, on answers written by hand in
TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set as a list
of vertices with their projected fields under "attributes", and a printed map as an object.

The parser builds each scored account's payment graph from the accounts and payments printed
once and the ids printed per scored account, and refuses what the query cannot give.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries import (
    READ_PAYMENT_EVENTS,
    VERSIONS,
    read_payment_events,
)
from mule_pattern_learner.tigergraph.queries.read_payment_events import (
    EventAccount,
    OwnedBy,
    PaymentEvent,
    PaymentGraph,
)
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS, LOOKBACK_MS, QueryRefused

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday
HOUR_MS = 3_600_000
MINUTE_MS = 60_000


def test_a_request_takes_a_monday_its_accounts_and_the_hubs_each_once() -> None:
    """The accounts go as a list of ids, which the query checks itself; the hubs as a set of
    vertices, in TigerGraph's JSON for SET<VERTEX<Account>>."""
    assert read_payment_events.params(T_MS, ["a_02", "a_01"], ["h_01"]) == {
        "t_ms": T_MS,
        "account_ids": ["a_02", "a_01"],
        "hub_accounts": [{"id": "h_01"}],
    }
    assert read_payment_events.params(T_MS, ["a_01"], [])["hub_accounts"] == []
    with pytest.raises(ValueError, match="Monday"):
        read_payment_events.params(T_MS + DAY_MS, ["a_01"], [])
    with pytest.raises(ValueError, match="repeats an account"):
        read_payment_events.params(T_MS, ["a_01", "a_01"], [])
    with pytest.raises(ValueError, match="at least one account"):
        read_payment_events.params(T_MS, [], [])
    with pytest.raises(ValueError, match="repeats an id"):
        read_payment_events.params(T_MS, ["a_01"], ["h_01", "h_01"])


# A mule a_01 of customer q_01 and a saver a_02 of customer q_02 with no payment in the
# lookback, on t = 2024-04-01. Times are before t by their age:
# - p_01, v_01 to a_01 on 2024-03-30 10:00, 38 hours before t; v_01 had sent nothing before
#   and a_01 had received nothing before, so no gap and no pair.
# - p_03, the external v_02 to a_01 at 10:05: a_01 last received at 10:00, 5 minutes before.
# - p_02, v_01 to a_01 again at 10:20: one earlier pair payment, p_01, 20 minutes before,
#   which is also v_01's latest payment sent; a_01 last received p_03, 15 minutes before.
# - p_04, a_01 to o_01 at 10:30, USD 662 by ach: no earlier payment sent by a_01.
# - p_05, a_01 to the biller h_01, a hub, on 2024-03-31 09:00 by card: a_01 last sent p_04,
#   22 hours 30 minutes before; the receiver's gap is absent at the hub.
# - p_06, v_01 to w_01 at 2024-03-30 11:00 by Zelle: v_01 is expanded as a hop-1 account of
#   a_01, w_01 is a hop-2 account; v_01 last sent p_02, 40 minutes before.
# a_01's customer q_01 owns b_02 too, another account in its subgraph; v_01 is expanded, so
# its ownership edge to its customer q_05 is in it; o_01 and v_02 have no owner printed.
P_01 = PaymentEvent(
    payment_id="p_01",
    sender_id="v_01",
    receiver_id="a_01",
    amount=50_000,
    rail="ach",
    age_ms=38 * HOUR_MS,
    pair_prior_count=0,
    pair_gap_present=False,
    pair_gap_ms=0,
    pair_age_present=False,
    pair_age_ms=0,
    sender_gap_present=False,
    sender_gap_ms=0,
    receiver_gap_present=False,
    receiver_gap_ms=0,
)
P_02 = replace(
    P_01,
    payment_id="p_02",
    amount=45_000,
    age_ms=38 * HOUR_MS - 20 * MINUTE_MS,
    pair_prior_count=1,
    pair_gap_present=True,
    pair_gap_ms=20 * MINUTE_MS,
    pair_age_present=True,
    pair_age_ms=20 * MINUTE_MS,
    sender_gap_present=True,
    sender_gap_ms=20 * MINUTE_MS,
    receiver_gap_present=True,
    receiver_gap_ms=15 * MINUTE_MS,
)
P_03 = replace(
    P_01,
    payment_id="p_03",
    sender_id="v_02",
    rail="zelle",
    age_ms=38 * HOUR_MS - 5 * MINUTE_MS,
    receiver_gap_present=True,
    receiver_gap_ms=5 * MINUTE_MS,
)
P_04 = replace(
    P_01,
    payment_id="p_04",
    sender_id="a_01",
    receiver_id="o_01",
    amount=66_200,
    age_ms=38 * HOUR_MS - 30 * MINUTE_MS,
)
P_05 = replace(
    P_04,
    payment_id="p_05",
    receiver_id="h_01",
    amount=3_800,
    rail="card",
    age_ms=15 * HOUR_MS,
    sender_gap_present=True,
    sender_gap_ms=22 * HOUR_MS + 30 * MINUTE_MS,
)
P_06 = replace(
    P_01,
    payment_id="p_06",
    receiver_id="w_01",
    rail="zelle",
    age_ms=37 * HOUR_MS,
    sender_gap_present=True,
    sender_gap_ms=40 * MINUTE_MS,
)
PAYMENTS = [P_01, P_02, P_03, P_04, P_05, P_06]
ACCOUNTS = [
    EventAccount(account_id, account_id == "v_02")
    for account_id in ("a_01", "a_02", "b_02", "h_01", "o_01", "v_01", "v_02", "w_01")
]
MEMBERS = {
    "a_01": ["a_01", "v_01", "v_02", "o_01", "h_01", "w_01", "b_02"],
    "a_02": ["a_02"],
}
EVENTS = {"a_01": ["p_01", "p_02", "p_03", "p_04", "p_05", "p_06"]}
OWNED = {
    "a_01": [OwnedBy("v_01", "q_05"), OwnedBy("a_01", "q_01"), OwnedBy("b_02", "q_01")],
    "a_02": [OwnedBy("a_02", "q_02")],
}


def _vertices(kind: str, rows: list[Any], key: str) -> list[dict[str, Any]]:
    return [{"v_id": getattr(row, key), "v_type": kind, "attributes": vars(row)} for row in rows]


def _answer(
    *,
    accounts: list[EventAccount] = ACCOUNTS,
    payments: list[PaymentEvent] = PAYMENTS,
    members: dict[str, list[str]] = MEMBERS,
    events: dict[str, list[str]] = EVENTS,
    owned: dict[str, list[OwnedBy]] = OWNED,
) -> list[dict[str, Any]]:
    """An answer as TigerGraph prints it."""
    return [
        {"status": "ok", "version": VERSIONS[READ_PAYMENT_EVENTS]},
        {"accounts": _vertices("Account", accounts, "account_id")},
        {"payments": _vertices("Payment_Transaction", payments, "payment_id")},
        {"subgraph_accounts": members},
        {"subgraph_payments": events},
        {"owned_by": {key: [vars(edge) for edge in edges] for key, edges in owned.items()}},
    ]


def test_an_answer_parses_each_scored_accounts_graph_sorted_by_id() -> None:
    mule, saver = read_payment_events.parse(_answer(), ["a_02", "a_01"])
    assert mule == PaymentGraph(
        "a_01",
        tuple(row for row in ACCOUNTS if row.account_id in MEMBERS["a_01"]),
        tuple(PAYMENTS),
        (OwnedBy("a_01", "q_01"), OwnedBy("b_02", "q_01"), OwnedBy("v_01", "q_05")),
    )
    assert [row.account_id for row in mule.accounts] == [
        "a_01",
        "b_02",
        "h_01",
        "o_01",
        "v_01",
        "v_02",
        "w_01",
    ]
    # The saver's graph holds itself alone: no payment, and its own ownership edge.
    assert saver == PaymentGraph(
        "a_02", (EventAccount("a_02", False),), (), (OwnedBy("a_02", "q_02"),)
    )


def test_a_scored_account_with_no_owner_has_no_ownership_edge() -> None:
    owned = {"a_01": OWNED["a_01"]}
    _, saver = read_payment_events.parse(_answer(owned=owned), ["a_01", "a_02"])
    assert saver.owned_by == ()


def test_a_status_row_or_another_version_is_refused() -> None:
    for status in (
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
        "invisible_hub",
    ):
        with pytest.raises(QueryRefused, match=status) as refused:
            read_payment_events.parse([{"status": status}], ["a_01"])
        assert refused.value.status == status
    with pytest.raises(ValueError, match="mule install"):
        read_payment_events.parse(
            [{"status": "ok", "version": "read_payment_events-0123456789ab"}], []
        )


@pytest.mark.parametrize(
    ("answer", "asked", "message"),
    [
        (_answer(), ["a_01"], "printed subgraph_accounts of"),
        (_answer(members={"a_01": MEMBERS["a_01"]}), ["a_01", "a_02"], "answered the accounts"),
        (
            _answer(members={**MEMBERS, "a_02": ["b_02"]}),
            ["a_01", "a_02"],
            "left a_02 out of its own subgraph",
        ),
        (
            _answer(members={**MEMBERS, "a_02": ["a_02", "z_09"]}),
            ["a_01", "a_02"],
            "an account or payment it lacks",
        ),
        (
            _answer(events={**EVENTS, "a_02": ["p_09"]}),
            ["a_01", "a_02"],
            "an account or payment it lacks",
        ),
        (
            _answer(events={**EVENTS, "a_02": ["p_01"]}),
            ["a_01", "a_02"],
            "p_01 without its ends",
        ),
        (
            _answer(owned={**OWNED, "a_02": [OwnedBy("b_02", "q_01")]}),
            ["a_01", "a_02"],
            "an owner of no account of it",
        ),
        (
            _answer(owned={**OWNED, "a_02": [OwnedBy("a_02", "q_02"), OwnedBy("a_02", "q_03")]}),
            ["a_01", "a_02"],
            "two owners",
        ),
        (
            _answer(accounts=[*ACCOUNTS, EventAccount("z_09", False)]),
            ["a_01", "a_02"],
            "of no subgraph",
        ),
        (
            _answer(accounts=[*ACCOUNTS, ACCOUNTS[0]]),
            ["a_01", "a_02"],
            "an account or a payment twice",
        ),
        (
            _answer(members={**MEMBERS, "a_02": ["a_02", "a_02"]}),
            ["a_01", "a_02"],
            "repeats an id",
        ),
    ],
    ids=[
        "a subgraph not asked for",
        "a subgraph missing",
        "the scored account left out",
        "an account without its row",
        "a payment without its row",
        "a payment without its ends",
        "an owner of an account outside",
        "an account with two owners",
        "an account of no subgraph",
        "an account twice",
        "a member twice",
    ],
)
def test_an_answer_that_is_no_set_of_subgraphs_is_refused(
    answer: list[dict[str, Any]], asked: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_payment_events.parse(answer, asked)


def test_a_subgraph_with_more_payments_than_its_caps_is_refused() -> None:
    """32 payments each way of the scored account and 32 of each of its at most 64 hop-1
    accounts: 64 x 33 = 2,112 at most."""
    assert read_payment_events.MOST_PAYMENTS == 2_112
    many = [replace(P_04, payment_id=f"p_{index:05}") for index in range(2_113)]
    answer = _answer(
        payments=many,
        members={"a_01": ["a_01", "o_01"]},
        events={"a_01": [row.payment_id for row in many]},
        owned={},
        accounts=[EventAccount("a_01", False), EventAccount("o_01", False)],
    )
    with pytest.raises(ValueError, match="more payments than its caps"):
        read_payment_events.parse(answer, ["a_01"])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(P_04, receiver_id="a_01"), "a self-payment"),
        (replace(P_04, rail="wire"), "the rail 'wire'"),
        (replace(P_04, age_ms=0), "an age outside the lookback"),
        (replace(P_04, age_ms=LOOKBACK_MS + 1), "an age outside the lookback"),
        (replace(P_02, pair_prior_count=0), "pair gaps that disagree"),
        (replace(P_02, pair_age_present=False, pair_age_ms=0), "pair gaps that disagree"),
        (replace(P_01, pair_prior_count=2), "pair gaps that disagree"),
        (replace(P_04, sender_gap_ms=5), "an absent sender_gap that is not 0"),
        (replace(P_04, receiver_gap_ms=5), "an absent receiver_gap that is not 0"),
        (
            replace(P_02, pair_age_ms=LOOKBACK_MS - P_02.age_ms + 1),
            "a pair_age before the lookback",
        ),
        (replace(P_02, pair_gap_ms=P_02.pair_age_ms + 1), "a latest pair payment before"),
        (replace(P_02, sender_gap_ms=P_02.pair_gap_ms + 1), "a sender gap longer than its pair"),
        (
            replace(P_02, receiver_gap_ms=P_02.pair_gap_ms + 1),
            "a receiver gap longer than its pair",
        ),
    ],
    ids=[
        "a self-payment",
        "an unknown rail",
        "an age of 0",
        "an age beyond the lookback",
        "pair gaps without a pair",
        "a pair age missing",
        "a pair without gaps",
        "an absent sender gap",
        "an absent receiver gap",
        "a pair age beyond the lookback",
        "a pair gap beyond its age",
        "a sender gap beyond the pair's",
        "a receiver gap beyond the pair's",
    ],
)
def test_a_payment_row_the_query_cannot_give_is_refused(row: PaymentEvent, message: str) -> None:
    payments = [row if other.payment_id == row.payment_id else other for other in PAYMENTS]
    members = {**MEMBERS, "a_01": [*MEMBERS["a_01"]]}
    with pytest.raises(ValueError, match=message):
        read_payment_events.parse(_answer(payments=payments, members=members), ["a_01", "a_02"])


def test_a_payment_rail_may_be_any_of_the_contracts_and_zelle() -> None:
    assert read_payment_events.RAILS == (
        "ach",
        "card",
        "cash",
        "check",
        "internal",
        "unknown",
        "zelle",
    )
    for rail in read_payment_events.RAILS:
        payments = [replace(P_04, rail=rail) if row is P_04 else row for row in PAYMENTS]
        parsed = read_payment_events.parse(_answer(payments=payments), ["a_01", "a_02"])
        assert parsed[0].payments[3].rail == rail
