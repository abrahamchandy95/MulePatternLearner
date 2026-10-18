"""The reference answer of read_payment_events, B4, on worlds worked out by hand from
docs/features.md ("What one example's subgraph holds", items 1 to 5 and 10, and "Payment
edges", attributes 1 to 14): one subgraph with every row by hand, a scored hub, the caps and
their ties, the customer's other accounts, the hub parameter, the refusals, and a
self-payment.

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it. Amounts are in cents and times in
milliseconds. The leak tests over B4 to B6 together are in test_leaks.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import READ_PAYMENT_EVENTS, read_payment_events
from mule_pattern_learner.tigergraph.queries.read_payment_events import (
    EventAccount,
    OwnedBy,
    PaymentEvent,
    PaymentGraph,
)
from mule_pattern_learner.tigergraph.queries.rows import QueryRefused

D, H = DAY_MS, HOUR_MS
JOINED, OPENED = fixtures.JOINED, fixtures.OPENED
HUBS = ("x_shop", "g_loans", "g_biz")


def _graphs(
    graph: ReferenceGraph,
    accounts: Sequence[str],
    hubs: Sequence[str] = HUBS,
    t: int = T,
) -> tuple[PaymentGraph, ...]:
    found = graph.run(READ_PAYMENT_EVENTS, read_payment_events.params(t, accounts, hubs))
    return read_payment_events.parse(found, accounts)


def _world() -> WorldBuilder:
    """p_s's checking account a_s, scored, and the accounts around it.

    - p_s also holds a credit card a_s_card, last used the day before t at 13:00; savings
      a_s_save, credited 5 days before t; a_s_closed, closed 30 days before t and last used 40
      days before t; a_s_idle, never used; a_s_gone, closed 100 days before t, so its ownership
      link ended before the lookback; and a_s_new, opened 3 days after t.
    - Its employer, another bank's x_emp, pays it 2,000.00 by ACH 95 days before t, exactly
      90 days before t, and 40 and 10 days before t at 09:00; in the same millisecond as the
      last it pays p_e's a_e 1,500.00.
    - It pays p_c's a_c 300.00 by Zelle 9 days before t at 12:00 and 150.00 3 days before t at
      12:00, and once more exactly at t; 60.00 by Zelle to g_biz, p_biz's business account
      and a hub, 4 days before t; 19.99 by card at x_shop, a hub, 2 days before t at 18:00.
    - The bank's loan ledger g_loans, a hub, pays x_shop 7.00 15 days before t and refunds a_s
      12.00 the day before t.
    - a_c is paid 50.00 by p_d's a_d 30 and 20 days before t, and pays a_d 100.00 8 days
      before t.
    """
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.account("a_s_card", "p_s", JOINED, account_type="credit")
    w.account("a_s_save", "p_s", JOINED)
    w.account("a_s_closed", "p_s", JOINED, closed_ms=T - 30 * D)
    w.account("a_s_idle", "p_s", JOINED)
    w.account("a_s_gone", "p_s", JOINED, closed_ms=T - 100 * D)
    w.account("a_s_new", "p_s", T + 3 * D)
    for name in ("c", "d", "e"):
        w.customer(name, JOINED)
    w.party("p_biz", JOINED)
    w.account("g_biz", "p_biz", JOINED)
    for account in ("x_emp", "x_shop", "x_bank"):
        w.external(account, OPENED)
    w.account("g_loans", None, OPENED, account_type="gl")
    w.pay("x_pay0", "x_emp", "a_s", T - 95 * D, 2_000.0, "ach")
    w.pay("x_pay_edge", "x_emp", "a_s", T - 90 * D, 2_000.0, "ach")
    w.pay("x_pay1", "x_emp", "a_s", T - 40 * D + 9 * H, 2_000.0, "ach")
    w.pay("x_pay2", "x_emp", "a_s", T - 10 * D + 9 * H, 2_000.0, "ach")
    w.pay("x_e1", "x_emp", "a_e", T - 10 * D + 9 * H, 1_500.0, "ach")
    w.pay("z_s1", "a_s", "a_c", T - 9 * D + 12 * H, 300.0, "zelle")
    w.pay("z_s2", "a_s", "a_c", T - 3 * D + 12 * H, 150.0, "zelle")
    w.pay("z_s_at", "a_s", "a_c", T, 99.0, "zelle")
    w.pay("z_s_biz", "a_s", "g_biz", T - 4 * D, 60.0, "zelle")
    w.pay("x_loan_shop", "g_loans", "x_shop", T - 15 * D, 7.0, "ach")
    w.pay("x_s3", "a_s", "x_shop", T - 2 * D + 18 * H, 19.99, "card")
    w.pay("x_refund", "g_loans", "a_s", T - D, 12.0, "ach")
    w.pay("z_d0", "a_d", "a_c", T - 30 * D, 50.0, "zelle")
    w.pay("z_d1", "a_d", "a_c", T - 20 * D, 50.0, "zelle")
    w.pay("z_c1", "a_c", "a_d", T - 8 * D, 100.0, "zelle")
    w.pay("x_card1", "a_s_card", "x_shop", T - D + 13 * H, 20.0, "card")
    w.pay("x_int1", "x_bank", "a_s_save", T - 5 * D, 1.23, "ach")
    w.pay("x_old1", "a_s_closed", "x_shop", T - 40 * D, 10.0, "card")
    return w


def _event(
    payment_id: str, sender: str, receiver: str, amount: int, rail: str, age: int
) -> PaymentEvent:
    """A payment's row with no earlier payment of its pair, its sender or its receiver."""
    return PaymentEvent(
        payment_id, sender, receiver, amount, rail, age, 0, False, 0, False, 0, False, 0, False, 0
    )


# Every payment row of a_s's subgraph by hand. A row depends on the payment, t and the hubs
# alone.
ROWS = {
    # Exactly 90 days before t, so in the lookback; x_pay0, 95 days before t, is not, so
    # nothing of its pair, its sender or its receiver came before it.
    "x_pay_edge": _event("x_pay_edge", "x_emp", "a_s", 200_000, "ach", 90 * D),
    # After x_pay_edge: (t - 40 d + 9 h) - (t - 90 d) = 50 d + 9 h, for its pair, its sender
    # and its receiver.
    "x_pay1": replace(
        _event("x_pay1", "x_emp", "a_s", 200_000, "ach", 40 * D - 9 * H),
        pair_prior_count=1,
        pair_gap_present=True,
        pair_gap_ms=50 * D + 9 * H,
        pair_age_present=True,
        pair_age_ms=50 * D + 9 * H,
        sender_gap_present=True,
        sender_gap_ms=50 * D + 9 * H,
        receiver_gap_present=True,
        receiver_gap_ms=50 * D + 9 * H,
    ),
    # Two pair payments before it: the latest 30 days earlier, the earliest
    # (t - 10 d + 9 h) - (t - 90 d) = 80 d + 9 h earlier. x_e1, of the same millisecond, is not
    # strictly earlier, so x_emp's latest payment before it is x_pay1, 30 days earlier.
    "x_pay2": replace(
        _event("x_pay2", "x_emp", "a_s", 200_000, "ach", 10 * D - 9 * H),
        pair_prior_count=2,
        pair_gap_present=True,
        pair_gap_ms=30 * D,
        pair_age_present=True,
        pair_age_ms=80 * D + 9 * H,
        sender_gap_present=True,
        sender_gap_ms=30 * D,
        receiver_gap_present=True,
        receiver_gap_ms=30 * D,
    ),
    # x_emp's first payment to a_e; its latest before it is x_pay1, 30 days earlier, x_pay2
    # being of the same millisecond.
    "x_e1": replace(
        _event("x_e1", "x_emp", "a_e", 150_000, "ach", 10 * D - 9 * H),
        sender_gap_present=True,
        sender_gap_ms=30 * D,
    ),
    # From the hub g_loans, so no sender gap; a_s last received x_pay2,
    # (t - 1 d) - (t - 10 d + 9 h) = 9 d - 9 h earlier.
    "x_refund": replace(
        _event("x_refund", "g_loans", "a_s", 1_200, "ach", D),
        receiver_gap_present=True,
        receiver_gap_ms=9 * D - 9 * H,
    ),
    # a_s's first payment sent; a_c last received z_d1, (t - 9 d + 12 h) - (t - 20 d) =
    # 11 d + 12 h earlier.
    "z_s1": replace(
        _event("z_s1", "a_s", "a_c", 30_000, "zelle", 9 * D - 12 * H),
        receiver_gap_present=True,
        receiver_gap_ms=11 * D + 12 * H,
    ),
    # To the hub g_biz, so no receiver gap; a_s last sent z_s1,
    # (t - 4 d) - (t - 9 d + 12 h) = 4 d + 12 h earlier.
    "z_s_biz": replace(
        _event("z_s_biz", "a_s", "g_biz", 6_000, "zelle", 4 * D),
        sender_gap_present=True,
        sender_gap_ms=4 * D + 12 * H,
    ),
    # z_s1, 6 days earlier, is its pair's and its receiver's latest; a_s last sent z_s_biz,
    # (t - 3 d + 12 h) - (t - 4 d) = 1 d + 12 h earlier.
    "z_s2": replace(
        _event("z_s2", "a_s", "a_c", 15_000, "zelle", 3 * D - 12 * H),
        pair_prior_count=1,
        pair_gap_present=True,
        pair_gap_ms=6 * D,
        pair_age_present=True,
        pair_age_ms=6 * D,
        sender_gap_present=True,
        sender_gap_ms=D + 12 * H,
        receiver_gap_present=True,
        receiver_gap_ms=6 * D,
    ),
    # 19.99 x 100 is 1,998.9999999999998 in doubles, so floor(19.99 x 100 + 0.5) = 1,999
    # cents; to the hub x_shop, so no receiver gap; a_s last sent z_s2,
    # (t - 2 d + 18 h) - (t - 3 d + 12 h) = 30 h earlier; its age is 2 d - 18 h = 30 h.
    "x_s3": replace(
        _event("x_s3", "a_s", "x_shop", 1_999, "card", 30 * H),
        sender_gap_present=True,
        sender_gap_ms=30 * H,
    ),
    "z_d0": _event("z_d0", "a_d", "a_c", 5_000, "zelle", 30 * D),
    # z_d0, 10 days earlier, is its pair's, its sender's and its receiver's latest.
    "z_d1": replace(
        _event("z_d1", "a_d", "a_c", 5_000, "zelle", 20 * D),
        pair_prior_count=1,
        pair_gap_present=True,
        pair_gap_ms=10 * D,
        pair_age_present=True,
        pair_age_ms=10 * D,
        sender_gap_present=True,
        sender_gap_ms=10 * D,
        receiver_gap_present=True,
        receiver_gap_ms=10 * D,
    ),
    # a_c's first payment sent, and a_d's first received.
    "z_c1": _event("z_c1", "a_c", "a_d", 10_000, "zelle", 8 * D),
    # Between two hubs: neither gap is read. It is in no subgraph of a_s.
    "x_loan_shop": _event("x_loan_shop", "g_loans", "x_shop", 700, "ach", 15 * D),
}
# The payments of a_s's subgraph.
OWN_ROWS = tuple(payment for payment in ROWS if payment != "x_loan_shop")
EXTERNAL = frozenset({"x_emp", "x_shop", "x_bank"})


def _accounts(*ids: str) -> tuple[EventAccount, ...]:
    return tuple(EventAccount(account, account in EXTERNAL) for account in sorted(ids))


def _rows(*ids: str) -> tuple[PaymentEvent, ...]:
    return tuple(ROWS[payment] for payment in sorted(ids))


def test_a_subgraph_holds_its_payments_neighbours_other_accounts_and_owners_by_hand() -> None:
    # s's payments in the lookback: x_pay_edge, x_pay1, x_pay2 and x_refund received, z_s1,
    # z_s_biz, z_s2 and x_s3 sent; x_pay0 is before the lookback and z_s_at at t. Its hop-1
    # accounts: x_emp and a_c, expanded, and the hubs x_shop, g_loans and g_biz, which are
    # not, so x_card1, x_old1 and x_loan_shop stay out, and g_biz, though owned, has no
    # ownership edge. x_emp adds x_e1 and its hop-2 account a_e; a_c adds z_d0,
    # z_d1 and z_c1 and a_d. p_s's other accounts that exist at t through a visible ownership
    # link: a_s_card, a_s_save, a_s_closed (its link ended 30 days before t) and a_s_idle, but
    # not a_s_gone or a_s_new; their payments are not expanded. Ownership edges: a_s and a_c to
    # their customers, and every account of p_s in the subgraph; a_d and a_e, hop-2 accounts
    # of other customers, have none.
    graph = _world().graph()
    expected = PaymentGraph(
        "a_s",
        _accounts(
            "a_s",
            "a_c",
            "x_emp",
            "x_shop",
            "g_loans",
            "g_biz",
            "a_d",
            "a_e",
            "a_s_card",
            "a_s_save",
            "a_s_closed",
            "a_s_idle",
        ),
        _rows(*OWN_ROWS),
        (
            OwnedBy("a_c", "p_c"),
            OwnedBy("a_s", "p_s"),
            OwnedBy("a_s_card", "p_s"),
            OwnedBy("a_s_closed", "p_s"),
            OwnedBy("a_s_idle", "p_s"),
            OwnedBy("a_s_save", "p_s"),
        ),
    )
    assert _graphs(graph, ["a_s"]) == (expected,)


def test_a_scored_hub_is_expanded_and_its_payments_rows_are_the_same() -> None:
    # Scored, the hub g_loans is expanded: x_loan_shop goes to the hub x_shop, which is not,
    # and x_refund makes a_s its hop-1 account, which is; a_s's payments bring a_c, g_biz and
    # x_emp. g_loans has no owner, and so no other accounts; a_s's ownership edge is that of
    # an expanded hop-1 account. Each payment's row is the one it has in a_s's subgraph.
    graph = _world().graph()
    expected = PaymentGraph(
        "g_loans",
        _accounts("g_loans", "a_s", "a_c", "g_biz", "x_emp", "x_shop"),
        _rows(
            "x_loan_shop",
            "x_refund",
            "z_s1",
            "z_s_biz",
            "z_s2",
            "x_s3",
            "x_pay_edge",
            "x_pay1",
            "x_pay2",
        ),
        (OwnedBy("a_s", "p_s"),),
    )
    assert _graphs(graph, ["g_loans"]) == (expected,)


def test_the_hubs_are_those_of_the_parameter_alone() -> None:
    # With no hub, x_shop, g_loans and g_biz are expanded too: x_shop adds x_card1, x_old1 and
    # x_loan_shop, whose senders a_s_card, a_s_closed and g_loans are in the subgraph already;
    # g_biz is owned, so it gains its ownership edge; and the gaps at their ends are read. A
    # hub the graph lacks is dropped, as TigerGraph drops it.
    graph = _world().graph()
    (found,) = _graphs(graph, ["a_s"], ["x_nowhere"])
    added = {"x_card1", "x_old1", "x_loan_shop"}
    assert {row.payment_id for row in found.payments} == set(OWN_ROWS) | added
    assert OwnedBy("g_biz", "p_biz") in found.owned_by
    rows = {row.payment_id: row for row in found.payments}
    # x_s3 at x_shop: of x_shop's payments received, x_card1, the day before t at 13:00, came
    # after it, x_old1 40 days before t and x_loan_shop (t - 2 d + 18 h) - (t - 15 d) =
    # 13 d + 18 h before it.
    expected = replace(ROWS["x_s3"], receiver_gap_present=True, receiver_gap_ms=13 * D + 18 * H)
    assert rows["x_s3"] == expected
    # x_refund from g_loans: g_loans last sent x_loan_shop, (t - 1 d) - (t - 15 d) = 14 d
    # before it.
    expected = replace(ROWS["x_refund"], sender_gap_present=True, sender_gap_ms=14 * D)
    assert rows["x_refund"] == expected
    # z_s_biz to g_biz: g_biz received nothing before it, so its receiver gap stays absent.
    assert rows["z_s_biz"] == ROWS["z_s_biz"]


def test_the_scored_accounts_lists_keep_the_newest_32_each_way_ties_by_id() -> None:
    # a_k sends 33 payments to x_k and receives 33 from it: 31 an hour apart from an hour
    # before t, then two in the second 40 hours before t, b 999 ms after a. The newest 31 and
    # a, the smaller id of the tied second, are kept; b is not, though it is the later.
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_k", OPENED)
    for prefix, sender, receiver in (("p", "a_k", "x_k"), ("q", "x_k", "a_k")):
        for i in range(31):
            w.pay(f"{prefix}{i:02d}", sender, receiver, T - (i + 1) * H, 1.0, "ach")
        w.pay(f"{prefix}_a", sender, receiver, T - 40 * H, 1.0, "ach")
        w.pay(f"{prefix}_b", sender, receiver, T - 40 * H + 999, 1.0, "ach")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    kept = {f"{prefix}{i:02d}" for prefix in "pq" for i in range(31)} | {"p_a", "q_a"}
    assert {row.payment_id for row in found.payments} == kept


def test_an_expanded_accounts_lists_keep_its_newest_16_each_way_ties_by_id() -> None:
    # a_k receives z_in from a_c2 20 days before t. a_c2 also sends 17 payments to x_sink: 15
    # an hour apart from an hour before t, then y_a and y_b in the second 30 hours before t, b
    # 999 ms after a; it receives 17 from x_src a day apart from a day before t. a_c2 keeps
    # y00 to y14 and y_a, and r00 to r15: y_b, z_in and r16 fall out of its lists, z_in
    # staying as a_k's own payment.
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.customer("c2", JOINED)
    w.external("x_sink", OPENED)
    w.external("x_src", OPENED)
    w.pay("z_in", "a_c2", "a_k", T - 20 * D, 1.0, "zelle")
    for i in range(15):
        w.pay(f"y{i:02d}", "a_c2", "x_sink", T - (i + 1) * H, 1.0, "ach")
    w.pay("y_a", "a_c2", "x_sink", T - 30 * H, 1.0, "ach")
    w.pay("y_b", "a_c2", "x_sink", T - 30 * H + 999, 1.0, "ach")
    for i in range(17):
        w.pay(f"r{i:02d}", "x_src", "a_c2", T - (i + 1) * D, 1.0, "ach")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    kept = {"z_in", "y_a"} | {f"y{i:02d}" for i in range(15)} | {f"r{i:02d}" for i in range(16)}
    assert {row.payment_id for row in found.payments} == kept
    assert [row.account_id for row in found.accounts] == ["a_c2", "a_k", "x_sink", "x_src"]


def _own_accounts(active: dict[str, int], idle: Sequence[str]) -> set[str]:
    """The accounts p_k keeps beside a_k, when it also owns the accounts of active, each with
    one payment at its time, and the accounts of idle, with none in the lookback."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    for account, ts in active.items():
        w.account(account, "p_k", JOINED)
        w.pay(f"x_{account}", account, "x_any", ts, 1.0, "ach")
    for account in idle:
        w.account(account, "p_k", JOINED)
    (found,) = _graphs(w.graph(), ["a_k"], [])
    return {row.account_id for row in found.accounts} - {"a_k"}


def test_the_customers_other_accounts_are_the_8_most_recently_active_ties_by_id() -> None:
    # Seven accounts active on the 7 days before t, and two active in the second 8 days before
    # t, o_8b 999 ms after o_8a: the 8th kept is o_8a, the smaller id of the tied second.
    active = {f"o_{day}": T - day * D for day in range(1, 8)}
    active |= {"o_8a": T - 8 * D, "o_8b": T - 8 * D + 999}
    assert _own_accounts(active, []) == set(active) - {"o_8b"}


def test_every_account_of_the_customer_in_the_subgraph_has_its_ownership_edge() -> None:
    # As above, but a_k pays x_any too: x_any, expanded, brings every account that paid it,
    # o_8b among them, which is p_k's but not one of the 8 kept. Every account of p_k in the
    # subgraph has its ownership edge.
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    active = {f"o_{day}": T - day * D for day in range(1, 8)}
    active |= {"o_8a": T - 8 * D, "o_8b": T - 8 * D + 999}
    for account, ts in active.items():
        w.account(account, "p_k", JOINED)
        w.pay(f"x_{account}", account, "x_any", ts, 1.0, "ach")
    w.pay("x_k", "a_k", "x_any", T - 9 * D, 1.0, "ach")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    assert {row.account_id for row in found.accounts} == {"a_k", "x_any"} | set(active)
    assert found.owned_by == tuple(OwnedBy(a, "p_k") for a in sorted({"a_k"} | set(active)))


def test_the_customers_accounts_with_no_payment_come_last_by_id() -> None:
    # Six active accounts named z1 to z6 and three with no payment in the lookback, a7, a8 and
    # a9: the active come first, then a7 and a8 by id, and a9 is left out, though a7 to a9
    # come first by id alone. a8's one payment is before the lookback, and a7's a
    # self-payment, which no query reads.
    active = {f"z{n}": T - n * D for n in range(1, 7)}
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    for account, ts in active.items():
        w.account(account, "p_k", JOINED)
        w.pay(f"x_{account}", account, "x_any", ts, 1.0, "ach")
    for account in ("a7", "a8", "a9"):
        w.account(account, "p_k", JOINED)
    w.pay("x_a7_self", "a7", "a7", T - D, 1.0, "internal")
    w.pay("x_a8_old", "a8", "x_any", T - 91 * D, 1.0, "ach")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    accounts = {row.account_id for row in found.accounts} - {"a_k"}
    assert accounts == set(active) | {"a7", "a8"}


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"t_ms": T + 1}, "invalid_parameters"),
        ({"account_ids": []}, "invalid_parameters"),
        ({"account_ids": ["a_s", "a_s"]}, "duplicate_account"),
        ({"account_ids": ["a_nobody"]}, "unknown_account"),
        ({"account_ids": ["a_s_new"]}, "invisible_account"),
        ({"hub_accounts": [{"id": "x_shop"}, {"id": "a_s_new"}]}, "invisible_hub"),
        ({"t_ms": T - D, "account_ids": ["a_s", "a_s"]}, "invalid_parameters"),
        ({"account_ids": ["a_nobody", "a_nobody"]}, "duplicate_account"),
        ({"account_ids": ["a_s_new", "a_nobody"]}, "unknown_account"),
        (
            {"account_ids": ["a_s_new"], "hub_accounts": [{"id": "a_s_new"}]},
            "invisible_account",
        ),
    ],
    ids=[
        "no Monday",
        "no account",
        "an account twice",
        "an unknown account",
        "opened after t",
        "a hub opened after t",
        "no Monday before an account twice",
        "an account twice before an unknown one",
        "an unknown account before one opened after t",
        "an account opened after t before a hub opened after t",
    ],
)
def test_the_query_refuses_a_request_it_cannot_answer(
    params: dict[str, object], status: str
) -> None:
    given = {"t_ms": T, "account_ids": ["a_s"], "hub_accounts": [], **params}
    rows = _world().graph().run(READ_PAYMENT_EVENTS, given)
    assert rows == [{"status": status}]
    with pytest.raises(QueryRefused, match=status):
        read_payment_events.parse(rows, ["a_s"])


def test_a_self_payment_is_read_nowhere() -> None:
    w = _world()
    before = _graphs(w.graph(), ["a_s", "a_c"])
    w.pay("x_s_self", "a_s", "a_s", T - 3 * D + 12 * H - 1, 5.0, "internal")
    w.pay("x_c_self", "a_c", "a_c", T - 9 * D, 5.0, "internal")
    assert _graphs(w.graph(), ["a_s", "a_c"]) == before
