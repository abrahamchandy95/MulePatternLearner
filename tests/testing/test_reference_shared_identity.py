"""The reference answer of read_shared_identity, B5, on worlds worked out by hand from
docs/features.md ("What one example's subgraph holds", items 3 and 5 to 9, and the customer,
device, IP, enrolment and seen tables) and docs/design.md ("Time", "Shared rules" and
"read_shared_identity (B5): the device and IP graph"): one subgraph with every row and edge by
hand, the inline leaf flags against read_hubs, the caps on links, seen nodes and co-users and
their order, the account a co-user brings, the hub parameter and the refusals.

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it. Times are in milliseconds. The leak
tests over B4 to B6 together are in test_leaks.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import (
    READ_HUBS,
    READ_SHARED_IDENTITY,
    read_hubs,
    read_shared_identity,
)
from mule_pattern_learner.tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityAccount,
    IdentityGraph,
    NodeRow,
    OwnerEnrolled,
    Seen,
)
from mule_pattern_learner.tigergraph.queries.rows import QueryRefused

D, H = DAY_MS, HOUR_MS
JOINED, OPENED = fixtures.JOINED, fixtures.OPENED
HUBS = ("x_shop", "g_s_biz")


def _graphs(
    graph: ReferenceGraph,
    accounts: Sequence[str],
    hubs: Sequence[str] = HUBS,
    t: int = T,
) -> tuple[IdentityGraph, ...]:
    found = graph.run(READ_SHARED_IDENTITY, read_shared_identity.params(t, accounts, hubs))
    return read_shared_identity.parse(found, accounts)


def _world() -> WorldBuilder:
    """p_s's checking account a_s, scored, and the customers, devices and IPs around it.

    - p_s, who joined 400 days before t, holds a_s, savings a_s_save, a credit card
      a_s_card, and a business account g_s_biz, a hub. Since joining it has been enrolled on
      its phone d_phone, its home IP i_home and the carrier IP i_crowd, on which 8 others, p_k1
      to p_k8, are enrolled too. It enrolled a tablet d_tab 45 days before t and a new phone
      d_new exactly 7 days before t.
    - a_s pays the hub x_shop 30.00 60 days before t with no device, its first payment in the
      lookback; 50.00 20 days before t from d_phone and i_home; p_c's a_c 200.00 5 days before
      t from d_new and i_crowd; and x_shop 100.00 2 days before t from d_ring, a device no one
      registered. a_s_card pays x_shop 9.00 from d_phone the day before t.
    - p_r's a_r pays another bank 25.00 from d_ring exactly 30 days before t, its one payment
      in the lookback.
    - p_h, of p_s's household, has been enrolled on i_home since joining and enrolled on
      d_phone 12 days before t; a_h pays x_shop 15.00 from i_home 40 days before t, its first
      payment in the lookback, and 80.00 from it 7 days before t.
    - p_old was enrolled on i_home and on its phone d_old from joining until 45 days before
      t, and its a_old never paid.
    - p_c, a_c's customer, has been enrolled on d_c since joining, and a_c pays its rent of
      900.00 from d_c 6 days before t, its first payment in the lookback, and 50.00 more 3
      days before t.
    """
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.account("a_s_save", "p_s", JOINED)
    w.account("g_s_biz", "p_s", JOINED)
    w.account("a_s_card", "p_s", JOINED, account_type="credit")
    for name in ("r", "h", "c", "old"):
        w.customer(name, JOINED)
    for account in ("x_shop", "x_out", "x_rent"):
        w.external(account, OPENED)
    w.device("p_s", "d_phone", JOINED)
    w.device("p_s", "d_tab", T - 45 * D)
    w.device("p_s", "d_new", T - 7 * D)
    w.ip("p_s", "i_home", JOINED)
    w.ip("p_s", "i_crowd", JOINED)
    for n in range(1, 9):
        w.party(f"p_k{n}", JOINED)
        w.ip(f"p_k{n}", "i_crowd", JOINED)
    w.ip("p_h", "i_home", JOINED)
    w.device("p_h", "d_phone", T - 12 * D)
    w.ip("p_old", "i_home", JOINED, T - 45 * D)
    w.device("p_old", "d_old", JOINED, T - 45 * D)
    w.device("p_c", "d_c", JOINED)
    w.pay("x_s0", "a_s", "x_shop", T - 60 * D, 30.0, "card")
    w.pay("x_s1", "a_s", "x_shop", T - 20 * D, 50.0, "card", device="d_phone", ip="i_home")
    w.pay("z_s2", "a_s", "a_c", T - 5 * D, 200.0, "zelle", device="d_new", ip="i_crowd")
    w.pay("x_s3", "a_s", "x_shop", T - 2 * D, 100.0, "card", device="d_ring")
    w.pay("x_r0", "a_r", "x_out", T - 30 * D, 25.0, "ach", device="d_ring")
    w.pay("x_card", "a_s_card", "x_shop", T - D, 9.0, "card", device="d_phone")
    w.pay("x_h0", "a_h", "x_shop", T - 40 * D, 15.0, "card", ip="i_home")
    w.pay("x_h1", "a_h", "x_shop", T - 7 * D, 80.0, "card", ip="i_home")
    w.pay("x_c1", "a_c", "x_rent", T - 6 * D, 900.0, "ach", device="d_c")
    w.pay("x_c2", "a_c", "x_rent", T - 3 * D, 50.0, "ach", device="d_c")
    return w


# a_s's device and IP graph by hand.
#
# B4's accounts: a_s; its hop-1 accounts a_c, expanded, and the hub x_shop; a_c's hop-2
# account x_rent; and p_s's other accounts a_s_card, a_s_save and g_s_biz, a hub.
# - D_s, p_s's kept links by newest evidence: d_phone (an unknown start, x_card the day before
#   t, from a credit card), d_new (its known start 7 days before t, then z_s2 5 days before
#   t) and d_tab (its known start 45 days before t, never used); i_crowd (z_s2) and i_home
#   (x_s1). S_s, seen on a_s's payments: d_ring, d_new, d_phone; i_crowd, i_home. i_crowd has
#   9 enrolled customers: a leaf. a_c keeps d_c, as p_c's link and as seen.
# - A start on a node is known when one of the customer's own payments came earlier, not
#   seen on it: x_s0 makes every start of p_s in the lookback known, and x_h0 p_h's link to
#   d_phone; x_r0, x_h0 and x_c1, first payments, are unknown first uses, and p_old's link
#   began before the lookback.
# - Crowds, by the one account rule: d_phone p_s and p_h, x_card's credit card counting for
#   no one; d_ring p_s and p_r; i_home p_s, p_h and p_old, whose link ended within the
#   lookback. So the co-users of a_s's nodes that are
#   no leaf are p_h, p_r and p_old, each with an account to bring; p_h and p_old, linked to
#   i_home, are customers of the subgraph, and p_r, seen only through payments, is not.
# - Sharing in the last 30 days: d_phone by p_s and p_h; d_ring by p_s (from x_s3, known, 2
#   days before t) and p_r (from x_r0, exactly 30 days before t, unknown: t - 90 days); i_home
#   by p_s and p_h, whose links are current, but not p_old's. Pairs: p_s and p_r began 2 days
#   before t, the later of their starts; p_s and p_h on d_phone 12 days before t and on i_home
#   at t - 90 days, so at t - 90 days, the earlier.
CUSTOMERS = (
    # a_c; d_c held; no co-user.
    CustomerRow("p_c", 1, 1, 0, 0, 0, 0, 0, 0),
    # a_h; d_phone held and new; i_home held; p_s, on a device and an IP, not new.
    CustomerRow("p_h", 1, 1, 1, 1, 0, 1, 0, 1),
    # a_old; its links ended 45 days before t: held in no 30 days, shared with no one.
    CustomerRow("p_old", 1, 0, 0, 0, 0, 0, 0, 0),
    # a_s, a_s_card, a_s_save and g_s_biz; d_phone, d_tab and d_new held, d_new new over 30
    # days but d_tab not; i_home and i_crowd held; p_r new, and p_h on a device and an IP.
    CustomerRow("p_s", 4, 3, 2, 1, 0, 2, 1, 1),
)
DEVICES = (
    # p_c alone: its link began before the lookback; a_c paid from it twice in 30 days.
    NodeRow("d_c", False, 1, 1, 0, 0, 1, 2, False),
    # p_s alone, from its link exactly 7 days before t, new over 7 and 30 days; z_s2.
    NodeRow("d_new", False, 1, 1, 1, 1, 1, 1, False),
    # p_s from before the lookback and p_h from 12 days before t; x_s1 from a_s and x_card
    # from a_s_card.
    NodeRow("d_phone", False, 2, 2, 0, 1, 2, 2, False),
    # p_s from 2 days before t, and p_r from x_r0, exactly 30 days before t; no one enrolled.
    NodeRow("d_ring", False, 2, 2, 1, 1, 2, 2, True),
    # p_s from 45 days before t; never used.
    NodeRow("d_tab", False, 1, 1, 0, 0, 0, 0, False),
)
IPS = (
    NodeRow("i_crowd", True, 0, 0, 0, 0, 0, 0, False),
    # p_s and p_h share it from before the lookback; p_old counts by its link alone. a_s and
    # a_h once each in 30 days: x_h0, 40 days before t, is older.
    NodeRow("i_home", False, 2, 3, 0, 0, 2, 2, False),
)
# Each link's fields: the time since its known start, and since the customer's last payment
# seen on the node.
S_NEW = (True, 7 * D, True, 5 * D)
S_PHONE = (False, 0, True, D)
S_TAB = (True, 45 * D, False, 0)
S_CROWD = (False, 0, True, 5 * D)
S_HOME = (False, 0, True, 20 * D)
H_PHONE = (True, 12 * D, False, 0)
H_HOME = (False, 0, True, 7 * D)
OLD_HOME = (False, 0, False, 0)
C_DC = (False, 0, True, 3 * D)
D_S = {"d_new": S_NEW, "d_phone": S_PHONE, "d_tab": S_TAB, "i_crowd": S_CROWD, "i_home": S_HOME}
EXPECTED = IdentityGraph(
    "a_s",
    CUSTOMERS,
    DEVICES,
    IPS,
    (
        IdentityAccount("a_h", False),
        IdentityAccount("a_old", False),
        IdentityAccount("a_r", False),
    ),
    (
        Enrolled("p_h", "d_phone", *H_PHONE),
        Enrolled("p_h", "i_home", *H_HOME),
        Enrolled("p_old", "i_home", *OLD_HOME),
        *(Enrolled("p_s", node, *fields) for node, fields in D_S.items()),
    ),
    (
        # a_c to p_c's kept link; the co-users' a_h and a_old to the nodes of a_s their
        # customers are linked to; a_s, a_s_card and a_s_save, the accounts of p_s, to D_s,
        # but not the hub g_s_biz.
        OwnerEnrolled("a_c", "d_c", *C_DC),
        OwnerEnrolled("a_h", "d_phone", *H_PHONE),
        OwnerEnrolled("a_h", "i_home", *H_HOME),
        OwnerEnrolled("a_old", "i_home", *OLD_HOME),
        *(OwnerEnrolled("a_s", node, *fields) for node, fields in D_S.items()),
        *(OwnerEnrolled("a_s_card", node, *fields) for node, fields in D_S.items()),
        *(OwnerEnrolled("a_s_save", node, *fields) for node, fields in D_S.items()),
    ),
    (
        # Uses, the time since a known first use, and since the last.
        Seen("a_c", "d_c", 2, False, 0, 3 * D),
        Seen("a_h", "i_home", 2, False, 0, 7 * D),
        Seen("a_r", "d_ring", 1, False, 0, 30 * D),
        Seen("a_s", "d_new", 1, True, 5 * D, 5 * D),
        Seen("a_s", "d_phone", 1, True, 20 * D, 20 * D),
        Seen("a_s", "d_ring", 1, True, 2 * D, 2 * D),
        Seen("a_s", "i_crowd", 1, True, 5 * D, 5 * D),
        Seen("a_s", "i_home", 1, True, 20 * D, 20 * D),
        Seen("a_s_card", "d_phone", 1, True, D, D),
    ),
)


def test_a_subgraph_holds_its_customers_nodes_co_users_and_edges_by_hand() -> None:
    assert _graphs(_world().graph(), ["a_s"]) == (EXPECTED,)


def test_the_inline_leaf_flags_equal_read_hubs() -> None:
    # Every account's graph: a node is printed as a leaf exactly when read_hubs lists it.
    graph = _world().graph()
    accounts = [a for a in graph.ids["Account"] if graph.accounts[a].opened_ms < T]
    leaves = read_hubs.parse(graph.run(READ_HUBS, read_hubs.params(T)))
    listed = {leaf.node_id for leaf in (*leaves.devices, *leaves.ips)}
    printed = {
        node.node_id: node.leaf
        for found in _graphs(graph, accounts)
        for node in (*found.devices, *found.ips)
    }
    assert listed == {"i_crowd"}
    assert {node for node, leaf in printed.items() if leaf} == listed
    assert set(printed) >= listed


def test_the_hubs_are_those_of_the_parameter_alone() -> None:
    # With no hub, g_s_biz, an account of p_s, carries its owner-enrolment edges to D_s. x_shop
    # is expanded too: another bank's account, it has no customer and sent nothing, so it
    # brings no node and carries no edge; a_h, now also its hop-2 account, is a co-user's
    # account already. A hub the graph lacks is dropped, as TigerGraph drops it.
    graph = _world().graph()
    business = tuple(OwnerEnrolled("g_s_biz", node, *fields) for node, fields in D_S.items())
    expected = replace(EXPECTED, owner_enrolled=(*EXPECTED.owner_enrolled, *business))
    assert _graphs(graph, ["a_s"], []) == (expected,)
    assert _graphs(graph, ["a_s"], ["x_nowhere"]) == (expected,)


def _links_world(evidence: dict[str, tuple[int | None, int | None]]) -> WorldBuilder:
    """p_k's account a_k, its one payment 60 days before t with no device, and a device link of
    p_k per entry of evidence: the link's start, inside the lookback when given and else
    before it, and the time of a payment from p_k's other account a_k2 seen on the device, if
    any, so that no device is seen on a_k's own payments. p_k's credit card a_k_card pays from
    d_0 12 hours before t."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.account("a_k2", "p_k", JOINED)
    w.account("a_k_card", "p_k", JOINED, account_type="credit")
    w.external("x_any", OPENED)
    w.pay("x_first", "a_k", "x_any", T - 60 * D, 1.0, "ach")
    w.pay("x_card", "a_k_card", "x_any", T - 12 * H, 1.0, "card", device="d_0")
    for device, (start, used) in evidence.items():
        w.device("p_k", device, JOINED if start is None else start)
        if used is not None:
            w.pay(f"x_{device}", "a_k2", "x_any", used, 1.0, "ach", device=device)
    return w


def test_a_customer_keeps_its_8_links_of_a_kind_with_the_newest_evidence() -> None:
    # Eleven device links. The evidence of each is the later of its known start (after x_first,
    # 60 days before t) and p_k's latest own payment seen on it: d_0's, enrolled before the
    # lookback, from p_k's credit card 12 hours before t, since an own payment from any account
    # counts; d_i's start the day before t; d_h's use 2 days before t; d_g's start 3 days
    # before t; ... d_c's start 7 days before t, though it was used 50 days before t; d_b's use
    # 8 days before t; d_a's start 9 days before t. d_z has neither and comes last. Kept: d_0
    # and d_c to d_i; d_a, d_b and d_z are left out, though d_a and d_b come first by id.
    evidence: dict[str, tuple[int | None, int | None]] = {
        "d_a": (T - 9 * D, None),
        "d_b": (None, T - 8 * D),
        "d_c": (T - 7 * D, T - 50 * D),
        "d_d": (None, T - 6 * D),
        "d_e": (T - 5 * D, None),
        "d_f": (None, T - 4 * D),
        "d_g": (T - 3 * D, None),
        "d_h": (None, T - 2 * D),
        "d_i": (T - D, None),
        "d_0": (None, None),
        "d_z": (None, None),
    }
    (found,) = _graphs(_links_world(evidence).graph(), ["a_k"], [])
    enrolled = [edge.node_id for edge in found.enrolled]
    assert enrolled == ["d_0", "d_c", "d_d", "d_e", "d_f", "d_g", "d_h", "d_i"]


def test_a_customers_links_to_one_node_are_one_link_from_the_earliest_start() -> None:
    # p_k held d_k from joining until 60 days before t, and again from 20 days before t, after
    # a_k's first payment 50 days before t. One link: its start is the earlier tenure's,
    # before the lookback, so unknown; it is held in the last 30 days, and not new.
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    w.device("p_k", "d_k", JOINED, T - 60 * D)
    w.device("p_k", "d_k", T - 20 * D)
    w.pay("x_k", "a_k", "x_any", T - 50 * D, 1.0, "ach")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    assert found.customers == (CustomerRow("p_k", 1, 1, 0, 0, 0, 0, 0, 0),)
    assert found.enrolled == (Enrolled("p_k", "d_k", False, 0, False, 0),)
    assert found.devices == (NodeRow("d_k", False, 1, 1, 0, 0, 0, 0, False),)


def test_an_account_keeps_its_8_seen_nodes_of_a_kind_latest_use_first() -> None:
    # a_k's payments seen on ten IPs: i_09 the day before t, i_08 2 days before t, down to
    # i_03 7 days before t; i_02 and i_99 in the same millisecond 8 days before t; i_01 9 days
    # before t. The latest 7 and, of the tied pair, i_02 by id are kept; i_99 and i_01 are
    # left out.
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    for n in range(1, 10):
        w.pay(f"x_i{n}", "a_k", "x_any", T - (10 - n) * D, 1.0, "ach", ip=f"i_{n:02d}")
    w.pay("x_i99", "a_k", "x_any", T - 8 * D, 1.0, "ach", ip="i_99")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    assert [edge.node_id for edge in found.seen] == [f"i_{n:02d}" for n in range(2, 10)]


def _co_users(late: str | None) -> WorldBuilder:
    """p_k enrolled on 7 devices, d_0 to d_6, and on each 7 others, c_j_0 to c_j_6, all since
    before the lookback. When late is "deposit" a_c_6_6, c_6_6's deposit account, pays from
    d_6 the day before t; when it is "card" c_6_6's credit card does. a_k pays another bank's
    x_mid, which pays a_c_6_6: a hop-2 account of a_k."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    w.external("x_mid", OPENED)
    w.pay("x_k_mid", "a_k", "x_mid", T - 20 * D, 1.0, "ach")
    w.pay("x_mid_c", "x_mid", "a_c_6_6", T - 19 * D, 1.0, "ach")
    for d in range(7):
        w.device("p_k", f"d_{d}", JOINED)
        for c in range(7):
            w.customer(f"c_{d}_{c}", JOINED)
            w.device(f"p_c_{d}_{c}", f"d_{d}", JOINED)
    if late == "deposit":
        w.pay("x_late", "a_c_6_6", "x_any", T - D, 1.0, "ach", device="d_6")
    if late == "card":
        w.account("a_c_6_6_card", "p_c_6_6", JOINED, account_type="credit")
        w.pay("x_late", "a_c_6_6_card", "x_any", T - D, 1.0, "card", device="d_6")
    return w


@pytest.mark.parametrize(
    ("late", "left_out"),
    [(None, "c_6_6"), ("deposit", "c_6_5"), ("card", "c_6_6")],
    ids=["no evidence", "a deposit account's use", "a credit card's use"],
)
def test_the_newest_48_co_users_are_kept_by_evidence_then_id(
    late: str | None, left_out: str
) -> None:
    # 49 candidates. With no evidence, each is t - 90 days, so they are kept by id: c_0_0 to
    # c_6_5, and not c_6_6. A use of d_6 from c_6_6's deposit account the day before t is the
    # newest evidence, so c_6_6 is kept and c_6_5, now last, is left out; a credit card's use
    # is no evidence, as it counts for no customer. Each kept co-user, linked to a node of a_k,
    # is a customer of the subgraph and brings its deposit account. a_c_6_6, a hop-2 account
    # of a_k, has its owner-enrolment edge to d_6, a node of a_k, whether or not c_6_6 is kept.
    (found,) = _graphs(_co_users(late).graph(), ["a_k"], [])
    kept = {f"c_{d}_{c}" for d in range(7) for c in range(7)} - {left_out}
    assert {row.customer_id for row in found.customers} == {"p_k"} | {f"p_{c}" for c in kept}
    assert {row.account_id for row in found.accounts} == {f"a_{c}" for c in kept}
    assert ("a_c_6_6", "d_6") in {(e.account_id, e.node_id) for e in found.owner_enrolled}


def test_a_co_user_brings_its_deposit_account_last_seen_then_last_active_then_by_id() -> None:
    # p_k pays from d_k, on which three customers are enrolled.
    # - p_u's a_u1 paid from d_k 4 days before t and a_u2 2 days before t, then a_u2 paid
    #   elsewhere with no device; a_u3 is newer but never used d_k: a_u2, last seen on d_k.
    # - p_v's accounts never used d_k; a_v2 paid the day before t, a_v1 10 days before t, and
    #   its credit card a_v_card an hour before t: a_v2, the last active deposit account.
    # - p_w's a_w2 and a_w1 have no payment, and a_w0, opened 3 days after t, does not exist
    #   at t: a_w1, by id.
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    w.device("p_k", "d_k", JOINED)
    w.pay("x_k", "a_k", "x_any", T - 8 * D, 1.0, "ach", device="d_k")
    for customer, accounts in (
        ("p_u", ("a_u1", "a_u2", "a_u3")),
        ("p_v", ("a_v1", "a_v2")),
        ("p_w", ("a_w1", "a_w2")),
    ):
        w.party(customer, JOINED)
        w.device(customer, "d_k", JOINED)
        for account in accounts:
            w.account(account, customer, JOINED)
    w.account("a_v_card", "p_v", JOINED, account_type="credit")
    w.account("a_w0", "p_w", T + 3 * D)
    w.pay("x_u1", "a_u1", "x_any", T - 4 * D, 1.0, "ach", device="d_k")
    w.pay("x_u2", "a_u2", "x_any", T - 2 * D, 1.0, "ach", device="d_k")
    w.pay("x_u2b", "a_u2", "x_any", T - D, 1.0, "ach")
    w.pay("x_u3", "a_u3", "x_any", T - H, 1.0, "ach")
    w.pay("x_v2", "a_v2", "x_any", T - D, 1.0, "ach")
    w.pay("x_v1", "a_v1", "x_any", T - 10 * D, 1.0, "ach")
    w.pay("x_v_card", "a_v_card", "x_any", T - H, 1.0, "card")
    (found,) = _graphs(w.graph(), ["a_k"], [])
    assert {row.account_id for row in found.accounts} == {"a_u2", "a_v2", "a_w1"}


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"t_ms": T + 1}, "invalid_parameters"),
        ({"account_ids": []}, "invalid_parameters"),
        ({"account_ids": ["a_s", "a_s"]}, "duplicate_account"),
        ({"account_ids": ["a_nobody"]}, "unknown_account"),
        ({"account_ids": ["a_late"]}, "invisible_account"),
        ({"hub_accounts": [{"id": "x_shop"}, {"id": "a_late"}]}, "invisible_hub"),
    ],
    ids=[
        "no Monday",
        "no account",
        "an account twice",
        "an unknown account",
        "opened at t",
        "a hub opened at t",
    ],
)
def test_the_query_refuses_a_request_it_cannot_answer(
    params: dict[str, object], status: str
) -> None:
    w = _world()
    w.customer("late", T)
    given = {"t_ms": T, "account_ids": ["a_s"], "hub_accounts": [], **params}
    rows = w.graph().run(READ_SHARED_IDENTITY, given)
    assert rows == [{"status": status}]
    with pytest.raises(QueryRefused, match=status):
        read_shared_identity.parse(rows, ["a_s"])
