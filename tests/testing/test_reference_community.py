"""The reference answer of read_community, B6, on worlds worked out by hand from
docs/features.md ("The community block") and docs/design.md ("read_community (B6): how the
mule network formed"): the rounds by sharing, which never step through a leaf, the payment
round, the block by hand, a closed account of a member, an account with no owner, a hub's own
payments, a crowd that sits between two members (which B6 checks by the parity of its whole
block, since it prints no per-node leaf flag), the cap of 64 and its order in every round, a
start and a payment exactly 30 days before t, the hub parameter and the refusals.

Every answer goes through ReferenceGraph.run and the query's own parser, so it has the shape
TigerGraph prints and the parser's checks hold of it. Amounts are in cents. The leak tests
over B4 to B6 together are in test_leaks.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import (
    READ_COMMUNITY,
    READ_HUBS,
    read_community,
    read_hubs,
)
from mule_pattern_learner.tigergraph.queries.read_community import Community
from mule_pattern_learner.tigergraph.queries.rows import QueryRefused

D = DAY_MS
JOINED, OPENED = fixtures.JOINED, fixtures.OPENED
HUBS = ("g_hub",)


def _blocks(
    graph: ReferenceGraph,
    accounts: Sequence[str],
    hubs: Sequence[str] = HUBS,
    t: int = T,
) -> tuple[Community, ...]:
    found = graph.run(READ_COMMUNITY, read_community.params(t, accounts, hubs))
    return read_community.parse(found, accounts)


def _world() -> WorldBuilder:
    """p_s's account a_s, scored, and the customers around it.

    - p_s holds a_s, savings a_s_save and a_s_closed, which closed 20 days before t. Since
      joining, 400 days before t, it has been enrolled on d_1 and on the carrier IP i_crowd,
      on which 8 others, p_k1 to p_k8, are enrolled too.
    - a_a paid another bank 70 days before t and p_o's a_o 40 days before t. p_a has been
      enrolled on i_crowd since joining, and enrolled on i_2 50 days before t and on d_1 20
      days before t. p_b enrolled on i_2 10 days before t, after a_b paid another bank 50 days
      before t.
    - In the last 30 days: p_x's a_x paid a_s 300.00 3 days before t; a_b paid p_y's a_y 70.00
      10 days before t; a_a paid 15.00 to the hub g_hub, p_g's account, 5 days before t; a_s
      paid another bank 40.00 2 days before t, its savings 100.00 the day before t, and itself
      5.00; and p_o's a_o paid a_s_closed 11.00 25 days before t, before it closed.
    """
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.account("a_s_save", "p_s", JOINED)
    w.account("a_s_closed", "p_s", JOINED, closed_ms=T - 20 * D)
    for name in ("a", "b", "x", "y", "o"):
        w.customer(name, JOINED)
    w.party("p_g", JOINED)
    w.account("g_hub", "p_g", JOINED)
    w.external("x_ext", OPENED)
    w.device("p_s", "d_1", JOINED)
    w.ip("p_s", "i_crowd", JOINED)
    for n in range(1, 9):
        w.party(f"p_k{n}", JOINED)
        w.ip(f"p_k{n}", "i_crowd", JOINED)
    w.device("p_a", "d_1", T - 20 * D)
    w.ip("p_a", "i_2", T - 50 * D)
    w.ip("p_a", "i_crowd", JOINED)
    w.ip("p_b", "i_2", T - 10 * D)
    w.pay("x_a_old", "a_a", "x_ext", T - 70 * D, 20.0, "ach")
    w.pay("z_a_o", "a_a", "a_o", T - 40 * D, 20.0, "zelle")
    w.pay("x_b_ext", "a_b", "x_ext", T - 50 * D, 10.0, "ach")
    w.pay("z_x_s", "a_x", "a_s", T - 3 * D, 300.0, "zelle")
    w.pay("z_b_y", "a_b", "a_y", T - 10 * D, 70.0, "zelle")
    w.pay("x_a_hub", "a_a", "g_hub", T - 5 * D, 15.0, "card")
    w.pay("x_s_ext", "a_s", "x_ext", T - 2 * D, 40.0, "ach")
    w.pay("x_s_save", "a_s", "a_s_save", T - D, 100.0, "internal")
    w.pay("x_s_self", "a_s", "a_s", T - D, 5.0, "internal")
    w.pay("z_o_closed", "a_o", "a_s_closed", T - 25 * D, 11.0, "zelle")
    return w


# The members of p_s's community: p_s (round 0); p_a, sharing d_1 with p_s (round 1); p_b,
# sharing i_2 with p_a (round 2); p_x, whose a_x paid a_s, p_y, whom a_b paid, and p_o, whose
# a_o paid p_s's closed account, which is a member's account all the same (the payment
# round). The leaf i_crowd is never stepped through, so p_k1 to p_k8 are no members, and
# though p_s and p_a both hold it, it is shared by no one; g_hub is a hub and x_ext another
# bank's, so p_g joins by no payment, and a_a's earlier payment to p_o is 40 days before t.
# - Shared nodes: d_1 (p_s, p_a) and i_2 (p_a, p_b): 2 nodes and 4 links; the density is
#   4 / (6 members x 2 nodes) = 1/3. Every start of p_a and p_b comes after an own payment,
#   so each is known: p_a's link to d_1 began 20 days before t and p_b's to i_2 10 days
#   before t, new; p_a's to i_2 50 days before t, and p_s's to d_1 before the lookback, not
#   new: 2 of 4.
# - Payments of the last 30 days: z_x_s (30,000), z_b_y (7,000) and z_o_closed (1,100)
#   between two members, internal; x_a_hub to a hub and x_s_ext to another bank, external;
#   x_s_save between accounts of p_s, neither; the self-payment is read nowhere. Isolation
#   3 / 5.
BLOCK = Community("a_s", 6, False, 2, 4 / 12, 2 / 4, 38_100, 3 / 5, 5)


def test_a_community_grows_by_sharing_then_by_payments_by_hand() -> None:
    assert _blocks(_world().graph(), ["a_s"]) == (BLOCK,)


def test_the_same_members_give_the_same_block_from_another_member() -> None:
    # From p_a, p_s and p_b share a node with it in round 1, and the payment round adds p_x,
    # p_y and p_o as before: the same 6 members, nodes, links and payments.
    expected = Community("a_a", 6, False, 2, 4 / 12, 2 / 4, 38_100, 3 / 5, 5)
    assert _blocks(_world().graph(), ["a_a"]) == (expected,)


def test_an_account_with_no_owner_has_an_empty_block() -> None:
    assert _blocks(_world().graph(), ["x_ext"]) == (
        Community("x_ext", 0, False, 0, 0.0, 0.0, 0, 0.0, 0),
    )


def test_a_hub_is_no_members_account() -> None:
    # Scored, the hub g_hub has its owner p_g as the one member; but a member's accounts leave
    # out account hubs, so p_g has none and no payment is read: x_a_hub makes no member.
    assert _blocks(_world().graph(), ["g_hub"]) == (
        Community("g_hub", 1, False, 0, 0.0, 0.0, 0, 0.0, 0),
    )


def test_the_hubs_are_those_of_the_parameter_alone() -> None:
    # With no hub, g_hub is p_g's account: a_a's payment to it makes p_g a member of the
    # payment round, and internal; 7 members, density 4 / (7 x 2), isolation 4 / 5. A hub the
    # graph lacks is dropped, as TigerGraph drops it.
    expected = Community("a_s", 7, False, 2, 4 / 14, 2 / 4, 39_600, 4 / 5, 5)
    graph = _world().graph()
    assert _blocks(graph, ["a_s"], []) == (expected,)
    assert _blocks(graph, ["a_s"], ["x_nowhere"]) == (expected,)


def _between(*, unregistered: bool) -> WorldBuilder:
    """Two members with a crowd between them and nothing else shared.

    - p_s holds a_s; p_m's a_m paid a_s 50.00 five days before t, so p_m joins p_s's community
      in the payment round and no other way.
    - p_s, p_m and p_k1 to p_k6 have been enrolled on the carrier device d_crowd since joining:
      8 customers, which is no crowd. When unregistered is set, p_u, who never enrolled on it,
      sent a payment from a_u seen on d_crowd 20 days before t: a ninth customer by the crowd
      rule, which counts registered and unregistered customers alike, so d_crowd is a crowd.
    """
    w = WorldBuilder()
    w.external("x_ext", OPENED)
    for name in ("s", "m", *(f"k{n}" for n in range(1, 7))):
        w.customer(name, JOINED)
        w.device(f"p_{name}", "d_crowd", JOINED)
    w.pay("z_m_s", "a_m", "a_s", T - 5 * D, 50.0, "zelle")
    if unregistered:
        w.customer("u", JOINED)
        w.pay("x_u_seen", "a_u", "x_ext", T - 20 * D, 5.0, "ach", device="d_crowd")
    return w


def _crowds(graph: ReferenceGraph) -> set[str]:
    """The devices read_hubs lists as crowds on t."""
    leaves = read_hubs.parse(graph.run(READ_HUBS, read_hubs.params(T)))
    return {leaf.node_id for leaf in (*leaves.devices, *leaves.ips)}


def test_a_crowd_between_two_members_is_no_shared_node_and_brings_no_member() -> None:
    # d_crowd holds 9 customers, p_u counted though it never enrolled: read_hubs lists it, so
    # the community does not step through it. p_s and p_m are both enrolled on it and are
    # members, p_m by a payment, yet the community holds those two and no more: no shared
    # node, so no link, density 0 and no new link; p_k1 to p_k6 and p_u are no members. The
    # one payment of the last 30 days, a_m to a_s, is between two members: 5,000 cents, an
    # isolation of 1 and a count of 1. From p_m's account the block is the same two members.
    graph = _between(unregistered=True).graph()
    assert _crowds(graph) == {"d_crowd"}
    assert _blocks(graph, ["a_s", "a_m"], []) == (
        Community("a_m", 2, False, 0, 0.0, 0.0, 5_000, 1.0, 1),
        Community("a_s", 2, False, 0, 0.0, 0.0, 5_000, 1.0, 1),
    )
    # A customer of the crowd alone is no community but itself. So is p_u, who never enrolled
    # on d_crowd: its one payment of the last 30 days, a_u's to another bank 20 days before t,
    # is external, so an isolation of 0 over a count of 1.
    assert _blocks(graph, ["a_k1", "a_u"], []) == (
        Community("a_k1", 1, False, 0, 0.0, 0.0, 0, 0.0, 0),
        Community("a_u", 1, False, 0, 0.0, 0.0, 0, 0.0, 1),
    )


def test_the_same_node_with_one_customer_fewer_is_no_crowd_and_joins_its_users() -> None:
    # Without p_u, d_crowd holds 8 customers, which is no crowd (a leaf holds more than 8): the
    # community steps through it. Round 1 adds p_m and p_k1 to p_k6, 7 candidates, all
    # members; the payment round adds nobody, p_m being one already. d_crowd is shared by the
    # 8 members: 8 links over 8 members x 1 node, a density of 1, and none new, every link
    # having begun before the lookback. The payment is between two members as before.
    graph = _between(unregistered=False).graph()
    assert _crowds(graph) == set()
    assert _blocks(graph, ["a_s"], []) == (Community("a_s", 8, False, 1, 1.0, 0.0, 5_000, 1.0, 1),)


def _crowded(
    devices: int, extra: int = 0, *, newest: bool = False, paid: bool = False
) -> WorldBuilder:
    """p_s enrolled since joining on devices d_0 onwards, and on each 7 others, c_j_0 to
    c_j_6, all since joining; then extra more on one more device. When newest is set, c_9_6's
    link to d_9 began 5 days before t, after its first payment 60 days before t, and c_9_6
    has held d_x, which p_s holds too, since joining. When paid is set, a_c_0_0 paid a_c_0_1
    1.00 the day before t."""
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.external("x_any", OPENED)
    for d in range(devices + (1 if extra else 0)):
        w.device("p_s", f"d_{d}", JOINED)
        for c in range(7 if d < devices else extra):
            name = f"c_{d}_{c}"
            w.customer(name, JOINED)
            late = newest and name == "c_9_6"
            w.device(f"p_{name}", f"d_{d}", T - 5 * D if late else JOINED)
    if newest:
        w.pay("x_first", "a_c_9_6", "x_any", T - 60 * D, 1.0, "ach")
        w.device("p_s", "d_x", JOINED)
        w.device("p_c_9_6", "d_x", JOINED)
    if paid:
        w.pay("z_paid", "a_c_0_0", "a_c_0_1", T - D, 1.0, "zelle")
    return w


def test_a_community_keeps_64_members_newest_first_ties_by_id() -> None:
    # 70 candidates in round 1, every one's time t - 90 days: the 63 smallest ids join p_s,
    # so the 7 of d_9 are left out. Nine nodes are shared, each by p_s and 7 members:
    # density 72 / (64 x 9) = 0.125. Every link began before the lookback.
    assert _blocks(_crowded(10).graph(), ["a_s"], []) == (
        Community("a_s", 64, True, 9, 72 / (64 * 9), 0.0, 0, 0.0, 0),
    )
    # c_9_6 shares d_9 with p_s from 5 days before t, and d_x from before the lookback: its
    # time is the latest, 5 days before t, the newest. It joins first, then the 62 smallest
    # ids, c_0_0 to c_8_5; c_8_6 and c_9_0 to c_9_5 are left out. Eleven nodes are shared:
    # d_0 to d_7 by 8 members, d_8 by 7, d_9 and d_x by p_s and c_9_6, 64 + 7 + 2 + 2 = 75
    # links; density 75 / (64 x 11), and c_9_6's link to d_9 is the one new. Its first payment
    # went to another bank 60 days before t, out of the last 30 days.
    assert _blocks(_crowded(10, newest=True).graph(), ["a_s"], []) == (
        Community("a_s", 64, True, 11, 75 / (64 * 11), 1 / 75, 0, 0.0, 0),
    )


def test_capped_says_a_candidate_was_left_out_not_that_the_community_is_full() -> None:
    # 63 candidates on d_0 to d_8 fill the community exactly: none is left out, and the
    # payment between two members, internal, proposes no one. One more candidate, on d_9, is
    # the one left out, being last by id, and d_9 is then shared by p_s alone.
    assert _blocks(_crowded(9, paid=True).graph(), ["a_s"], []) == (
        Community("a_s", 64, False, 9, 72 / (64 * 9), 0.0, 100, 1.0, 1),
    )
    assert _blocks(_crowded(9, extra=1).graph(), ["a_s"], []) == (
        Community("a_s", 64, True, 9, 72 / (64 * 9), 0.0, 0, 0.0, 0),
    )


def test_a_start_and_a_payment_exactly_30_days_before_t_are_in_the_last_30_days() -> None:
    # p_s and p_n each paid another bank 60 days before t from no device and enrolled on d_b
    # exactly 30 days before t: known starts, the first moment of the last 30 days, so both
    # links of d_b are new, 2 of 2. a_n paid a_s 7.00 and p_m's a_m paid it 3.00, both exactly
    # 30 days before t: the first payments of the last 30 days. p_m joins the community by its
    # payment, and the two payments are between members, internal, 1,000 cents, 2 of 2.
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.customer("n", JOINED)
    w.customer("m", JOINED)
    w.external("x_s0", OPENED)
    w.external("x_n0", OPENED)
    w.pay("x_s", "a_s", "x_s0", T - 60 * D, 1.0, "ach")
    w.pay("x_n", "a_n", "x_n0", T - 60 * D, 1.0, "ach")
    w.device("p_s", "d_b", T - 30 * D)
    w.device("p_n", "d_b", T - 30 * D)
    w.pay("z_n_s", "a_n", "a_s", T - 30 * D, 7.0, "zelle")
    w.pay("z_m_s", "a_m", "a_s", T - 30 * D, 3.0, "zelle")
    assert _blocks(w.graph(), ["a_s"], []) == (
        Community("a_s", 3, False, 1, 2 / 3, 1.0, 1_000, 1.0, 2),
    )


def _payment_candidates(candidates: bool) -> WorldBuilder:
    """p_s enrolled since joining on d_0 to d_8, and 61 others on them: seven on each of d_0 to
    d_7 and five on d_8, so that with p_s the community holds 62, 2 short of the cap. When
    candidates is set, four others pay a_s in the last 30 days: p_e's a_e 100.00 2 days before
    t; p_b's a_b 40.00 4 days before t and 80.00 28 days before t; p_c's a_c 30.00 4 days before
    t, the same millisecond as a_b's 40.00; and p_a's a_a 20.00 25 days before t."""
    w = _crowded(8, extra=5)
    if candidates:
        for name, payments in (
            ("e", ((2, 100.0),)),
            ("b", ((4, 40.0), (28, 80.0))),
            ("c", ((4, 30.0),)),
            ("a", ((25, 20.0),)),
        ):
            w.customer(name, JOINED)
            for n, (days, amount) in enumerate(payments):
                w.pay(f"z_{name}{n}", f"a_{name}", "a_s", T - days * D, amount, "zelle")
    return w


def test_the_payment_round_keeps_the_candidates_with_the_latest_payments_ties_by_id() -> None:
    # 62 members before the payment round, room for 2 of its 4 candidates, each timed by its
    # latest payment: p_e 2 days before t, p_b and p_c, tied, 4 days before t (p_b's other
    # payment, 28 days before t, is not its latest), p_a 25 days before t. p_e, then p_b, the
    # smaller id of the tie, are kept: p_c and p_a are left out and capped is set. The 64
    # members share nine nodes in 8 x 8 + 6 = 70 links, none new; the payments of the last 30
    # days are 3 between members, 22,000 cents (10,000 + 4,000 + 8,000), and 2 with p_c and
    # p_a, which are no members.
    graph = _payment_candidates(True).graph()
    assert _blocks(graph, ["a_s"], []) == (
        Community("a_s", 64, True, 9, 70 / (64 * 9), 0.0, 22_000, 3 / 5, 5),
    )
    # Without the candidates the community is the 62, not capped.
    assert _blocks(_payment_candidates(False).graph(), ["a_s"], []) == (
        Community("a_s", 62, False, 9, 70 / (62 * 9), 0.0, 0, 0.0, 0),
    )


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"t_ms": T + 1}, "invalid_parameters"),
        ({"account_ids": []}, "invalid_parameters"),
        ({"account_ids": ["a_s", "a_s"]}, "duplicate_account"),
        ({"account_ids": ["a_nobody"]}, "unknown_account"),
        ({"account_ids": ["a_late"]}, "invisible_account"),
        ({"hub_accounts": [{"id": "g_hub"}, {"id": "a_late"}]}, "invisible_hub"),
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
    rows = w.graph().run(READ_COMMUNITY, given)
    assert rows == [{"status": status}]
    with pytest.raises(QueryRefused, match=status):
        read_community.parse(rows, ["a_s"])
