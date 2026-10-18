"""The reference answer of read_community, B6, on worlds worked out by hand from
docs/features.md ("The community block") and docs/design.md ("read_community (B6): how the
mule network formed"): the rounds by sharing, which never step through a leaf, the payment
round, the block by hand, an account with no owner, a hub's own payments, the cap of 64 and
its order, the hub parameter and the refusals.

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
from mule_pattern_learner.tigergraph.queries import READ_COMMUNITY, read_community
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

    - p_s holds a_s and savings a_s_save. Since joining, 400 days before t, it has been
      enrolled on d_1 and on the carrier IP i_crowd, on which 8 others, p_k1 to p_k8, are
      enrolled too.
    - a_a paid another bank 70 days before t and p_o's a_o 40 days before t. p_a has been
      enrolled on i_crowd since joining, and enrolled on i_2 50 days before t and on d_1 20
      days before t. p_b enrolled on i_2 10 days before t, after a_b paid another bank 50 days
      before t.
    - In the last 30 days: p_x's a_x paid a_s 300.00 3 days before t; a_b paid p_y's a_y 70.00
      10 days before t; a_a paid 15.00 to the hub g_hub, p_g's account, 5 days before t; a_s
      paid another bank 40.00 2 days before t, its savings 100.00 the day before t, and itself
      5.00.
    """
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.account("a_s_save", "p_s", JOINED)
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
    return w


# The members of p_s's community: p_s (round 0); p_a, sharing d_1 with p_s (round 1); p_b,
# sharing i_2 with p_a (round 2); p_x, whose a_x paid a_s, and p_y, whom a_b paid (the payment
# round). The leaf i_crowd is never stepped through, so p_k1 to p_k8 are no members, and
# though p_s and p_a both hold it, it is shared by no one; g_hub is a hub and x_ext another
# bank's, so p_g joins by no payment, and p_o's payment is 40 days before t.
# - Shared nodes: d_1 (p_s, p_a) and i_2 (p_a, p_b): 2 nodes and 4 links; the density is
#   4 / (5 members x 2 nodes) = 0.4. Every start of p_a and p_b comes after an own payment,
#   so each is known: p_a's link to d_1 began 20 days before t and p_b's to i_2 10 days
#   before t, new; p_a's to i_2 50 days before t, and p_s's to d_1 before the lookback, not
#   new: 2 of 4.
# - Payments of the last 30 days: z_x_s (30,000) and z_b_y (7,000) between two members,
#   internal; x_a_hub to a hub and x_s_ext to another bank, external; x_s_save between
#   accounts of p_s, neither; the self-payment is read nowhere. Isolation 2 / 4.
BLOCK = Community("a_s", 5, False, 2, 4 / 10, 2 / 4, 37_000, 2 / 4, 4)


def test_a_community_grows_by_sharing_then_by_payments_by_hand() -> None:
    assert _blocks(_world().graph(), ["a_s"]) == (BLOCK,)


def test_the_same_members_give_the_same_block_from_another_member() -> None:
    # From p_a, p_s and p_b share a node with it in round 1, and the payment round adds p_x and
    # p_y as before: the same 5 members, nodes, links and payments.
    expected = Community("a_a", 5, False, 2, 4 / 10, 2 / 4, 37_000, 2 / 4, 4)
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
    # payment round, and internal; 6 members, density 4 / (6 x 2), isolation 3 / 4. A hub the
    # graph lacks is dropped, as TigerGraph drops it.
    expected = Community("a_s", 6, False, 2, 4 / 12, 2 / 4, 38_500, 3 / 4, 4)
    graph = _world().graph()
    assert _blocks(graph, ["a_s"], []) == (expected,)
    assert _blocks(graph, ["a_s"], ["x_nowhere"]) == (expected,)


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
