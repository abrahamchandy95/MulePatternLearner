"""The leak tests over the graph queries B4 to B6 (docs/design.md, "Leak rules and how tests
enforce them"), on the reference graph:

- Account Z: a payment-less account that shares a ring device with an active mule reads the
  mule, and one that shares nothing does not;
- known starts: a joiner's first links read no new link and no time since start; a joiner and
  a long-standing customer with the same behaviour read the same; a long-standing customer's
  new phone reads new;
- link ends: a tenure whose end later uses set reads as one that ended, whether its last use
  is recent or is out of the last 30 days, so that the link alone puts its holder in the
  sharing;
- a caught neighbour reads exactly like an uncaught one;
- an account fetched alone equals itself among 31 others;
- a co-user's account opened after t never appears;
- a node with 9 customers by the one account rule is a leaf, never expanded, and a node that
  is no leaf has at most 8 customers sharing it;
- erasing every label column and moving every date of record before the lookback change no
  answer, in the catalogue and in worlds where an identified mule shares a node through its
  payments alone and where a cap cuts links, accounts, seen nodes or co-users among ties;
  swapping the ids of same-second payments changes no value.

Each answer goes through ReferenceGraph.run and the query's own parser, or is compared as the
bytes TigerGraph would send.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import json
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, Mule, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import (
    READ_COMMUNITY,
    READ_HUBS,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    read_community,
    read_hubs,
    read_payment_events,
    read_shared_identity,
)
from mule_pattern_learner.tigergraph.queries.read_community import Community
from mule_pattern_learner.tigergraph.queries.read_payment_events import PaymentGraph
from mule_pattern_learner.tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityGraph,
    Seen,
)
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS

D, H, M = DAY_MS, HOUR_MS, fixtures.MINUTE_MS
JOINED, OPENED = fixtures.JOINED, fixtures.OPENED
QUERIES = (READ_PAYMENT_EVENTS, READ_SHARED_IDENTITY, READ_COMMUNITY)
PARAMS = {
    READ_PAYMENT_EVENTS: read_payment_events.params,
    READ_SHARED_IDENTITY: read_shared_identity.params,
    READ_COMMUNITY: read_community.params,
}


def _existing(graph: ReferenceGraph, t: int = T) -> list[str]:
    """Every account that exists at t, which B4 to B6 may be asked for."""
    return [a for a in graph.ids["Account"] if graph.accounts[a].opened_ms < t]


def _payments(
    graph: ReferenceGraph, accounts: Sequence[str], hubs: Sequence[str] = ()
) -> tuple[PaymentGraph, ...]:
    params = read_payment_events.params(T, accounts, hubs)
    return read_payment_events.parse(graph.run(READ_PAYMENT_EVENTS, params), accounts)


def _identity(
    graph: ReferenceGraph, accounts: Sequence[str], hubs: Sequence[str] = ()
) -> tuple[IdentityGraph, ...]:
    params = read_shared_identity.params(T, accounts, hubs)
    return read_shared_identity.parse(graph.run(READ_SHARED_IDENTITY, params), accounts)


def _community(
    graph: ReferenceGraph, accounts: Sequence[str], hubs: Sequence[str] = ()
) -> tuple[Community, ...]:
    params = read_community.params(T, accounts, hubs)
    return read_community.parse(graph.run(READ_COMMUNITY, params), accounts)


def _printed(graph: ReferenceGraph, hubs: Sequence[str] = fixtures.HUBS) -> dict[str, str]:
    """Each query's answer for every account that exists at t, as the bytes TigerGraph would
    send."""
    accounts = _existing(graph)
    return {name: json.dumps(graph.run(name, PARAMS[name](T, accounts, hubs))) for name in QUERIES}


# Account Z.


def _ring(identified: bool = False) -> WorldBuilder:
    """A ring device d_ring held from 12 days before t by p_m, the customer of an active mule
    a_m, and by p_z, whose account a_z has never paid; and p_y's a_y, which has never paid
    either, with its own phone d_y since joining. Victims pay a_m 6 days before t, and it
    forwards the money from d_ring 10 minutes later; it is identified when identified is
    set."""
    w = WorldBuilder()
    onset = T - 6 * D
    w.customer("m", JOINED, mule=Mule(onset, onset + 10 * M, 3, identified))
    w.customer("z", JOINED)
    w.customer("y", JOINED)
    w.external("x_vic", OPENED)
    w.external("x_out", OPENED)
    w.device("p_m", "d_ring", T - 12 * D)
    w.device("p_z", "d_ring", T - 12 * D)
    w.device("p_y", "d_y", JOINED)
    w.pay("x_vic1", "x_vic", "a_m", onset, 900.0, "ach")
    w.pay("x_fwd1", "a_m", "x_out", onset + 10 * M, 850.0, "ach", device="d_ring")
    return w


def test_account_z_reads_the_mule_it_shares_a_ring_device_with() -> None:
    graph = _ring().graph()
    y, z = _identity(graph, ["a_y", "a_z"])
    # Both have no payment, so their payment graphs hold themselves alone.
    assert [len(g.accounts) for g in _payments(graph, ["a_y", "a_z"])] == [1, 1]
    # Through d_ring, a_z's graph holds the mule's customer and account, with its
    # owner-enrolment and seen edges: the mule's rows reach a_z in two layers.
    assert {row.customer_id for row in z.customers} == {"p_z", "p_m"}
    assert {row.account_id for row in z.accounts} == {"a_m"}
    assert {(e.account_id, e.node_id) for e in z.owner_enrolled} == {
        ("a_z", "d_ring"),
        ("a_m", "d_ring"),
    }
    assert {(e.account_id, e.node_id) for e in z.seen} == {("a_m", "d_ring")}
    # a_y's holds itself, its customer and its own phone.
    assert {row.customer_id for row in y.customers} == {"p_y"}
    assert y.accounts == () and y.seen == ()
    assert [block.members for block in _community(graph, ["a_y", "a_z"])] == [1, 2]


def test_a_caught_neighbour_reads_exactly_like_an_uncaught_one() -> None:
    # Identification closes nothing: a_m identified or not, every answer is the same.
    assert _printed(_ring(True).graph(), []) == _printed(_ring(False).graph(), [])


# Known starts.


def _joiner(joined: int) -> WorldBuilder:
    """p_j, who joined at joined, and whose first device and IP links, d_j and i_j, start at
    its first payment, made from them at 03:00 10 days before t; it pays from them again 5
    days before t."""
    w = WorldBuilder()
    first = T - 10 * D + 3 * H
    w.customer("j", joined)
    w.external("x_any", OPENED)
    w.device("p_j", "d_j", first)
    w.ip("p_j", "i_j", first)
    w.pay("x_j1", "a_j", "x_any", first, 10.0, "ach", device="d_j", ip="i_j")
    w.pay("x_j2", "a_j", "x_any", T - 5 * D, 10.0, "ach", device="d_j", ip="i_j")
    return w


def test_a_joiners_first_links_read_no_new_link_and_no_time_since_start() -> None:
    # The links start at p_j's first payment, so no own payment came before them: unknown,
    # and so is its first use of each node.
    (found,) = _identity(_joiner(T - 10 * D).graph(), ["a_j"])
    assert found.customers == (CustomerRow("p_j", 1, 1, 1, 0, 0, 0, 0, 0),)
    assert found.enrolled == (
        Enrolled("p_j", "d_j", False, 0, True, 5 * D),
        Enrolled("p_j", "i_j", False, 0, True, 5 * D),
    )
    assert found.seen == (
        Seen("a_j", "d_j", 2, False, 0, 5 * D),
        Seen("a_j", "i_j", 2, False, 0, 5 * D),
    )


def test_a_joiner_and_a_long_standing_customer_with_the_same_behaviour_read_the_same() -> None:
    joiner, standing = _joiner(T - 10 * D).graph(), _joiner(JOINED).graph()
    assert _printed(joiner, []) == _printed(standing, [])


def test_a_long_standing_customers_new_phone_reads_new() -> None:
    # p_l has had d_old since joining and paid from it 40 days before t; it enrolled d_new 10
    # days before t and paid from it 5 days before t. Both starts on d_new follow the payment
    # from d_old, so both are known.
    w = WorldBuilder()
    w.customer("l", JOINED)
    w.external("x_any", OPENED)
    w.device("p_l", "d_old", JOINED)
    w.device("p_l", "d_new", T - 10 * D)
    w.pay("x_l1", "a_l", "x_any", T - 40 * D, 10.0, "ach", device="d_old")
    w.pay("x_l2", "a_l", "x_any", T - 5 * D, 10.0, "ach", device="d_new")
    (found,) = _identity(w.graph(), ["a_l"])
    assert found.customers == (CustomerRow("p_l", 1, 2, 0, 1, 0, 0, 0, 0),)
    assert found.enrolled == (
        Enrolled("p_l", "d_new", True, 10 * D, True, 5 * D),
        Enrolled("p_l", "d_old", False, 0, True, 40 * D),
    )
    assert found.seen == (
        Seen("a_l", "d_new", 1, True, 5 * D, 5 * D),
        Seen("a_l", "d_old", 1, False, 0, 40 * D),
    )
    assert [row.new_customers_30d for row in found.devices] == [1, 0]


def test_a_link_after_the_customers_own_use_of_its_node_reads_no_new_link() -> None:
    # p_u's first payment, 20 days before t, was seen on d_u, which it enrolled 10 days before
    # t: no own payment came before the link not seen on d_u, so its start is unknown. Its
    # first use, the first payment, is unknown too.
    w = WorldBuilder()
    w.customer("u", JOINED)
    w.external("x_any", OPENED)
    w.device("p_u", "d_u", T - 10 * D)
    w.pay("x_u1", "a_u", "x_any", T - 20 * D, 10.0, "ach", device="d_u")
    (found,) = _identity(w.graph(), ["a_u"])
    assert found.customers == (CustomerRow("p_u", 1, 1, 0, 0, 0, 0, 0, 0),)
    assert found.enrolled == (Enrolled("p_u", "d_u", False, 0, True, 20 * D),)
    assert found.seen == (Seen("a_u", "d_u", 1, False, 0, 20 * D),)


def test_a_start_at_joining_is_never_known() -> None:
    # The join clause alone: p_q's account a_q, opened before p_q joined 20 days before t (no
    # world of PhantomLedger's opens one so), paid 30 days before t with no device, and p_q
    # enrolled d_q exactly when it joined. An own payment came first, but a start at joining
    # dates the customer: unknown.
    w = WorldBuilder()
    w.party("p_q", T - 20 * D)
    w.account("a_q", "p_q", T - 40 * D)
    w.external("x_any", OPENED)
    w.device("p_q", "d_q", T - 20 * D)
    w.pay("x_q1", "a_q", "x_any", T - 30 * D, 10.0, "ach")
    (found,) = _identity(w.graph(), ["a_q"])
    assert found.customers == (CustomerRow("p_q", 1, 1, 0, 0, 0, 0, 0, 0),)
    assert found.enrolled == (Enrolled("p_q", "d_q", False, 0, False, 0),)


# Link ends.


def _tenures(resume: bool, last_use_days: int) -> WorldBuilder:
    """Two ring-mates, p_r1 and p_r2, held d_ring from 40 days before t, and two neighbours,
    p_h1 and p_h2, held the hotspot i_cafe from 45 days before t. Each paid another bank 50
    days before t with no device, and made its last use of its node at 14:00, last_use_days
    before t.

    Without resume each tenure ended 3.5 days after the start of its last use's day, inside
    the law of 4 days: 1.5 days before t for a last use 5 days before t, and 29.5 days before
    t for one 33 days before t, which is still current, since the current test holds a link
    that ended at any time in the last 30 days. With resume p_r1 makes a ring payment and p_h1
    uses the hotspot again 10 days after t, and their tenures run on to 12 days after t. After
    a last use 5 days before t that is within the 21 days of the law. After one 33 days before
    t it is not, and PhantomLedger would have split the tenure; it is built so anyway, to show
    that no answer reads a tenure's end beyond the two tests."""
    w = WorldBuilder()
    w.external("x_out", OPENED)
    day = T - last_use_days * D
    for name, node, start in (
        ("r1", "d_ring", T - 40 * D),
        ("r2", "d_ring", T - 40 * D),
        ("h1", "i_cafe", T - 45 * D),
        ("h2", "i_cafe", T - 45 * D),
    ):
        w.customer(name, JOINED)
        resumes = resume and name in ("r1", "h1")
        ended = T + 12 * D if resumes else day + 3 * D + 12 * H
        device, ip = (node, None) if node == "d_ring" else (None, node)
        if device is not None:
            w.device(f"p_{name}", device, start, ended)
        if ip is not None:
            w.ip(f"p_{name}", ip, start, ended)
        w.pay(f"x_{name}_old", f"a_{name}", "x_out", T - 50 * D, 5.0, "ach")
        w.pay(f"x_{name}", f"a_{name}", "x_out", day + 14 * H, 5.0, "ach", device=device, ip=ip)
        if resumes:
            later = T + 10 * D
            w.pay(f"x_{name}_later", f"a_{name}", "x_out", later, 5.0, "ach", device=device, ip=ip)
    return w


@pytest.mark.parametrize("last_use_days", [5, 33], ids=["used 5 days ago", "used 33 days ago"])
def test_a_link_end_set_by_later_uses_reads_as_one_that_ended(last_use_days: int) -> None:
    # A last use 5 days before t puts each holder in the sharing of the last 30 days through
    # its payment as well as its link. One 33 days before t leaves the link alone: it ended
    # 29.5 days before t, so it is still current, and a read of the end against t, instead of
    # the current test, would tell the resumed holder from the one that ended.
    resumed, ended = _tenures(True, last_use_days).graph(), _tenures(False, last_use_days).graph()
    assert _printed(resumed, []) == _printed(ended, [])
    # In one world the two holders of each node, whose tenures end apart, read alike: in the
    # resumed holder's graph its own customer row and link equal its mate's.
    for resumer, mate in (("r1", "r2"), ("h1", "h2")):
        (found,) = _identity(resumed, [f"a_{resumer}"])
        rows = {row.customer_id: replace(row, customer_id="") for row in found.customers}
        assert rows[f"p_{resumer}"] == rows[f"p_{mate}"]
        links = {edge.customer_id: replace(edge, customer_id="") for edge in found.enrolled}
        assert links[f"p_{resumer}"] == links[f"p_{mate}"]
        # Each holder reads its link as held in the last 30 days, so the premise holds.
        assert rows[f"p_{resumer}"].devices_30d + rows[f"p_{resumer}"].ips_30d == 1


# One account, alone or among others.


def _identity_world() -> WorldBuilder:
    """The catalogue with devices and IPs.

    - d_ring, the collector's controller's device: p_mule2 and p_mule3 enrolled on it 12 days
      before t; a_mule2 pays a_collector 10.00 from it and i_ring 4 days before t, and
      a_collector pays another bank 20.00 from it 2 days before t at 13:00.
    - i_home_t, the tenants' home IP since they joined; a_tenant1 pays a cafe 12.00 from it 3
      days before t.
    - p_landlord's phone d_ll since joining, and a tablet d_ll_new enrolled 8 days before t,
      after its mortgage 30 days before t; a_landlord pays the loan ledger 1.00 from the tablet
      6 days before t.
    - The carrier IP i_carrier, on which 9 customers have been enrolled since joining: a leaf.
    - Two payments to a_mule in one second 2 days before t at 05:00, and two of a_collector in
      one second the day before t at 09:00.
    """
    w = fixtures.catalogue()
    w.device("p_mule2", "d_ring", T - 12 * D)
    w.device("p_mule3", "d_ring", T - 12 * D)
    w.pay(
        "z_ring_m2",
        "a_mule2",
        "a_collector",
        T - 4 * D,
        10.0,
        "zelle",
        device="d_ring",
        ip="i_ring",
    )
    w.pay("x_ring_c", "a_collector", "x_offshore", T - 2 * D + 13 * H, 20.0, "ach", device="d_ring")
    for tenant in ("p_tenant1", "p_tenant2"):
        w.ip(tenant, "i_home_t", JOINED)
    w.pay("x_t1_cafe", "a_tenant1", "x_chain_cafe", T - 3 * D, 12.0, "card", ip="i_home_t")
    w.device("p_landlord", "d_ll", JOINED)
    w.device("p_landlord", "d_ll_new", T - 8 * D)
    w.pay("x_ll_tab", "a_landlord", "g_loans", T - 6 * D, 1.0, "ach", device="d_ll_new")
    for name in ("split", "saver", "sleeper", "parent", "student", "newjob", "vic1", "plumber"):
        w.ip(f"p_{name}", "i_carrier", JOINED)
    w.ip("p_mule", "i_carrier", JOINED)
    w.external("x_tie1", OPENED)
    w.pay("x_tie_b", "x_tie1", "a_mule", T - 2 * D + 5 * H, 70.0, "ach")
    w.pay("x_tie_a", "x_tie1", "a_mule", T - 2 * D + 5 * H, 30.0, "ach")
    w.pay("x_tie_d", "a_collector", "x_offshore", T - D + 9 * H, 5.0, "ach")
    w.pay("x_tie_c", "a_collector", "x_collector_atm", T - D + 9 * H, 5.0, "cash")
    return w


def test_an_account_fetched_alone_equals_itself_among_31_others() -> None:
    graph = _identity_world().graph()
    accounts = _existing(graph)[:32]
    assert len(accounts) == 32 and "a_collector" in accounts
    readers: tuple[Callable[..., tuple[Any, ...]], ...] = (_payments, _identity, _community)
    for read in readers:
        together = read(graph, accounts, fixtures.HUBS)
        alone = tuple(read(graph, [account], fixtures.HUBS)[0] for account in sorted(accounts))
        assert together == alone


# Co-users and leaves.


def test_a_co_users_account_opened_after_t_never_appears() -> None:
    # p_q shares a_s's phone d_s, and its one deposit account a_q_late opens 3 days after t:
    # p_q is a co-user and a member, but owns no account at t and brings none.
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.party("p_q", JOINED)
    w.account("a_q_late", "p_q", T + 3 * D)
    w.device("p_s", "d_s", JOINED)
    w.device("p_q", "d_s", JOINED)
    graph = w.graph()
    (found,) = _identity(graph, ["a_s"])
    assert {row.customer_id: row.accounts_owned for row in found.customers} == {"p_s": 1, "p_q": 0}
    assert found.accounts == ()
    assert _community(graph, ["a_s"])[0].members == 2
    assert all("a_q_late" not in printed for printed in _printed(graph, []).values())


def _crowds() -> WorldBuilder:
    """p_s's a_s and two devices p_s has held since joining. d_eight has 8 customers by the
    one account rule: p_s, three enrolled since joining, and four whose deposit accounts paid
    from it in the last 30 days; p_card paid from it too, but from a credit card, which does
    not count. d_nine has 9: p_s, four enrolled and four paying."""
    w = WorldBuilder()
    w.customer("s", JOINED)
    w.external("x_any", OPENED)
    w.pay("x_s", "a_s", "x_any", T - 9 * D, 1.0, "ach", device="d_eight")
    for node, enrolled, paying in (("d_eight", 3, 4), ("d_nine", 4, 4)):
        w.device("p_s", node, JOINED)
        for n in range(enrolled):
            w.customer(f"{node}_e{n}", JOINED)
            w.device(f"p_{node}_e{n}", node, JOINED)
        for n in range(paying):
            w.customer(f"{node}_p{n}", JOINED)
            w.pay(
                f"x_{node}_p{n}",
                f"a_{node}_p{n}",
                "x_any",
                T - (n + 1) * D,
                1.0,
                "ach",
                device=node,
            )
    w.customer("card", JOINED, account_type="credit")
    w.pay("x_card", "a_card", "x_any", T - D, 1.0, "card", device="d_eight")
    return w


def test_a_node_with_9_customers_is_a_leaf_and_never_expanded() -> None:
    graph = _crowds().graph()
    leaves = read_hubs.parse(graph.run(READ_HUBS, read_hubs.params(T)))
    assert {leaf.node_id: leaf.customers_90d for leaf in leaves.devices} == {"d_nine": 9}
    (found,) = _identity(graph, ["a_s"])
    eight = [f"d_eight_e{n}" for n in range(3)] + [f"d_eight_p{n}" for n in range(4)]
    # d_nine is printed as a leaf, its flag alone; none of its customers is a co-user. The 7
    # others of d_eight are, the enrolled ones customers of the subgraph, and each brings its
    # account; p_card is no customer of d_eight.
    rows = {row.node_id: row for row in found.devices}
    assert rows["d_nine"].leaf and not rows["d_eight"].leaf
    assert rows["d_eight"].customers_90d == 8 and rows["d_eight"].customers_30d == 8
    assert {row.customer_id for row in found.customers} == {"p_s"} | {
        f"p_d_eight_e{n}" for n in range(3)
    }
    assert {row.account_id for row in found.accounts} == {f"a_{name}" for name in eight}
    # p_s's community steps through d_eight alone.
    assert _community(graph, ["a_s"])[0].members == 8


def test_a_node_that_is_no_leaf_has_at_most_8_customers_sharing_it() -> None:
    for world in (_crowds(), _identity_world()):
        graph = world.graph()
        for found in _identity(graph, _existing(graph), fixtures.HUBS):
            for row in (*found.devices, *found.ips):
                assert row.leaf or row.customers_30d <= row.customers_90d <= 8


# Labels, dates of record and same-second ids.


def _identified_by_payment() -> WorldBuilder:
    """a_m, the account of p_m, a mule the bank identified, takes a victim's money 10 days
    before t and passes it on 10 minutes later from the device d_x and the IP i_x, on which p_m
    never enrolled; p_n, whose account a_n has never paid, enrolled on both since joining. p_m
    shares d_x and i_x with p_n through its payment alone."""
    w = WorldBuilder()
    onset = T - 10 * D
    w.customer("m", JOINED, mule=Mule(onset, onset + 10 * M, 3, identified=True))
    w.customer("n", JOINED)
    w.external("x_vic", OPENED)
    w.external("x_out", OPENED)
    w.device("p_n", "d_x", JOINED)
    w.ip("p_n", "i_x", JOINED)
    w.pay("x_vic1", "x_vic", "a_m", onset, 900.0, "ach")
    w.pay("x_fwd1", "a_m", "x_out", onset + 10 * M, 850.0, "ach", device="d_x", ip="i_x")
    return w


def _cap_links() -> WorldBuilder:
    """p_k holds twelve device links, d_00 to d_11, and twelve IP links, i_00 to i_11, each of
    which began before the lookback, between 112 and 101 days before t, a day apart, the
    devices in the order of their ids and the IPs in the opposite one; so every start is
    unknown and none of them has a use. p_z, whose account the bank identified as a mule, is
    enrolled on d_11 and i_11, the last of each list. Each list is cut at 8 among ties."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.customer("z", JOINED, mule=Mule(T - 20 * D, T - 10 * D, 4, identified=True))
    w.device("p_z", "d_11", JOINED)
    w.ip("p_z", "i_11", JOINED)
    for n in range(12):
        w.device("p_k", f"d_{n:02d}", T - (112 - n) * D)
        w.ip("p_k", f"i_{n:02d}", T - (101 + n) * D)
    return w


def _cap_accounts() -> WorldBuilder:
    """p_k owns twelve more deposit accounts, o_00 to o_11, none of which has paid, so a_k
    keeps 8 of them among ties; o_11 is a mule account the bank identified."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    for n in range(12):
        mule = Mule(T - 20 * D, T - 10 * D, 4, identified=True) if n == 11 else None
        w.account(f"o_{n:02d}", "p_k", JOINED, mule=mule)
    return w


def _cap_seen() -> WorldBuilder:
    """a_k sent twelve payments in one millisecond 5 days before t, each seen on another
    device, d_00 to d_11, so its latest uses tie. On each device p_n, of the account a_n, is
    enrolled since joining and a_n paid from it 60 days before t; a_11 is a mule the bank
    identified."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    w.external("x_any", OPENED)
    for n in range(12):
        name = f"{n:02d}"
        mule = Mule(T - 20 * D, T - 10 * D, 4, identified=True) if n == 11 else None
        w.customer(name, JOINED, mule=mule)
        w.device(f"p_{name}", f"d_{name}", JOINED)
        w.pay(f"x_old_{name}", f"a_{name}", "x_any", T - 60 * D, 1.0, "ach", device=f"d_{name}")
        w.pay(f"x_k_{name}", "a_k", "x_any", T - 5 * D, 1.0, "ach", device=f"d_{name}")
    return w


def _cap_co_users() -> WorldBuilder:
    """p_k enrolled since joining on eight devices, d_0 to d_7, on each of which seven others,
    c_j_0 to c_j_6, are enrolled too, none of whom has paid: 56 co-user candidates with the
    same evidence, 8 more than a scored account keeps. c_7_6 is a mule the bank identified."""
    w = WorldBuilder()
    w.customer("k", JOINED)
    for d in range(8):
        w.device("p_k", f"d_{d}", JOINED)
        for c in range(7):
            mule = Mule(T - 20 * D, T - 10 * D, 4, identified=True) if (d, c) == (7, 6) else None
            w.customer(f"c_{d}_{c}", JOINED, mule=mule)
            w.device(f"p_c_{d}_{c}", f"d_{d}", JOINED)
    return w


# The worlds in which a cap cuts a list among ties, or an identified account shares a node
# through its payments alone: the catalogue's never does either, so a label or a date of record
# that decided a tie, or a mule's sharing, would show nowhere in it.
WORLDS: dict[str, Callable[[], WorldBuilder]] = {
    "the catalogue": _identity_world,
    "an identified mule sharing through payments alone": _identified_by_payment,
    "links cut among ties": _cap_links,
    "accounts cut among ties": _cap_accounts,
    "seen nodes cut among ties": _cap_seen,
    "co-users cut among ties": _cap_co_users,
}


@pytest.mark.parametrize("world", WORLDS)
def test_erasing_every_label_column_changes_no_answer(world: str) -> None:
    tables = WORLDS[world]().tables()
    erased = fixtures.erase_labels(tables)
    assert not erased["Account"].equals(tables["Account"])
    assert _printed(ReferenceGraph(erased)) == _printed(ReferenceGraph(tables))


@pytest.mark.parametrize("world", WORLDS)
def test_moving_dates_of_record_before_the_lookback_changes_no_answer(world: str) -> None:
    tables = WORLDS[world]().tables()
    moved = fixtures.move_dates_of_record(tables, T, LOOKBACK_MS)
    for name in ("Party", "Account", "Party_Owns_Account", "Party_Uses_Device", "Party_Uses_IP"):
        assert tables[name].empty or not moved[name].equals(tables[name])
    assert _printed(ReferenceGraph(moved)) == _printed(ReferenceGraph(tables))


def _renamed(graph: PaymentGraph, original: Mapping[str, str]) -> PaymentGraph:
    """A payment graph with each payment given back the id it had."""
    rows = sorted(
        (replace(row, payment_id=original[row.payment_id]) for row in graph.payments),
        key=lambda row: row.payment_id,
    )
    return replace(graph, payments=tuple(rows))


def test_swapping_the_ids_of_same_second_payments_changes_no_value() -> None:
    # No cap of the world cuts a second, so B4 gives the same subgraphs with the payments
    # renamed, and B5 and B6, which print no payment, the same bytes.
    tables: dict[str, pd.DataFrame] = _identity_world().tables()
    swapped, original = fixtures.swap_same_second_ids(tables)
    assert any(new != old for new, old in original.items())
    before, after = ReferenceGraph(tables), ReferenceGraph(swapped)
    accounts = _existing(before)
    renamed = tuple(
        _renamed(found, original) for found in _payments(after, accounts, fixtures.HUBS)
    )
    assert renamed == _payments(before, accounts, fixtures.HUBS)
    printed_before, printed_after = _printed(before), _printed(after)
    for name in (READ_SHARED_IDENTITY, READ_COMMUNITY):
        assert printed_after[name] == printed_before[name]
