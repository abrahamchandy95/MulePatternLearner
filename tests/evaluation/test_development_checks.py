"""The development checks beside the feature audit: no positive is an account hub, the leaf share
of ring devices and IPs, and whether busy ordinary accounts' card spending fills the short
dwell shares (docs/design.md, "The feature audit")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mule_pattern_learner.evaluation.development_checks import (
    OwnNodes,
    card_spending,
    flow_shares,
    own_nodes,
    positive_hubs,
    ring_leaf_share,
)
from mule_pattern_learner.subgraph.build import read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, Mule, T, WorldBuilder

W = 7 * DAY_MS


def test_an_active_mule_that_is_an_account_hub_is_found() -> None:
    rows = [(T, "a_m1"), (T, "a_m2"), (T + W, "a_m1")]
    hubs = {T: frozenset({"a_m2", "a_shop"}), T + W: frozenset({"a_shop"})}
    assert positive_hubs(rows, hubs).values.tolist() == [[T, "a_m2"]]
    assert positive_hubs(rows[:1], hubs).empty


def test_a_mules_own_nodes_from_its_rows(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: ["a_hr1", "a_o00"]}, parity=parity)
    hidden, ordinary = read_rows(cache, [(T, "a_hr1"), (T, "a_o00")])
    # a_hr1's customer is enrolled on the ring's device, which its payment was also seen on.
    assert own_nodes(hidden).nodes == {("device", "d_hidden"): False}
    assert own_nodes(ordinary).nodes == {}
    # a_o00 spent 20.00 by card, matched to no inflow; it sent one payment.
    assert flow_shares(ordinary) == ([0.0, 0.0, 0.0, 0.0, 0.0], 1.0)


def test_a_scored_accounts_own_nodes_are_only_those_of_its_own_edges(
    parity: Path, tmp_path: Path
) -> None:
    # a_s's customer is enrolled on d_enrolled and i_enrolled, which no payment of a_s was seen
    # on, and on i_crowd, which nine more customers are enrolled on (a leaf). a_s's one payment,
    # to a_c, was seen on d_seen and i_seen, which its customer is not enrolled on. a_c's
    # customer is enrolled on d_other and i_other: a_s's subgraph holds them on a_c's edges,
    # and they are not a_s's own.
    w = WorldBuilder()
    w.customer("s", fixtures.JOINED)
    w.customer("c", fixtures.JOINED)
    for node in ("i_enrolled", "i_crowd"):
        w.ip("p_s", node, T - 60 * DAY_MS)
    w.device("p_s", "d_enrolled", T - 60 * DAY_MS)
    w.device("p_c", "d_other", T - 60 * DAY_MS)
    w.ip("p_c", "i_other", T - 60 * DAY_MS)
    for index in range(9):
        w.customer(f"k{index}", fixtures.JOINED)
        w.ip(f"p_k{index}", "i_crowd", T - 60 * DAY_MS)
    w.pay("s_c", "a_s", "a_c", T - 3 * DAY_MS, 50.0, "ach", device="d_seen", ip="i_seen")
    cache = Cache(tmp_path / "cache")
    fetch(Spy(w.graph()), cache, {T: ["a_s"]}, parity=parity)
    (rows,) = read_rows(cache, [(T, "a_s")])
    in_subgraph = {node.node_id for node in (*rows.identity.devices, *rows.identity.ips)}
    assert {"d_other", "i_other"} <= in_subgraph
    assert own_nodes(rows).nodes == {
        ("device", "d_enrolled"): False,
        ("ip", "i_enrolled"): False,
        ("ip", "i_crowd"): True,
        ("device", "d_seen"): False,
        ("ip", "i_seen"): False,
    }


def test_the_leaf_share_of_ring_nodes() -> None:
    # Ring 1's two mules both reach d1 and i1 at T, i1 a leaf: two ring nodes, one leaf. Ring 2
    # has one mule at T, and a_m4 no ring, so neither makes a ring node.
    rows = [(T, "a_m1", 1), (T, "a_m2", 1), (T, "a_m3", 2), (T, "a_m4", -1), (T + W, "a_m1", 1)]
    nodes = [
        OwnNodes({("device", "d1"): False, ("ip", "i1"): True}),
        OwnNodes({("device", "d1"): False, ("ip", "i1"): True}),
        OwnNodes({("device", "d1"): False}),
        OwnNodes({("device", "d1"): False}),
        OwnNodes({("ip", "i1"): True}),
    ]
    found = ring_leaf_share(rows, nodes)
    assert found.to_dict("records") == [{"ring_nodes": 2, "leaves": 1, "leaf_share": 0.5}]
    assert np.isnan(ring_leaf_share(rows[2:4], nodes[2:4])["leaf_share"].iloc[0])


def test_a_node_reached_by_mules_of_two_rings_is_no_ring_node() -> None:
    # Each ring has one active mule on d1: neither ring has two.
    rows = [(T, "a_m1", 1), (T, "a_m2", 2)]
    nodes = [OwnNodes({("device", "d1"): False})] * 2
    found = ring_leaf_share(rows, nodes)
    assert found["ring_nodes"].tolist() == [0] and np.isnan(found["leaf_share"].iloc[0])


def test_a_node_reached_by_two_ringless_mules_is_no_ring_node() -> None:
    # A ringless mule (ring -1) belongs to no ring, so two of them are not a ring's two mules.
    rows = [(T, "a_m1", -1), (T, "a_m2", -1)]
    nodes = [OwnNodes({("device", "d1"): False})] * 2
    found = ring_leaf_share(rows, nodes)
    assert found["ring_nodes"].tolist() == [0] and np.isnan(found["leaf_share"].iloc[0])


def test_two_accounts_of_one_customer_count_as_two_mules(parity: Path, tmp_path: Path) -> None:
    # One customer holds two mule accounts of ring 7 and is enrolled on its home IP, which each
    # account copies as its own. The ring's nodes count accounts, as the design states, so the
    # IP is the ring's although no second customer reaches it.
    w = WorldBuilder()
    w.party("p_p", fixtures.JOINED)
    for account in ("a_p1", "a_p2"):
        w.account(account, "p_p", fixtures.JOINED, mule=Mule(T - 5 * DAY_MS, T, ring=7))
    w.ip("p_p", "i_home", T - 60 * DAY_MS)
    cache = Cache(tmp_path / "cache")
    fetch(Spy(w.graph()), cache, {T: ["a_p1", "a_p2"]}, parity=parity)
    rows = read_rows(cache, [(T, "a_p1"), (T, "a_p2")])
    nodes = [own_nodes(row) for row in rows]
    assert [held.nodes for held in nodes] == [{("ip", "i_home"): False}] * 2
    found = ring_leaf_share([(T, "a_p1", 7), (T, "a_p2", 7)], nodes)
    assert found.to_dict("records") == [{"ring_nodes": 1, "leaves": 0, "leaf_share": 0.0}]


def test_the_card_spending_report() -> None:
    # Ten ordinary account-Mondays sending 1 to 10 payments, and two mules. The busy ones are
    # those at or above the 90th percentile by weight, 9 payments (9 of the 10 send at most 9):
    # the one sending 9, with nothing matched, and the one sending 10, whose left_1d is 0.8,
    # 0.5 of it by card.
    out_30d = np.array([*range(1, 11), 3, 4], dtype=np.float64)
    positive = np.array([False] * 10 + [True] * 2)
    shares = np.zeros((12, 5))
    shares[9] = [0.1, 0.2, 0.4, 0.8, 0.5]
    shares[10:] = [0.3, 0.6, 0.9, 1.0, 0.0]
    found = card_spending(shares, out_30d, positive, np.ones(12)).set_index("group")
    assert found.loc["busy_ordinary", "account_mondays"] == 2
    assert found.loc["busy_ordinary", "left_1d"] == pytest.approx(0.4)
    assert found.loc["busy_ordinary", "card_part_of_left_1d"] == pytest.approx(0.2)
    assert found.loc["ordinary", "left_1d"] == pytest.approx(0.08)
    assert found.loc["mules", "left_15m"] == pytest.approx(0.3)
    assert found.loc["mules", "card_part_of_left_1d"] == 0
