"""The reference graph on the standard world (testing.world) and on small worlds built here.

Every answer goes through the parsers of tigergraph.queries and evaluation.answer_key, so it
has the format the GSQL prints. The values are worked out by hand from docs/design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, SubgraphConfig
from mule_pattern_learner.evaluation.answer_key import AnswerKeyRow, parse_answer_key
from mule_pattern_learner.testing.reference_graph import DAY_MS, TABLES, ReferenceGraph
from mule_pattern_learner.testing.world import (
    HOUR_MS,
    LATER,
    Mule,
    T,
    WorldBuilder,
    standard_world,
)
from mule_pattern_learner.tigergraph.executor import account_pages
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    LIST_POPULATION,
    READ_ANSWER_KEY,
    READ_SUPERVISION,
    WRITE_SPLIT,
    AccountNode,
    IdentityNode,
    LinkEdge,
    PartyNode,
    PaymentEdge,
    Phase,
    PopulationAccount,
    SentEdge,
    SplitWritten,
    Subgraph,
    SubgraphBatch,
    SubgraphRequest,
    Supervision,
    parse_population,
    parse_split,
    parse_subgraphs,
    parse_supervision,
    population_params,
    split_params,
    subgraph_params,
    supervision_params,
)

D, H = DAY_MS, HOUR_MS
CONFIG = DEFAULT_CONFIG.subgraph
LABELS = {
    "discovered_ms",
    "discovery_channel",
    "mule_onset_ms",
    "is_mule",
    "mule_ring_id",
    "mule_layer",
    "mule_flow",
    "mule_last_ms",
}


def _fetch(
    graph: ReferenceGraph,
    seeds: list[str],
    *,
    t: int = T,
    phase: Phase = Phase.TRAINING,
    config: SubgraphConfig = CONFIG,
) -> SubgraphBatch:
    request = SubgraphRequest(tuple(seeds), t, phase)
    return parse_subgraphs(graph.run(FETCH_SUBGRAPHS, subgraph_params(request, config)), request)


def _subgraph(
    seed: str,
    *,
    t: int = T,
    phase: Phase = Phase.TRAINING,
    config: SubgraphConfig = CONFIG,
    graph: ReferenceGraph | None = None,
) -> Subgraph:
    """The seed's subgraph, by default in the standard world at T in training."""
    graph = graph or standard_world().graph()
    return _fetch(graph, [seed], t=t, phase=phase, config=config).subgraphs[0]


def _ids(items: tuple[AccountNode | PartyNode | IdentityNode | PaymentEdge, ...]) -> set[str]:
    return {item.id for item in items}


def _pairs(edges: tuple[LinkEdge | SentEdge, ...]) -> set[tuple[str, str]]:
    return {(edge.from_id, edge.to_id) for edge in edges}


def _by_id[N: (AccountNode, PartyNode, IdentityNode, PaymentEdge)](
    items: tuple[N, ...], item_id: str
) -> N:
    [found] = [item for item in items if item.id == item_id]
    return found


def _edge[E: (LinkEdge, SentEdge)](edges: tuple[E, ...], from_id: str, to_id: str) -> E:
    [found] = [edge for edge in edges if (edge.from_id, edge.to_id) == (from_id, to_id)]
    return found


def _answers(graph: ReferenceGraph) -> list[tuple[SubgraphRequest, list[dict[str, Any]]]]:
    """fetch_subgraphs of every account opened by t that the phase sees, at T and LATER, in
    both phases."""
    answers: list[tuple[SubgraphRequest, list[dict[str, Any]]]] = []
    for t in (T, LATER):
        for phase in Phase:
            seeds = sorted(
                account.id
                for account in graph.accounts.values()
                if account.opened_ms <= t
                and (phase.held_out_visible or graph.account_split[account.id] != 1)
            )
            request = SubgraphRequest(tuple(seeds), t, phase)
            answers.append((request, graph.run(FETCH_SUBGRAPHS, subgraph_params(request, CONFIG))))
    return answers


def test_a_mules_subgraph_holds_its_payments_their_accounts_and_its_ring_device() -> None:
    """Vic paid Mia, who paid Max (hop 0); Max (hop 1) adds his cash-out to X_CASH. Mia's
    device D_RING0 adds Quy, linked to it, and Max, linked to it and seen sending from it."""
    sub = _subgraph("A_MIA")
    assert _ids(sub.accounts) == {"A_MIA", "A_VIC", "A_MAX", "X_CASH", "A_QUY"}
    assert _ids(sub.parties) == {"P_MIA", "P_VIC", "P_MAX"}
    assert (_ids(sub.devices), _ids(sub.ips), _ids(sub.addresses)) == (
        {"D_RING0", "D_VIC"},
        {"I_RING0", "I_VIC"},
        set[str](),
    )
    assert _ids(sub.pays) == {"T08", "T09", "T10"}
    assert _pairs(sub.owned_by) == {("A_MIA", "P_MIA"), ("A_VIC", "P_VIC"), ("A_MAX", "P_MAX")}
    assert _pairs(sub.enrolled_on_device) == {
        ("A_MIA", "D_RING0"),
        ("A_MAX", "D_RING0"),
        ("A_QUY", "D_RING0"),
        ("A_VIC", "D_VIC"),
    }
    assert _pairs(sub.sent_from_device) == {
        ("A_MIA", "D_RING0"),
        ("A_MAX", "D_RING0"),
        ("A_VIC", "D_VIC"),
    }
    assert _pairs(sub.sent_from_ip) == {("A_MIA", "I_RING0"), ("A_VIC", "I_VIC")}
    assert not sub.enrolled_on_ip and not sub.at_address


def test_a_payment_counts_from_the_window_start_until_just_before_t() -> None:
    """T01 is at the window's start, T02 a second before it, and T05 at t itself."""
    sub = _subgraph("A_ANN")
    assert _ids(sub.pays) == {"T01", "T03", "T04", "T06"}
    ann = _by_id(sub.accounts, "A_ANN")
    assert (ann.out_7d, ann.out_30d, ann.out_365d) == (1, 1, 3)
    assert (ann.in_7d, ann.in_30d, ann.in_365d) == (0, 1, 1)
    ben = _by_id(sub.accounts, "A_BEN")
    assert (ben.in_365d, ben.out_365d, ben.last_out_present) == (3, 0, False)
    short = _subgraph("A_ANN", config=replace(CONFIG, lookback_days=30))
    assert _ids(short.pays) == {"T04", "T06"}
    assert _by_id(short.accounts, "A_ANN").out_365d == 1
    assert _by_id(short.pays, "T04").first_in_pair


def test_a_link_counts_from_its_start_and_one_that_ended_stays_as_a_past_fact() -> None:
    ann = _subgraph("A_ANN")
    # Ben enrolled D_BEN at t itself.
    assert _edge(ann.enrolled_on_device, "A_BEN", "D_BEN") == LinkEdge(
        "A_BEN", "D_BEN", 0, True, 0, 0
    )
    assert _by_id(ann.devices, "D_BEN") == IdentityNode("D_BEN", 0, 1, 1, 1, 1, False, False)
    # Old's account closed 20 days before t; its ownership stays, ended.
    assert _edge(ann.owned_by, "A_OLD", "P_OLD") == LinkEdge(
        "A_OLD", "P_OLD", 500 * D, False, 20 * D, 480 * D
    )
    assert _by_id(ann.parties, "P_OLD") == PartyNode("P_OLD", 500 * D, 0, 1)
    assert _by_id(ann.accounts, "A_OLD").closed
    # Roy's link ends after t, so at t it is active and New's, which starts later, is not
    # there. Four weeks later Roy's has ended and New's is there.
    roy = _subgraph("A_ROY")
    assert _edge(roy.enrolled_on_device, "A_ROY", "D_RING1") == LinkEdge(
        "A_ROY", "D_RING1", 45 * D, True, 0, 45 * D
    )
    assert _ids(roy.accounts) == {"A_ROY", "A_SUE"}
    assert _by_id(roy.devices, "D_RING1") == IdentityNode(
        "D_RING1", 45 * D, 2, 2, 2, 0, False, False
    )
    later = _subgraph("A_ROY", t=LATER)
    assert _edge(later.enrolled_on_device, "A_ROY", "D_RING1") == LinkEdge(
        "A_ROY", "D_RING1", 73 * D, False, 23 * D, 50 * D
    )
    assert _ids(later.accounts) == {"A_ROY", "A_SUE", "A_NEW"}
    assert _by_id(later.devices, "D_RING1") == IdentityNode(
        "D_RING1", 73 * D, 2, 3, 3, 1, False, False
    )
    # Mia's link to the ring's device ended 9 days before t; Quy still sees her through it.
    quy = _subgraph("A_QUY")
    assert _edge(quy.enrolled_on_device, "A_MIA", "D_RING0") == LinkEdge(
        "A_MIA", "D_RING0", 62 * D, False, 9 * D, 53 * D
    )


def test_a_link_ownership_or_token_that_starts_after_t_is_not_there_yet() -> None:
    """A week after t Ann binds a third token, enrolls D_ANN_NEW and opens A_ANN_NEW; three
    days after t Ben binds his only token."""
    graph = standard_world().graph()
    ann = _subgraph("A_ANN", graph=graph)
    assert _ids(ann.accounts) == {"A_ANN", "A_ANN_CARD", "A_BEN", "A_OLD"}
    assert _ids(ann.devices) == {"D_ANN", "D_BEN"}
    assert _by_id(ann.parties, "P_ANN") == PartyNode("P_ANN", 400 * D, 2, 0)
    # The token age is the earliest token's, and a token bound after t is not one yet.
    row, ben = _by_id(ann.accounts, "A_ANN"), _by_id(ann.accounts, "A_BEN")
    assert (row.token_present, row.token_age_ms) == (True, 200 * D)
    assert (ben.token_present, ben.token_age_ms) == (False, 0)
    later = _subgraph("A_ANN", t=LATER, graph=graph)
    assert "A_ANN_NEW" in _ids(later.accounts)
    assert _edge(later.enrolled_on_device, "A_ANN", "D_ANN_NEW") == LinkEdge(
        "A_ANN", "D_ANN_NEW", 21 * D, True, 0, 21 * D
    )
    assert _by_id(later.devices, "D_ANN_NEW") == IdentityNode(
        "D_ANN_NEW", 21 * D, 1, 1, 3, 1, False, False
    )
    assert _by_id(later.parties, "P_ANN") == PartyNode("P_ANN", 428 * D, 3, 0)
    row, ben = _by_id(later.accounts, "A_ANN"), _by_id(later.accounts, "A_BEN")
    assert (row.token_present, row.token_age_ms) == (True, 228 * D)
    assert (ben.token_present, ben.token_age_ms) == (True, 25 * D)


def test_held_out_customers_are_hidden_in_training_even_behind_a_shared_device() -> None:
    """Hal is held out: he holds D_RING0, as Quy does, paid Mia on it (T11) and was paid by
    Mia on it (T14)."""
    graph = standard_world().graph()
    ring = IdentityNode("D_RING0", 62 * D, 2, 3, 3, 0, False, False)
    quy = _subgraph("A_QUY", graph=graph)
    assert _ids(quy.accounts) == {"A_QUY", "A_MIA", "A_MAX"}
    assert _by_id(quy.devices, "D_RING0") == ring
    seen = _subgraph("A_QUY", phase=Phase.TEST, graph=graph)
    assert _ids(seen.accounts) == {"A_QUY", "A_MIA", "A_MAX", "A_HAL"}
    assert _by_id(seen.devices, "D_RING0") == IdentityNode(
        "D_RING0", 62 * D, 3, 4, 4, 0, False, False
    )
    assert ("A_HAL", "D_RING0") in _pairs(seen.enrolled_on_device) & _pairs(seen.sent_from_device)
    # Neither Hal's payment to Mia nor hers to him is there in training.
    mia = _subgraph("A_MIA", graph=graph)
    assert not {"T11", "T14"} & _ids(mia.pays) and "A_HAL" not in _ids(mia.accounts)
    row = _by_id(mia.accounts, "A_MIA")
    assert (row.in_30d, row.out_30d) == (1, 1)
    assert _by_id(mia.devices, "D_RING0") == ring
    mia_seen = _subgraph("A_MIA", phase=Phase.TEST, graph=graph)
    assert {"T11", "T14"} <= _ids(mia_seen.pays) and "A_HAL" in _ids(mia_seen.accounts)
    assert "P_HAL" in _ids(mia_seen.parties)
    row = _by_id(mia_seen.accounts, "A_MIA")
    assert (row.in_30d, row.amount_in_30d, row.counterparties_in_30d) == (2, 1200.0, 2)
    assert (row.out_30d, row.amount_out_30d, row.counterparties_out_30d) == (2, 880.0, 2)
    with pytest.raises(ValueError, match="invisible_seed"):
        _subgraph("A_HAL", graph=graph)
    assert _subgraph("A_HAL", phase=Phase.TEST, graph=graph).seed == "A_HAL"


def test_payment_caps_keep_the_newest_payments_and_leave_the_features_whole() -> None:
    sub = _subgraph("A_ANN", config=replace(CONFIG, seed_payments=1, other_payments=1))
    assert _ids(sub.pays) == {"T04", "T06"}
    assert _by_id(sub.accounts, "A_ANN").out_365d == 3


def test_caps_break_ties_by_id() -> None:
    """S paid R1 and R2 in the same millisecond, each payment seen on its own device, and
    enrolled two devices on one day."""
    w = WorldBuilder()
    for party, account in (("P_S", "S"), ("P_1", "R1"), ("P_2", "R2")):
        w.party(party, T - 100 * D)
        w.account(account, party, T - 100 * D)
    w.device("P_S", "D_OLD", T - 90 * D)
    w.device("P_S", "D_B", T - 50 * D)
    w.device("P_S", "D_A", T - 50 * D)
    w.pay("PAY_B", "S", "R2", T - D, 10.0, "ach", device="X_A")
    w.pay("PAY_A", "S", "R1", T - D, 10.0, "ach", device="X_B")
    graph = w.graph()
    caps = replace(CONFIG, seed_payments=1, enrolled_devices=1, sent_devices=1)
    sub = _subgraph("S", graph=graph, config=caps)
    assert _ids(sub.pays) == {"PAY_A"} and _ids(sub.accounts) == {"S", "R1"}
    assert _pairs(sub.enrolled_on_device) == {("S", "D_A")}
    assert _pairs(sub.sent_from_device) == {("S", "X_A")}
    whole = _subgraph("S", graph=graph)
    assert _ids(whole.devices) == {"D_OLD", "D_A", "D_B", "X_A", "X_B"}


def test_sent_caps_keep_the_nodes_used_last() -> None:
    """S sent from X 30 days and a day before t, and from Y 10 days before t."""
    w = WorldBuilder()
    w.party("P_S", T - 100 * D)
    w.account("S", "P_S", T - 100 * D)
    w.account("R", None, T - 100 * D)
    w.pay("PAY_1", "S", "R", T - 30 * D, 10.0, "ach", device="X")
    w.pay("PAY_2", "S", "R", T - 10 * D, 10.0, "ach", device="Y")
    w.pay("PAY_3", "S", "R", T - D, 10.0, "ach", device="X")
    sub = _subgraph("S", graph=w.graph(), config=replace(CONFIG, sent_devices=1))
    assert _ids(sub.devices) == {"X"}
    assert sub.sent_from_device == (SentEdge("S", "X", 30 * D, D, 2),)


def test_owner_and_sibling_caps_keep_the_newest_ownership() -> None:
    # Ann's newest account is her card: with one sibling her deposit account brings it, and
    # the card brings nothing more, its owner's newest account being itself.
    narrow = replace(CONFIG, siblings=1)
    assert "A_ANN_CARD" in _ids(_subgraph("A_ANN", config=narrow).accounts)
    assert "A_ANN" not in _ids(_subgraph("A_ANN_CARD", config=narrow).accounts)
    assert "A_ANN" in _ids(_subgraph("A_ANN_CARD").accounts)
    # P_1 opened J; P_3 and P_2 joined it on one day, P_3's link first in the table, and P_4
    # joins after t. P_2 opened K_B and K_A on one day, K_B first.
    w = WorldBuilder()
    for party in ("P_1", "P_2", "P_3", "P_4"):
        w.party(party, T - 100 * D)
    w.account("J", "P_1", T - 100 * D)
    w.link("Party_Owns_Account", "P_3", "J", T - 50 * D)
    w.link("Party_Owns_Account", "P_2", "J", T - 50 * D)
    w.link("Party_Owns_Account", "P_4", "J", T + 7 * D)
    w.account("K_B", "P_2", T - 20 * D)
    w.account("K_A", "P_2", T - 20 * D)
    graph = w.graph()

    def picked(t: int = T, **caps: int) -> tuple[set[str], set[str]]:
        sub = _subgraph("J", t=t, graph=graph, config=replace(CONFIG, **caps))
        return _ids(sub.parties), _ids(sub.accounts)

    assert picked() == ({"P_2"}, {"J", "K_A", "K_B"})
    assert picked(siblings=1) == ({"P_2"}, {"J", "K_A"})
    assert picked(owners=2)[0] == {"P_2", "P_3"}
    assert picked(owners=3)[0] == {"P_1", "P_2", "P_3"}
    assert picked(LATER)[0] == {"P_4"}


def test_identity_nodes_add_other_customers_deposit_accounts_only() -> None:
    """Ben's home IP and address bring Ann's deposit account, his counterparty already, and
    not her credit card."""
    ben = _subgraph("A_BEN")
    assert _ids(ben.accounts) == {"A_BEN", "A_ANN", "A_OLD"}
    assert _pairs(ben.enrolled_on_ip) == {("A_BEN", "I_HOME"), ("A_ANN", "I_HOME")}
    assert _pairs(ben.at_address) == {("A_BEN", "H_HOME"), ("A_ANN", "H_HOME")}


def test_identity_caps_take_the_newest_accounts_per_node_and_in_all() -> None:
    """In the test phase D_RING0 offers Quy, newest first: Max, Mia and Hal seen sending
    from it (last 8, 9 and 20 days before t), then Hal, Max and Mia linked to it (50, 58
    and 62 days before t)."""
    per_node = _subgraph("A_QUY", phase=Phase.TEST, config=replace(CONFIG, reverse_accounts=1))
    assert _ids(per_node.accounts) == {"A_QUY", "A_MAX", "A_HAL"}
    assert _pairs(per_node.sent_from_device) == {("A_MAX", "D_RING0")}
    assert _pairs(per_node.enrolled_on_device) == {("A_QUY", "D_RING0"), ("A_HAL", "D_RING0")}
    in_all = _subgraph("A_QUY", phase=Phase.TEST, config=replace(CONFIG, identity_accounts=2))
    assert _ids(in_all.accounts) == {"A_QUY", "A_MAX", "A_MIA"}
    assert _pairs(in_all.sent_from_device) == {("A_MAX", "D_RING0"), ("A_MIA", "D_RING0")}
    assert _pairs(in_all.enrolled_on_device) == {
        ("A_QUY", "D_RING0"),
        ("A_MAX", "D_RING0"),
        ("A_MIA", "D_RING0"),
    }


def test_an_identity_node_with_more_visible_owners_than_the_threshold_is_a_leaf() -> None:
    """In training Mia, Max and Quy own D_RING0's links and payments; in the test phase
    Hal too."""
    config = replace(CONFIG, hub_threshold=3)
    assert _ids(_subgraph("A_QUY", config=config).accounts) == {"A_QUY", "A_MIA", "A_MAX"}
    seen = _subgraph("A_QUY", phase=Phase.TEST, config=config)
    assert _ids(seen.accounts) == {"A_QUY"}
    assert _pairs(seen.enrolled_on_device) == {("A_QUY", "D_RING0")}
    assert _by_id(seen.devices, "D_RING0") == IdentityNode(
        "D_RING0", 62 * D, 3, 4, 4, 0, False, True
    )


def test_an_identity_node_counts_the_owners_of_accounts_observed_on_it() -> None:
    """The carrier IP I_NAT is linked to no one; S and C, of two parties, each paid on it."""
    w = WorldBuilder()
    for name in ("S", "C"):
        w.party(f"P_{name}", T - 100 * D)
        w.account(name, f"P_{name}", T - 100 * D)
        w.account(f"R_{name}", None, T - 100 * D)
        w.pay(f"PAY_{name}", name, f"R_{name}", T - D, 10.0, "ach", ip="I_NAT")
    graph = w.graph()
    assert _ids(_subgraph("S", graph=graph).accounts) == {"S", "R_S", "C"}
    hub = _subgraph("S", graph=graph, config=replace(CONFIG, hub_threshold=1))
    assert _ids(hub.accounts) == {"S", "R_S"}
    assert _pairs(hub.sent_from_ip) == {("S", "I_NAT")}
    assert hub.ips == (IdentityNode("I_NAT", D, 0, 0, 2, 0, True, True),)


def _biller_world() -> WorldBuilder:
    """S pays the ownerless biller B, as C1 (held out), C2 and C3 do."""
    w = WorldBuilder()
    w.account("B", None, T - 100 * D)
    for name in ("S", "C1", "C2", "C3"):
        w.party(f"P_{name}", T - 100 * D, held_out=name == "C1")
        w.account(name, f"P_{name}", T - 100 * D)
        w.pay(f"PAY_{name}", name, "B", T - D, 10.0, "ach")
    return w


def test_a_counterparty_paid_by_more_visible_owners_than_the_threshold_is_a_leaf() -> None:
    graph = _biller_world().graph()
    config = replace(CONFIG, hub_threshold=3)
    hub = _subgraph("S", phase=Phase.TEST, graph=graph, config=config)
    assert _ids(hub.accounts) == {"S", "B"} and _ids(hub.pays) == {"PAY_S"}
    assert _by_id(hub.accounts, "B").in_30d == 4
    assert _ids(_subgraph("S", graph=graph, config=config).accounts) == {"S", "B", "C2", "C3"}
    four = replace(CONFIG, hub_threshold=4)
    assert len(_subgraph("S", phase=Phase.TEST, graph=graph, config=four).accounts) == 5


def test_a_counterparty_with_no_more_payments_than_the_threshold_is_never_a_hub() -> None:
    """C has two payments, though the account it paid has three owners."""
    w = WorldBuilder()
    w.party("P_S", T - 100 * D)
    w.account("S", "P_S", T - 100 * D)
    w.account("C", None, T - 100 * D)
    for owner in ("P_1", "P_2", "P_3"):
        w.party(owner, T - 100 * D)
    w.account("J", "P_1", T - 100 * D)
    w.link("Party_Owns_Account", "P_2", "J", T - 100 * D)
    w.link("Party_Owns_Account", "P_3", "J", T - 100 * D)
    w.pay("PAY_1", "S", "C", T - 2 * D, 10.0, "ach")
    w.pay("PAY_2", "C", "J", T - D, 10.0, "ach")
    sub = _subgraph("S", graph=w.graph(), config=replace(CONFIG, hub_threshold=2))
    assert _ids(sub.accounts) == {"S", "C", "J"}


def test_every_duration_is_measured_back_from_t_or_between_events() -> None:
    mia = _subgraph("A_MIA")
    accounts = {
        node.id: (node.age_ms, node.token_age_ms, node.since_last_in_ms, node.since_last_out_ms)
        for node in mia.accounts
    }
    assert accounts == {
        "A_MIA": (300 * D, 0, 10 * D, 10 * D - 2 * H),
        "A_VIC": (365 * D, 0, 0, 10 * D),
        "A_MAX": (250 * D, 0, 10 * D - 2 * H, 8 * D),
        "X_CASH": (8 * D, 0, 8 * D, 0),
        "A_QUY": (90 * D, 0, 0, 0),
    }
    assert {party.id: party.age_ms for party in mia.parties} == {
        "P_MIA": 300 * D,
        "P_VIC": 365 * D,
        "P_MAX": 250 * D,
    }
    assert {node.id: node.age_ms for node in (*mia.devices, *mia.ips)} == {
        "D_RING0": 62 * D,
        "D_VIC": 365 * D,
        "I_RING0": 10 * D - 2 * H,
        "I_VIC": 10 * D,
    }
    pays = {pay.id: (pay.age_ms, pay.next_outflow_present, pay.next_outflow_ms) for pay in mia.pays}
    assert pays == {
        "T08": (10 * D, True, 2 * H),
        "T09": (10 * D - 2 * H, True, 2 * D - 2 * H),
        "T10": (8 * D, False, 0),
    }
    links = {
        (edge.from_id, edge.to_id): (
            edge.since_start_ms,
            edge.active,
            edge.since_end_ms,
            edge.held_ms,
        )
        for edge in (*mia.owned_by, *mia.enrolled_on_device)
    }
    assert links == {
        ("A_MIA", "P_MIA"): (300 * D, False, 5 * D, 295 * D),
        ("A_VIC", "P_VIC"): (365 * D, True, 0, 365 * D),
        ("A_MAX", "P_MAX"): (250 * D, True, 0, 250 * D),
        ("A_MIA", "D_RING0"): (62 * D, False, 9 * D, 53 * D),
        ("A_MAX", "D_RING0"): (58 * D, True, 0, 58 * D),
        ("A_QUY", "D_RING0"): (45 * D, True, 0, 45 * D),
        ("A_VIC", "D_VIC"): (365 * D, True, 0, 365 * D),
    }
    sent = {
        (edge.from_id, edge.to_id): (edge.since_first_ms, edge.since_last_ms, edge.uses)
        for edge in (*mia.sent_from_device, *mia.sent_from_ip)
    }
    assert sent == {
        ("A_MIA", "D_RING0"): (10 * D - 2 * H, 10 * D - 2 * H, 1),
        ("A_MAX", "D_RING0"): (8 * D, 8 * D, 1),
        ("A_VIC", "D_VIC"): (10 * D, 10 * D, 1),
        ("A_MIA", "I_RING0"): (10 * D - 2 * H, 10 * D - 2 * H, 1),
        ("A_VIC", "I_VIC"): (10 * D, 10 * D, 1),
    }
    ann = _subgraph("A_ANN")
    row = _by_id(ann.accounts, "A_ANN")
    assert (row.token_present, row.token_age_ms) == (True, 200 * D)
    # T02, the pair's oldest payment, is before the window.
    pairs = {
        pay.id: (pay.prior_count, pay.first_in_pair, pay.gap_present, pay.gap_ms, pay.pair_age_ms)
        for pay in ann.pays
    }
    assert pairs == {
        "T01": (0, True, False, 0, 0),
        "T03": (1, False, True, 325 * D, 325 * D),
        "T04": (2, False, True, 34 * D, 359 * D),
        "T06": (0, True, False, 0, 0),
    }
    assert _by_id(ann.pays, "T06").next_outflow_ms == 19 * D
    assert _edge(ann.sent_from_device, "A_ANN", "D_ANN") == SentEdge(
        "A_ANN", "D_ANN", 365 * D, 6 * D, 3
    )


def test_account_features_follow_the_visible_payments() -> None:
    mia_subgraph = _subgraph("A_MIA")
    mia = _by_id(mia_subgraph.accounts, "A_MIA")
    # Vic's 900 reached Mia ten days before t, a first from Vic, and she passed it on within
    # two hours.
    assert (mia.amount_in_30d, mia.amount_out_30d, mia.passed_on_30d) == (900.0, 870.0, 900.0)
    assert (mia.first_inflows_7d, mia.first_inflows_30d) == (0, 1)
    assert (mia.counterparties_in_30d, mia.counterparties_out_30d) == (1, 1)
    cash = _by_id(mia_subgraph.accounts, "X_CASH")
    assert (cash.is_external, cash.account_type, cash.in_7d, cash.in_30d) == (
        True,
        "unknown",
        0,
        1,
    )
    ann = _subgraph("A_ANN")
    # Ann paid Ben earlier in the window, so her latest payment is no first inflow. Of her
    # payments to him only T04 (80) is in the last 30 days.
    ben = _by_id(ann.accounts, "A_BEN")
    assert (ben.in_30d, ben.first_inflows_30d, ben.passed_on_30d) == (1, 0, 0.0)
    assert ben.amount_in_30d == _by_id(ann.accounts, "A_ANN").amount_out_30d == 80.0
    card = _by_id(ann.accounts, "A_ANN_CARD")
    assert (card.account_type, card.out_30d, card.amount_out_30d) == ("credit", 1, 25.0)
    assert _by_id(ann.parties, "P_ANN") == PartyNode("P_ANN", 400 * D, 2, 0)
    home = _by_id(ann.ips, "I_HOME")
    assert home == IdentityNode("I_HOME", 400 * D, 2, 2, 3, 0, False, False)


def _flow_world() -> WorldBuilder:
    """Z's payments on either side of the 30-day window and of the 24-hour pass-on.

    - C1 paid Z 100 forty days before t; Z passed it on to R1 an hour later.
    - C2 paid Z 200 ten days before t; Z passed it on to R2 exactly 24 hours later.
    - C3 paid Z 30 and 20 in one millisecond five days before t, and Z paid R2 10 in that
      same millisecond.
    """
    w = WorldBuilder()
    w.party("P_Z", T - 100 * D)
    w.account("Z", "P_Z", T - 100 * D)
    for other in ("C1", "C2", "C3", "R1", "R2"):
        w.account(other, None, T - 100 * D)
    w.pay("IN_1", "C1", "Z", T - 40 * D, 100.0, "ach")
    w.pay("OUT_1", "Z", "R1", T - 40 * D + H, 90.0, "ach")
    w.pay("IN_2", "C2", "Z", T - 10 * D, 200.0, "ach")
    w.pay("OUT_2", "Z", "R2", T - 9 * D, 150.0, "ach")
    w.pay("IN_3B", "C3", "Z", T - 5 * D, 20.0, "ach")
    w.pay("IN_3A", "C3", "Z", T - 5 * D, 30.0, "ach")
    w.pay("OUT_3", "Z", "R2", T - 5 * D, 10.0, "ach")
    return w


def test_30_day_features_leave_out_older_payments_and_24_hours_is_within_a_day() -> None:
    sub = _subgraph("Z", graph=_flow_world().graph())
    assert _by_id(sub.accounts, "Z") == AccountNode(
        id="Z",
        account_type="deposit",
        is_external=False,
        age_ms=100 * D,
        token_present=False,
        token_age_ms=0,
        in_7d=2,
        in_30d=3,
        in_365d=4,
        out_7d=1,
        out_30d=2,
        out_365d=3,
        amount_in_30d=250.0,
        amount_out_30d=160.0,
        counterparties_in_30d=2,
        counterparties_out_30d=1,
        first_inflows_7d=2,
        first_inflows_30d=3,
        passed_on_30d=250.0,
        last_in_present=True,
        since_last_in_ms=5 * D,
        last_out_present=True,
        since_last_out_ms=5 * D,
        closed=False,
    )
    # Two payments of a pair in one millisecond are not prior to each other, and an outflow
    # in the same millisecond is the next one.
    pays = {
        pay.id: (pay.prior_count, pay.first_in_pair, pay.next_outflow_present, pay.next_outflow_ms)
        for pay in sub.pays
    }
    assert pays == {
        "IN_1": (0, True, True, H),
        "OUT_1": (0, True, False, 0),
        "IN_2": (0, True, True, D),
        "OUT_2": (0, True, False, 0),
        "IN_3A": (0, True, True, 0),
        "IN_3B": (0, True, True, 0),
        "OUT_3": (1, False, False, 0),
    }


def _boundary_world() -> WorldBuilder:
    """P has held A since 100 days before t; A_SHUT, also P's, closed at t, and A_OPEN
    opened at t. P used the device D from 60 days before t until t, and Q, who owns A_Q, has
    used it since exactly 30 days before t."""
    w = WorldBuilder()
    for party in ("P", "Q"):
        w.party(party, T - 200 * D)
    w.account("A", "P", T - 100 * D)
    w.account("A_SHUT", "P", T - 100 * D, closed_ms=T)
    w.account("A_OPEN", "P", T)
    w.account("A_Q", "Q", T - 200 * D)
    w.device("P", "D", T - 60 * D, T)
    w.device("Q", "D", T - 30 * D)
    return w


def test_what_ends_at_t_has_ended_and_what_starts_at_t_has_started() -> None:
    graph = _boundary_world().graph()
    sub = _subgraph("A", graph=graph)
    assert _ids(sub.accounts) == {"A", "A_SHUT", "A_OPEN", "A_Q"}
    assert [_by_id(sub.accounts, account).closed for account in ("A", "A_SHUT")] == [False, True]
    assert _by_id(sub.accounts, "A_OPEN").age_ms == 0
    assert _by_id(sub.parties, "P") == PartyNode("P", 200 * D, 2, 1)
    assert _edge(sub.owned_by, "A_SHUT", "P") == LinkEdge("A_SHUT", "P", 100 * D, False, 0, 100 * D)
    # P's link to D ended at t, so P is not among its parties now; Q's link, exactly 30 days
    # old, makes Q a new party.
    assert _edge(sub.enrolled_on_device, "A", "D") == LinkEdge("A", "D", 60 * D, False, 0, 60 * D)
    assert _by_id(sub.devices, "D") == IdentityNode("D", 60 * D, 1, 2, 4, 1, False, False)
    assert _subgraph("A_OPEN", graph=graph).seed == "A_OPEN"
    population = _population(graph, T, Phase.TRAINING)
    assert [account.account_id for account in population] == ["A", "A_OPEN", "A_Q"]


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {k for item in value.values() for k in _keys(item)}
    if isinstance(value, list | tuple):
        return {key for item in value for key in _keys(item)}
    return set()


def test_no_supervision_or_answer_key_reaches_a_subgraph() -> None:
    """No label is printed, and erasing every label changes no answer."""
    world = standard_world()
    answers = _answers(world.graph())
    assert not _keys([rows for _, rows in answers]) & LABELS
    tables = world.tables()
    tables["Account"] = tables["Account"].assign(
        is_mule=0,
        mule_ring_id=-1,
        mule_layer="",
        mule_flow="",
        mule_onset_ms=0,
        mule_last_ms=0,
        discovered_ms=0,
        discovery_channel="",
    )
    blank = ReferenceGraph(tables)
    blank.run(WRITE_SPLIT, split_params(world.splits()))
    assert _answers(blank) == answers


def test_every_answer_keeps_the_response_contract() -> None:
    """The parsers accept every answer (each node once, the seed among its accounts, both
    ends of each edge in the subgraph, no negative duration), and a duration whose presence
    flag is false is 0."""
    seen = 0
    for request, rows in _answers(standard_world().graph()):
        for sub in parse_subgraphs(rows, request).subgraphs:
            seen += 1
            for account in sub.accounts:
                assert account.token_present or account.token_age_ms == 0
                assert account.last_in_present or account.since_last_in_ms == 0
                assert account.last_out_present or account.since_last_out_ms == 0
            for pay in sub.pays:
                assert pay.gap_present == (pay.prior_count > 0) == (not pay.first_in_pair)
                assert pay.gap_present or pay.gap_ms == pay.pair_age_ms == 0
                assert pay.next_outflow_present or pay.next_outflow_ms == 0
            links = (*sub.owned_by, *sub.enrolled_on_device, *sub.enrolled_on_ip, *sub.at_address)
            assert all(not link.active or link.since_end_ms == 0 for link in links)
    assert seen == 12 + 13 + 14 + 15


def test_a_seeds_subgraph_does_not_depend_on_the_other_seeds() -> None:
    graph = standard_world().graph()
    seeds = ["A_QUY", "A_MIA", "A_ANN", "A_ROY", "A_BEN", "A_MAX"]
    for phase in Phase:
        batch = _fetch(graph, seeds, phase=phase)
        assert [sub.seed for sub in batch.subgraphs] == seeds
        assert list(batch.subgraphs) == [
            _fetch(graph, [seed], phase=phase).subgraphs[0] for seed in seeds
        ]


def test_a_refused_request_prints_only_its_reason() -> None:
    graph = standard_world().graph()
    params = subgraph_params(SubgraphRequest(("A_ANN",), T, Phase.TRAINING), CONFIG)

    def answer(**change: Any) -> list[dict[str, Any]]:
        return graph.run(FETCH_SUBGRAPHS, {**params, **change})

    invalid: list[dict[str, Any]] = [
        {"seed_ids": []},
        {"lookback_days": 29},
        {"lookback_days": 3651},
        {"siblings": 0},
        {"hub_threshold": 0},
        {"as_of_ms": 364 * D},
    ]
    for change in invalid:
        assert answer(**change) == [{"status": "invalid_parameters"}], change
    assert answer(seed_ids=["A_ANN", "A_ANN"]) == [{"status": "duplicate_seed"}]
    assert answer(seed_ids=["A_ANN", "A_NONE"]) == [{"status": "unknown_seed", "seeds": ["A_NONE"]}]
    assert answer(seed_ids=["A_NEW", "A_ANN", "A_HAL"]) == [
        {"status": "invisible_seed", "seeds": ["A_HAL", "A_NEW"]}
    ]
    assert answer(seed_ids=["A_NEW", "A_HAL"], held_out_visible=True) == [
        {"status": "invisible_seed", "seeds": ["A_NEW"]}
    ]


def _population(
    graph: ReferenceGraph, t: int, phase: Phase, page_size: int = 10_000
) -> list[PopulationAccount]:
    pages = account_pages(graph, LIST_POPULATION, population_params(t, phase), page_size=page_size)
    return [account for page in pages for account in parse_population(page)]


OPEN_AT_T = ["A_ANN", "A_ANN_CARD", "A_BEN", "A_MAX", "A_QUY", "A_ROY", "A_SUE", "A_VIC"]


def test_the_population_is_the_phases_customer_accounts_open_at_t() -> None:
    graph = standard_world().graph()
    training = _population(graph, T, Phase.TRAINING, page_size=4)
    assert [account.account_id for account in training] == OPEN_AT_T
    assert training[0] == PopulationAccount("A_ANN", "deposit", T - 400 * D, 0)
    test = _population(graph, T, Phase.TEST)
    assert [account.account_id for account in test] == sorted([*OPEN_AT_T, "A_HAL"])
    assert PopulationAccount("A_HAL", "deposit", T - 200 * D, 1) in test
    # A month earlier Old's account and Mia's, closed the day she was found, were still
    # open. Four weeks after t New's account and Ann's new one have opened, and Max's has
    # closed the day he was found.
    earlier = {account.account_id for account in _population(graph, T - 30 * D, Phase.TEST)}
    assert {"A_OLD", "A_MIA"} <= earlier
    later = {account.account_id for account in _population(graph, LATER, Phase.TEST)}
    assert later == {*OPEN_AT_T, "A_HAL", "A_NEW", "A_ANN_NEW"} - {"A_MAX"}
    refused = {**population_params(T, Phase.TEST), "after_id": "", "batch_size": 0}
    assert graph.run(LIST_POPULATION, refused) == [{"status": "invalid_parameters"}]


def test_supervision_holds_the_mules_found_before_the_cutoff() -> None:
    world = standard_world()
    world.party("P_HELD", T - 100 * D, held_out=True)
    found = Mule(1, "first_layer", "pass_through", T - 50 * D, T - 4 * D, T - 3 * D, "trace")
    world.account("A_HELD", "P_HELD", T - 100 * D, closed_ms=T - 3 * D, mule=found)
    graph = world.graph()

    def supervision(cutoff: int, phase: Phase = Phase.TRAINING) -> list[Supervision]:
        rows = graph.run(READ_SUPERVISION, supervision_params(cutoff, phase))
        return parse_supervision(rows, cutoff)

    assert supervision(T) == [Supervision("A_MIA", T - 5 * D, "victim_report", T - 60 * D)]
    assert [row.account_id for row in supervision(T, Phase.TEST)] == ["A_HELD", "A_MIA"]
    # Max was found 20 days after t and Roy 30 days after it: the cutoff is strict.
    assert [row.account_id for row in supervision(T + 30 * D)] == ["A_MAX", "A_MIA"]
    assert [row.account_id for row in supervision(T + 30 * D + 1)] == ["A_MAX", "A_MIA", "A_ROY"]


def test_the_answer_key_pages_every_account_with_its_truth() -> None:
    graph = standard_world().graph()
    pages = account_pages(graph, READ_ANSWER_KEY, {}, page_size=5)
    rows = [row for page in pages for row in parse_answer_key(page)]
    assert [row.account_id for row in rows] == sorted(graph.accounts)
    truth = {row.account_id: row for row in rows}
    assert truth["A_MIA"] == AnswerKeyRow(
        "A_MIA", 1, 0, "first_layer", "pass_through", T - 60 * D, T - 10 * D + 2 * H, T - 5 * D
    )
    assert (truth["A_SUE"].is_mule, truth["A_SUE"].discovered_ms) == (1, 0)
    assert [truth[account].is_mule for account in ("X_CASH", "A_ANN", "G_FEES")] == [-1, 0, 0]


def test_write_split_writes_each_party_and_the_accounts_it_owns() -> None:
    graph = ReferenceGraph(standard_world().tables())
    assert set(graph.account_split.values()) == set(graph.party_split.values()) == {-1}
    assert parse_split(graph.run(WRITE_SPLIT, split_params({"P_ANN": 0, "P_HAL": 1}))) == (
        SplitWritten(2, 4)
    )
    # An account takes its owner's split whenever the ownership starts.
    splits = [graph.account_split[a] for a in ("A_ANN_CARD", "A_ANN_NEW", "A_HAL", "X_CASH")]
    assert splits == [0, 0, 1, -1]
    assert (graph.party_split["P_HAL"], graph.party_split["P_BEN"]) == (1, -1)
    refused: list[tuple[dict[str, Any], str]] = [
        ({"party_ids": [], "splits": []}, "invalid_parameters"),
        ({"party_ids": ["P_ANN"], "splits": [0, 1]}, "invalid_parameters"),
        ({"party_ids": ["P_ANN"], "splits": [2]}, "invalid_parameters"),
        ({"party_ids": ["P_ANN", "P_ANN"], "splits": [0, 0]}, "invalid_parameters"),
        ({"party_ids": ["P_NONE"], "splits": [0]}, "unknown_party"),
    ]
    for params, status in refused:
        assert graph.run(WRITE_SPLIT, params) == [{"status": status}]
    # A joint account takes its owners' largest split.
    w = WorldBuilder()
    w.party("P_1", T)
    w.party("P_2", T, held_out=True)
    w.account("J", "P_1", T)
    w.link("Party_Owns_Account", "P_2", "J", T)
    assert w.graph().account_split["J"] == 1


def test_a_world_read_from_csv_answers_as_its_frames(tmp_path: Path) -> None:
    world = standard_world()
    world.write_csv(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(f"{t}.csv" for t in TABLES)
    lines = (tmp_path / "Account.csv").read_text().splitlines()
    assert lines[0] == ",".join(TABLES["Account"])
    assert any(line.startswith("X_CASH,unknown,True,") for line in lines)
    graph = ReferenceGraph.from_csv(tmp_path)
    graph.run(WRITE_SPLIT, split_params(world.splits()))
    assert _answers(graph) == _answers(world.graph())


def _with_row(frame: pd.DataFrame, row: tuple[object, ...]) -> pd.DataFrame:
    return pd.concat([frame, pd.DataFrame([row], columns=frame.columns)], ignore_index=True)


# Changes to the standard world's tables, and what the refusal says.
BROKEN: list[tuple[Callable[[dict[str, pd.DataFrame]], object], str]] = [
    (lambda tables: tables.pop("IP"), "needs the tables"),
    (lambda tables: tables.update(Party=tables["Party"][["joined_ms", "id"]]), "columns"),
    (lambda tables: tables.update(Device=pd.concat([tables["Device"]] * 2)), "repeats an id"),
    (
        lambda tables: tables.update(
            Party_Uses_Device=_with_row(tables["Party_Uses_Device"], ("P_ANN", "D_NONE", T, 0))
        ),
        "unknown vertex",
    ),
    (
        lambda tables: tables.update(
            Party_Uses_IP=_with_row(tables["Party_Uses_IP"], ("P_ANN", "I_HOME", T - 400 * D, 0))
        ),
        "repeats a link",
    ),
    (
        lambda tables: tables.update(
            Transaction_To_Account=tables["Transaction_To_Account"].iloc[1:]
        ),
        "one sender and one receiver",
    ),
    (
        lambda tables: tables.update(
            Transaction_Used_Device=tables["Transaction_Used_Device"].assign(event_ts_ms=T)
        ),
        "another time",
    ),
    (
        lambda tables: tables.update(Account=tables["Account"].assign(is_external="maybe")),
        "boolean",
    ),
]


@pytest.mark.parametrize(("change", "message"), BROKEN)
def test_tables_that_describe_no_graph_are_refused(
    change: Callable[[dict[str, pd.DataFrame]], object], message: str
) -> None:
    tables = standard_world().tables()
    change(tables)
    with pytest.raises(ValueError, match=message):
        ReferenceGraph(tables)


def test_a_query_takes_exactly_its_installed_parameters() -> None:
    graph = standard_world().graph()
    with pytest.raises(ValueError, match="parameters"):
        graph.run(LIST_POPULATION, {"as_of_ms": T, "held_out_visible": False, "after_id": ""})
    with pytest.raises(ValueError, match="parameters"):
        graph.run(READ_SUPERVISION, {**supervision_params(T, Phase.TEST), "is_mule": 1})
    with pytest.raises(ValueError, match="No query"):
        graph.run("drop_all", {})
