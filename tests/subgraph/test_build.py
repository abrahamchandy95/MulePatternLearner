"""Subgraphs of the reference graph's standard world (testing.world) as one HeteroData.

At T (Monday 2024-07-01) in training, Mia's subgraph holds Vic, who paid her (T08, 10 days
before T), Max, whom she paid two hours later (T09) and who shares the ring's device
D_RING0 with her, Max's cash-out X_CASH, and Quy, who holds D_RING0 and has no payments.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.model.time_encoding import WIDTH, encode_reference
from mule_pattern_learner.subgraph.build import (
    ACCOUNT_TYPES,
    LAYOUTS,
    NODE_TYPES,
    RAILS,
    Layout,
    build,
)
from mule_pattern_learner.testing.reference_graph import DAY_MS
from mule_pattern_learner.testing.world import HOUR_MS, LATER, T, standard_world
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    NODES,
    RELATIONS,
    AccountNode,
    LinkEdge,
    PaymentEdge,
    Phase,
    Subgraph,
    SubgraphBatch,
    SubgraphRequest,
    parse_subgraphs,
    subgraph_params,
)

D, H = DAY_MS, HOUR_MS
GRAPH = standard_world().graph()
ACCOUNT, PAYMENT, LINK = LAYOUTS[AccountNode], LAYOUTS[PaymentEdge], LAYOUTS[LinkEdge]
ACCELERATORS = [
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


def _batch(seeds: list[str], t: int = T) -> SubgraphBatch:
    request = SubgraphRequest(tuple(seeds), t, Phase.TRAINING)
    params = subgraph_params(request, DEFAULT_CONFIG.subgraph)
    return parse_subgraphs(GRAPH.run(FETCH_SUBGRAPHS, params), request)


def _edge_type(name: str, *, reverse: bool = False) -> tuple[str, str, str]:
    _, start, end = RELATIONS[name]
    if reverse:
        return NODE_TYPES[end], f"rev_{name}", NODE_TYPES[start]
    return NODE_TYPES[start], name, NODE_TYPES[end]


def _position(items: tuple[AccountNode | PaymentEdge, ...], item_id: str) -> int:
    return [item.id for item in items].index(item_id)


def _link(edges: tuple[LinkEdge, ...], from_id: str, to_id: str) -> int:
    return [(edge.from_id, edge.to_id) for edge in edges].index((from_id, to_id))


def _pairs(data: HeteroData, name: str) -> set[tuple[int, int]]:
    edge_index: Tensor = data[_edge_type(name)].edge_index
    return {(int(a), int(b)) for a, b in edge_index.T}


def _bit(row: Tensor, layout: Layout, duration: str) -> float:
    return float(row[layout.columns.index(f"{duration}:present")])


def _encoding(row: Tensor, layout: Layout, duration: str) -> Tensor:
    start = layout.columns.index(f"{duration}:fourier0")
    return row[start : start + WIDTH]


def _assert_encodes(encoding: Tensor, duration_ms: int) -> None:
    expected = encode_reference(np.array(duration_ms))
    assert np.abs(encoding.numpy() - expected).max() < 1e-5


def test_every_node_type_and_relation_and_its_reverse_is_there_with_its_columns() -> None:
    batch = _batch(["A_MIA", "A_ANN", "A_QUY"])
    data = build([batch])
    assert all(getattr(batch.subgraphs[1], name) for name in (*NODES, *RELATIONS))
    assert sorted(data.node_types) == sorted(NODE_TYPES.values())
    assert len(data.edge_types) == 2 * len(RELATIONS)
    for name, kind in NODES.items():
        count = sum(len(getattr(subgraph, name)) for subgraph in batch.subgraphs)
        assert data[NODE_TYPES[name]].x.shape == (count, len(LAYOUTS[kind].columns))
    for name, (kind, _, _) in RELATIONS.items():
        count = sum(len(getattr(subgraph, name)) for subgraph in batch.subgraphs)
        forward, reverse = data[_edge_type(name)], data[_edge_type(name, reverse=True)]
        assert forward.edge_index.shape == (2, count)
        assert forward.edge_attr.shape == (count, len(LAYOUTS[kind].columns))
        assert torch.equal(reverse.edge_index, forward.edge_index.flip(0))
        assert reverse.edge_attr is forward.edge_attr
    # Mia reaches no address: the node type and relation are there, without rows.
    mia = build([_batch(["A_MIA"])])
    assert mia["account", "at_address", "address"].edge_index.shape == (2, 0)
    assert mia["account", "at_address", "address"].edge_attr.shape == (0, len(LINK.columns))
    assert mia["address"].x.shape == (0, len(LAYOUTS[NODES["addresses"]].columns))


def test_the_seed_index_and_the_as_of_times_follow_the_batches_and_their_seeds() -> None:
    at_t, later = _batch(["A_MIA", "A_ANN"]), _batch(["A_NEW"], LATER)
    data = build([at_t, later])
    subgraphs = [*at_t.subgraphs, *later.subgraphs]
    offsets = np.cumsum([0] + [len(subgraph.accounts) for subgraph in subgraphs[:-1]])
    expected = [
        int(offset) + _position(subgraph.accounts, subgraph.seed)
        for offset, subgraph in zip(offsets, subgraphs, strict=True)
    ]
    assert data.seed_index.tolist() == expected
    assert data.as_of_ms.tolist() == [T, T, LATER]
    # The seed's row is its own: New's account opened 14 days before LATER.
    _assert_encodes(_encoding(data["account"].x[expected[2]], ACCOUNT, "age_ms"), 14 * D)


def test_a_node_reached_by_two_paths_is_one_node_so_the_cycle_stays() -> None:
    """Mia paid Max on D_RING0, which Max holds: Mia, payment, Max, device, Mia."""
    batch = _batch(["A_MIA"])
    [sub] = batch.subgraphs
    data = build([batch])
    assert data["account"].x.shape[0] == len({account.id for account in sub.accounts}) == 5
    mia, maximo = _position(sub.accounts, "A_MIA"), _position(sub.accounts, "A_MAX")
    ring = [device.id for device in sub.devices].index("D_RING0")
    assert data.seed_index.tolist() == [mia]
    assert (mia, maximo) in _pairs(data, "pays")
    assert (maximo, ring) in _pairs(data, "enrolled_on_device")
    assert (mia, ring) in _pairs(data, "sent_from_device")
    assert (maximo, ring) in _pairs(data, "sent_from_device")


def test_seeds_that_reach_one_account_each_hold_a_copy_of_it() -> None:
    """Mia's and Max's subgraphs both hold both of them; no edge joins the two."""
    batch = _batch(["A_MIA", "A_MAX"])
    data = build([batch])
    first, second = batch.subgraphs
    assert data["account"].x.shape[0] == len(first.accounts) + len(second.accounts)
    for name, (_, start, end) in RELATIONS.items():
        cut = {key: len(getattr(first, key)) for key in (start, end)}
        for source, target in _pairs(data, name):
            assert (source < cut[start]) == (target < cut[end]), name


def test_each_duration_is_its_fourier_encoding_beside_a_presence_bit() -> None:
    batch = _batch(["A_MIA"])
    [sub] = batch.subgraphs
    data = build([batch])
    # Vic paid Mia 10 days before T, the first payment of the pair, and Mia passed it on two
    # hours later.
    t08 = data["account", "pays", "account"].edge_attr[_position(sub.pays, "T08")]
    durations = ("age_ms", "gap_ms", "pair_age_ms", "next_outflow_ms")
    assert [_bit(t08, PAYMENT, duration) for duration in durations] == [1, 0, 0, 1]
    _assert_encodes(_encoding(t08, PAYMENT, "age_ms"), 10 * D)
    _assert_encodes(_encoding(t08, PAYMENT, "next_outflow_ms"), 2 * H)
    assert not _encoding(t08, PAYMENT, "gap_ms").any()
    assert not _encoding(t08, PAYMENT, "pair_age_ms").any()
    # Mia's ownership ended when she was found, 5 days before T; Max's is open.
    owned_by = data["account", "owned_by", "party"].edge_attr
    mia = owned_by[_link(sub.owned_by, "A_MIA", "P_MIA")]
    assert _bit(mia, LINK, "since_end_ms") == 1
    _assert_encodes(_encoding(mia, LINK, "since_end_ms"), 5 * D)
    _assert_encodes(_encoding(mia, LINK, "held_ms"), 295 * D)
    maximo = owned_by[_link(sub.owned_by, "A_MAX", "P_MAX")]
    assert _bit(maximo, LINK, "since_end_ms") == 0
    assert not _encoding(maximo, LINK, "since_end_ms").any()
    # Quy's account is 90 days old, with no token and no payment.
    quy = data["account"].x[_position(sub.accounts, "A_QUY")]
    assert [_bit(quy, ACCOUNT, duration) for duration, _ in ACCOUNT.durations] == [1, 0, 0, 0]
    _assert_encodes(_encoding(quy, ACCOUNT, "age_ms"), 90 * D)
    assert not _encoding(quy, ACCOUNT, "since_last_in_ms").any()


def test_the_plain_numbers_are_one_hot_categories_flags_and_log_counts() -> None:
    batch = _batch(["A_MIA"])
    [sub] = batch.subgraphs
    data = build([batch])
    t08 = data["account", "pays", "account"].edge_attr[_position(sub.pays, "T08")]
    assert [float(t08[PAYMENT.columns.index(f"rail={rail}")]) for rail in RAILS] == [0] * 6 + [1]
    assert float(t08[PAYMENT.columns.index("first_in_pair")]) == 1
    assert float(t08[PAYMENT.columns.index("amount")]) == pytest.approx(np.log1p(900.0))
    assert float(t08[PAYMENT.columns.index("prior_count")]) == 0
    cash = data["account"].x[_position(sub.accounts, "X_CASH")]
    assert dict(zip(ACCOUNT_TYPES, cash.tolist(), strict=False))["unknown"] == 1
    assert cash[: len(ACCOUNT_TYPES)].sum() == 1
    assert float(cash[ACCOUNT.columns.index("is_external")]) == 1
    assert float(cash[ACCOUNT.columns.index("in_30d")]) == pytest.approx(np.log1p(1))


def test_every_field_of_a_row_is_an_end_a_feature_or_a_presence_flag() -> None:
    for kind, layout in LAYOUTS.items():
        used = {"id", "from_id", "to_id"} & {field.name for field in fields(kind)}
        used |= {name for name, _ in layout.categories} | {*layout.flags, *layout.counts}
        for duration, flag in layout.durations:
            used |= {duration, flag.removeprefix("not ")} - {""}
        assert used == {field.name for field in fields(kind)}, kind.__name__


def _first(batch: SubgraphBatch, **changes: object) -> SubgraphBatch:
    """batch with its first subgraph changed."""
    first: Subgraph = replace(batch.subgraphs[0], **changes)
    return replace(batch, subgraphs=(first, *batch.subgraphs[1:]))


def test_build_refuses_what_it_cannot_lay_out() -> None:
    batch = _batch(["A_MIA"])
    [sub] = batch.subgraphs
    with pytest.raises(ValueError, match="at least one seed"):
        build([replace(batch, subgraphs=())])
    with pytest.raises(ValueError, match="repeats a node of accounts"):
        build([_first(batch, accounts=(*sub.accounts, sub.accounts[0]))])
    savings = replace(sub.accounts[0], account_type="savings")
    with pytest.raises(ValueError, match="account_type"):
        build([_first(batch, accounts=(savings, *sub.accounts[1:]))])
    refund = replace(sub.pays[0], amount=-5.0)
    with pytest.raises(ValueError, match="amount is negative"):
        build([_first(batch, pays=(refund, *sub.pays[1:]))])


@pytest.mark.parametrize("device", ACCELERATORS)
def test_a_build_on_an_accelerator_matches_the_cpu(device: str) -> None:
    batch = _batch(["A_MIA", "A_ANN", "A_QUY"])
    on_cpu, on_device = build([batch]), build([batch], device=device)
    for store in (*on_cpu.node_types, *on_cpu.edge_types):
        for key, value in on_cpu[store].items():
            moved: Tensor = on_device[store][key]
            assert moved.device.type == device
            assert torch.allclose(moved.cpu(), value, atol=1e-5), (store, key)
    assert torch.equal(on_device.seed_index.cpu(), on_cpu.seed_index)
