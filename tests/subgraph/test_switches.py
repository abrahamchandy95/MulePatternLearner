"""The ablation switches on the hand-built batch of conftest.py: each zeroes exactly its listed
columns and drops exactly its listed edges and nodes, the degree columns are counted again
after it, and the batch it was given is left as it was (docs/features.md, "Ablation and
comparison switches")."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from mule_pattern_learner.model.encode import encode_batch
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.layouts import (
    ACCOUNT,
    ACCOUNT_LAYOUT,
    CUSTOMER_LAYOUT,
    FORWARD,
    IDENTITY_RELATIONS,
    LAYOUT,
    NODE_LAYOUTS,
    NODE_TYPES,
    OWNED_BY,
    PAYMENT_LAYOUT,
    PAYS,
    RELATIONS,
    degree_name,
    reverse,
)
from mule_pattern_learner.subgraph.layouts import (
    DEGREES as DEGREE_RELATIONS,
)
from mule_pattern_learner.subgraph.switches import Switch, apply
from mule_pattern_learner.testing.world import HOUR_MS

# The first degree column of an account row, after columns 1 to 46.
DEGREES = ACCOUNT_LAYOUT.index("degree_pays")
# The hand-built batch's scored accounts, a_s1 and a_s2, and its payment edges p1 to p6.
A_S1, A_S2 = 3, 6
P1, P2, P3, P4, P5, P6 = range(6)


def _pairs(edge_index: Tensor) -> list[tuple[int, int]]:
    return [(int(source), int(target)) for source, target in edge_index.T.tolist()]


def _zeroed(attrs: Tensor, rows: list[int], *names: str) -> Tensor:
    """attrs with the payment-edge columns named zeroed on rows."""
    found = attrs.clone()
    for row in rows:
        found[row, PAYMENT_LAYOUT.positions(*names)] = 0
    return found


def _reverses_match(batch: Batch) -> None:
    """Every reverse relation is its forward relation turned around and shares its attributes,
    one tensor, as in a built batch."""
    data = batch.data
    assert set(data.edge_types) == set(RELATIONS)
    for relation in FORWARD:
        backward = data[reverse(relation)]
        assert torch.equal(backward.edge_index, data[relation].edge_index.flip(0))
        if relation != OWNED_BY:
            assert backward.edge_attr is data[relation].edge_attr


def test_the_scored_account_alone(hand_batch: Batch) -> None:
    switched = apply(hand_batch, Switch.SCORED_ALONE)
    data, built = switched.data, hand_batch.data
    # Its own payments stay, p3 between a_two and x_payer goes.
    assert _pairs(data[PAYS].edge_index) == [(5, 3), (3, 1), (3, 2), (8, 6), (6, 7)]
    attrs = built[PAYS].edge_attr[[P1, P2, P4, P5, P6]]
    # On its outflows (p2, p4, p6) the fields measured at the receiver are zeroed, on its
    # inflows (p1, p5) the sender's gap.
    expected = _zeroed(attrs, [1, 2, 4], "receiver_gap", "dwell_tracked", "left_24h", "dwell")
    expected = _zeroed(expected, [0, 3], "sender_gap")
    assert torch.equal(data[PAYS].edge_attr, expected)
    # p1 kept its dwell, p4 lost its receiver gap, p6 its dwell and its receiver gap.
    assert data[PAYS].edge_attr[0, PAYMENT_LAYOUT.index("dwell_tracked")] == 1
    assert built[PAYS].edge_attr[P4, PAYMENT_LAYOUT.index("receiver_gap_ms")] > 0
    assert built[PAYS].edge_attr[P6, PAYMENT_LAYOUT.index("left_24h")] == 0.5
    dwell = PAYMENT_LAYOUT.positions("dwell")
    assert built[PAYS].edge_attr[P6, dwell].tolist() == [1, 2 * HOUR_MS]
    assert data[PAYS].edge_attr[4, dwell].tolist() == [0, 0]
    for relation in FORWARD[1:]:
        assert data[relation].edge_index.shape == (2, 0)
    # Each scored account keeps its row, with its degrees counted again: a_s1 received p1 and
    # sent p2 and p4, a_s2 received p5 and sent p6. Every other node is all zeros.
    x = data["account"].x
    assert torch.equal(x[A_S1, :DEGREES], built["account"].x[A_S1, :DEGREES])
    assert x[A_S1, DEGREES:].tolist() == [1, 2, 0, 0, 0, 0, 0]
    assert x[A_S2, DEGREES:].tolist() == [1, 1, 0, 0, 0, 0, 0]
    others = [row for row in range(x.shape[0]) if row not in (A_S1, A_S2)]
    assert not x[others].any()
    for kind in ("customer", "device", "ip"):
        assert data[kind].x.shape == built[kind].x.shape and not data[kind].x.any()
    assert not data.community.any()
    assert torch.equal(data.scored_index, built.scored_index)
    assert torch.equal(data.label, built.label) and torch.equal(data.weight, built.weight)
    _reverses_match(switched)


def test_the_scored_account_alone_zeroes_the_degrees_of_every_other_node(
    hand_batch: Batch,
) -> None:
    """docs/features.md: the switch zeroes every other node's features, degree columns
    included, though degrees are counted after the switch, from the edges it keeps. The built
    batch has degrees on those nodes, and the switch leaves none. The difference shows on the
    accounts s pays or is paid by: the payments it keeps would count a degree on them again
    (a_hub, node 1, receives p2), so only zeroing the whole row leaves them 0. No other edge is
    kept, so a customer, a device or an IP would count 0 whatever is zeroed. Each scored
    account keeps its own degrees, counted after the switch."""
    built, data = hand_batch.data, apply(hand_batch, Switch.SCORED_ALONE).data
    scored = set(built.scored_index.tolist())
    for kind in NODE_TYPES:
        degree_columns = [NODE_LAYOUTS[kind].index(degree_name(r)) for r in DEGREE_RELATIONS[kind]]
        others = [
            row for row in range(built[kind].x.shape[0]) if kind != ACCOUNT or row not in scored
        ]
        assert built[kind].x[others][:, degree_columns].any(), f"no {kind} has a degree to lose"
        assert not data[kind].x[others][:, degree_columns].any(), kind
    hub, pays = 1, ACCOUNT_LAYOUT.index("degree_pays")
    kept = data[PAYS].edge_index
    assert (
        1 in kept[1].tolist()
        and built[ACCOUNT].x[hub, pays] > 0
        and data[ACCOUNT].x[hub, pays] == 0
    )
    x = data[ACCOUNT].x
    assert x[A_S1, DEGREES:].any() and x[A_S2, DEGREES:].any()


def test_without_devices_and_ips(hand_batch: Batch) -> None:
    switched = apply(hand_batch, Switch.WITHOUT_DEVICES_AND_IPS)
    data, built = switched.data, hand_batch.data
    assert data["device"].x.shape == (0, 12) and data["ip"].x.shape == (0, 12)
    for relation in IDENTITY_RELATIONS:
        assert data[relation].edge_index.shape == (2, 0)
        assert data[relation].edge_attr.shape[0] == 0
    # a_co, reached only through d_home, and its customer p_co drop out; a_two stays, since
    # it paid x_payer. The accounts are now a_hub, a_own, a_s1, a_two, x_payer, then a_s2,
    # a_three, x_payer; the customers p_c1 and p_c2.
    kept = [1, 2, 3, 4, 5, 6, 7, 8]
    x = data["account"].x
    assert torch.equal(x[:, :DEGREES], built["account"].x[kept, :DEGREES])
    assert data.scored_index.tolist() == [2, 5]
    assert _pairs(data[PAYS].edge_index) == [(4, 2), (2, 0), (3, 4), (2, 1), (7, 5), (5, 6)]
    assert torch.equal(data[PAYS].edge_attr, built[PAYS].edge_attr)
    assert _pairs(data[OWNED_BY].edge_index) == [(1, 0), (2, 0), (5, 1)]
    # Degrees counted again: a_s1 and a_two lose their device and IP edges.
    assert x[2, DEGREES:].tolist() == [1, 2, 1, 0, 0, 0, 0]
    assert x[3, DEGREES:].tolist() == [0, 1, 0, 0, 0, 0, 0]
    # Each customer keeps accounts_owned alone, and its ownership degree.
    customers = data["customer"].x
    assert customers.tolist() == [
        [2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]
    assert CUSTOMER_LAYOUT.raw_names[0] == "accounts_owned"
    assert not data.community.any()
    _reverses_match(switched)


def test_without_community(hand_batch: Batch) -> None:
    switched = apply(hand_batch, Switch.WITHOUT_COMMUNITY)
    data, built = switched.data, hand_batch.data
    assert not data.community.any() and built.community.any()
    for kind in built.node_types:
        assert torch.equal(data[kind].x, built[kind].x)
    for relation in built.edge_types:
        assert torch.equal(data[relation].edge_index, built[relation].edge_index)
        if "edge_attr" in built[relation]:
            assert torch.equal(data[relation].edge_attr, built[relation].edge_attr)


def test_without_per_payment_timing(hand_batch: Batch) -> None:
    switched = apply(hand_batch, Switch.WITHOUT_PAYMENT_TIMING)
    data, built = switched.data, hand_batch.data
    attrs = data[PAYS].edge_attr
    kept = PAYMENT_LAYOUT.positions("amount", "pair_prior_count") + [
        PAYMENT_LAYOUT.index(name) for name in PAYMENT_LAYOUT.raw_names if name.startswith("rail=")
    ]
    zeroed = [column for column in range(PAYMENT_LAYOUT.raw_width) if column not in kept]
    assert len(zeroed) == 23 - 9
    assert torch.equal(attrs[:, kept], built[PAYS].edge_attr[:, kept])
    assert not attrs[:, zeroed].any() and built[PAYS].edge_attr[:, zeroed].any()
    for kind in built.node_types:
        assert torch.equal(data[kind].x, built[kind].x)
    assert torch.equal(data.community, built.community)
    for relation in FORWARD[1:]:
        assert torch.equal(data[relation].edge_index, built[relation].edge_index)
    _reverses_match(switched)


@pytest.mark.parametrize("switch", list(Switch))
def test_a_switch_keeps_each_reverse_sharing_its_forward_attributes(
    hand_batch: Batch, switch: Switch
) -> None:
    """One tensor each, so that encode_batch encodes the 401-wide payment attributes once."""
    switched = apply(hand_batch, switch)
    _reverses_match(switched)
    encoded = encode_batch(switched.data, LAYOUT)
    for relation in FORWARD:
        if relation != OWNED_BY:
            assert encoded[reverse(relation)].edge_attr is encoded[relation].edge_attr


@pytest.mark.parametrize("switch", list(Switch))
def test_a_switch_leaves_the_batch_it_was_given(hand_batch: Batch, switch: Switch) -> None:
    built = hand_batch.data.clone()
    switched = apply(hand_batch, switch)
    assert switched.keys == hand_batch.keys
    for kind in built.node_types:
        assert torch.equal(hand_batch.data[kind].x, built[kind].x)
    for relation in built.edge_types:
        for key in ("edge_index", "edge_attr"):
            if key in built[relation]:
                assert torch.equal(hand_batch.data[relation][key], built[relation][key])
    assert torch.equal(hand_batch.data.community, built.community)


def test_only_the_timing_ablation_reads_the_community_block() -> None:
    assert [switch for switch in Switch if switch.community] == [Switch.WITHOUT_PAYMENT_TIMING]
