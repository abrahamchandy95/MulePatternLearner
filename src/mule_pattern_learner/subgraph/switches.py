"""The ablation switches on built batches (docs/features.md, "Ablation and comparison
switches"). Each makes a new batch from a built one, needs no fetch, and recounts the degree
columns after it.

- Scored account alone: keeps each scored account's row and its own payment edges, pays and
  rev_pays, and drops every other edge. On its outflows it zeroes the fields measured at the
  other end, dwell_tracked, left_24h, dwell and receiver_gap, and on its inflows sender_gap;
  amount, rail, age and the pair fields stay. Every other node is zeroed, its degree columns
  included, and the community block is off.
- Without devices and IPs: drops every device and IP node and the six enrolment,
  owner-enrolment and seen relations with their reverses, and with them every account and
  customer node other than the scored account left with no edge: the co-users' accounts and
  customers reached only through devices and IPs. Every customer row keeps accounts_owned
  alone, and the community block is off.
- Without community: the community block is off; the model's head reads the scored account's
  64 numbers alone.
- Without per-payment timing: zeroes every payment-edge duration with its presence bit (age,
  pair_gap, pair_age, sender_gap, receiver_gap, dwell), dwell_tracked and left_24h; amount,
  rail and pair_prior_count stay.

A switch keeps every node type and every relation, empty where it drops them, so one layout
serves every batch. With the community block off the batch's block is all zeros, and the
model built for the switch does not read it (Switch.community).
"""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from . import degrees
from .build import Batch, put_relation
from .layouts import (
    ACCOUNT,
    CUSTOMER,
    CUSTOMER_LAYOUT,
    DEVICE,
    FORWARD,
    IDENTITY_RELATIONS,
    IP,
    NODE_TYPES,
    OWNED_BY,
    PAYMENT_LAYOUT,
    PAYS,
)


class Switch(StrEnum):
    """An ablation of the full model (docs/design.md, "Comparisons")."""

    SCORED_ALONE = "scored_alone"
    WITHOUT_DEVICES_AND_IPS = "without_devices_and_ips"
    WITHOUT_COMMUNITY = "without_community"
    WITHOUT_PAYMENT_TIMING = "without_payment_timing"

    @property
    def community(self) -> bool:
        """Whether the model of this ablation reads the community block."""
        return self is Switch.WITHOUT_PAYMENT_TIMING


# The payment-edge fields measured at the receiver, zeroed on a scored account's outflows when
# it is alone, and the one measured at the sender, zeroed on its inflows.
AT_RECEIVER = PAYMENT_LAYOUT.positions("receiver_gap", "dwell_tracked", "left_24h", "dwell")
AT_SENDER = PAYMENT_LAYOUT.positions("sender_gap")
# The payment-edge fields of per-payment timing.
TIMING = PAYMENT_LAYOUT.positions(
    "age",
    "pair_gap",
    "pair_age",
    "sender_gap",
    "receiver_gap",
    "dwell_tracked",
    "left_24h",
    "dwell",
)
# The customer columns 2 to 8, all but accounts_owned.
CUSTOMER_SHARING = CUSTOMER_LAYOUT.positions(
    "devices_30d",
    "ips_30d",
    "new_devices_30d",
    "new_ips_30d",
    "co_users",
    "new_co_users_30d",
    "co_users_both",
)


def _copy(data: HeteroData) -> HeteroData:
    """A copy of a batch's data in which each reverse relation shares its forward relation's
    attributes, one tensor, as in a built batch: HeteroData.clone copies them apart, which
    would have encode_batch encode every payment's 401 numbers twice."""
    found = data.clone()
    for relation in FORWARD:
        put_relation(found, relation, found[relation].edge_index, found[relation].get("edge_attr"))
    return found


def apply(batch: Batch, switch: Switch) -> Batch:
    """The batch under a switch, its degree columns recounted."""
    data = _copy(batch.data)
    if switch is Switch.SCORED_ALONE:
        _scored_alone(data)
    elif switch is Switch.WITHOUT_DEVICES_AND_IPS:
        _without_devices_and_ips(data)
    elif switch is Switch.WITHOUT_PAYMENT_TIMING:
        data[PAYS].edge_attr[:, TIMING] = 0
    if not switch.community:
        data.community = torch.zeros_like(data.community)
    degrees.count(data)
    if switch is Switch.SCORED_ALONE:
        _zero_all_but_scored(data)
    return Batch(batch.keys, data)


def _empty(data: HeteroData, relation: tuple[str, str, str]) -> None:
    """Drop every edge of a forward relation and of its reverse."""
    edges: Tensor = data[relation].edge_index
    attrs: Tensor | None = data[relation].get("edge_attr")
    put_relation(data, relation, edges[:, :0], None if attrs is None else attrs[:0])


def _scored_alone(data: HeteroData) -> None:
    scored: Tensor = data.scored_index
    source, target = data[PAYS].edge_index
    sent, received = torch.isin(source, scored), torch.isin(target, scored)
    kept = sent | received
    attrs: Tensor = data[PAYS].edge_attr[kept].clone()
    outflows, inflows = sent[kept], received[kept]
    for column in AT_RECEIVER:
        attrs[outflows, column] = 0
    for column in AT_SENDER:
        attrs[inflows, column] = 0
    put_relation(data, PAYS, data[PAYS].edge_index[:, kept], attrs)
    for relation in FORWARD[1:]:
        _empty(data, relation)


def _zero_all_but_scored(data: HeteroData) -> None:
    """Zero every node's row, degree columns included, but each scored account's."""
    accounts: Tensor = data[ACCOUNT].x
    others = torch.ones(accounts.shape[0], dtype=torch.bool, device=accounts.device)
    others[data.scored_index] = False
    accounts[others] = 0
    for kind in NODE_TYPES:
        if kind != ACCOUNT:
            data[kind].x.zero_()


def _without_devices_and_ips(data: HeteroData) -> None:
    for kind in (DEVICE, IP):
        data[kind].x = data[kind].x[:0]
    for relation in IDENTITY_RELATIONS:
        _empty(data, relation)
    data[CUSTOMER].x[:, CUSTOMER_SHARING] = 0
    # Every node but the scored accounts left with no edge drops out: payments and ownership
    # are the relations left.
    remaining = (PAYS, OWNED_BY)
    joined = {
        kind: torch.zeros(data[kind].num_nodes, dtype=torch.bool, device=data[kind].x.device)
        for kind in (ACCOUNT, CUSTOMER)
    }
    joined[ACCOUNT][data.scored_index] = True
    for start, name, end in remaining:
        edges: Tensor = data[start, name, end].edge_index
        joined[start][edges[0]] = True
        joined[end][edges[1]] = True
    moved = {kind: torch.cumsum(kept.long(), 0) - 1 for kind, kept in joined.items()}
    for kind, kept in joined.items():
        data[kind].x = data[kind].x[kept]
    for relation in remaining:
        start, _, end = relation
        edges = data[relation].edge_index
        renamed = torch.stack([moved[start][edges[0]], moved[end][edges[1]]])
        put_relation(data, relation, renamed, data[relation].get("edge_attr"))
    data.scored_index = moved[ACCOUNT][data.scored_index]
