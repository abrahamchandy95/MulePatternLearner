"""Cached rows as one PyG HeteroData per batch, raw before encoding (docs/design.md, "The
model", and docs/features.md, "What one example's subgraph holds" and "The other tensors of a
batch").

An example is an (account, Monday t) pair with its own copy of its two-hop subgraph, so no node
is shared between examples, and a node reached by two paths appears once in its example, which
keeps cycles visible. The examples' nodes are laid out one example after another, each
example's nodes of a type sorted by id.

- Account nodes are B4's accounts and B5's co-users' accounts. Each has the row of
  subgraph.layouts.ACCOUNT_LAYOUT: is_external, read_hubs' hub flag and measure, and B1 to B3.
  An account hub that is not the scored account has a flag-only row, is_external, hub and
  hub_customers_90d with every other column 0; the scored account keeps its full row even
  when it is a hub.
- Customer nodes are B5's customers; device and IP nodes are B5's. A leaf's row is joined from
  read_hubs' cached leaves by id: its measure customers_90d, observed_only, crowd 1 and hub,
  every other column 0. A node whose inline leaf flag in B5 disagrees with read_hubs is
  refused (read_community prints no per-node flag, so its own inline flags are left to
  parity).
- Payment edges are B4's, from sender to receiver. B1's tracked inflow of the receiver joins
  its edge by payment id as dwell_tracked, left_24h and dwell; at a hub receiver, the scored
  hub's included, the edge carries none, since an edge's row depends on the payment, the date
  and the hub set alone.
- Ownership edges are B4's, without attributes. Enrolment, owner-enrolment and seen edges are
  B5's, each split by its node's type into the device and the IP relation.
- Every forward relation has its reverse, rev_ and its name, with the same attributes.
- scored_index is the position of each example's scored account among the account nodes;
  community each example's block; in training, label (1 positive, 0 unlabelled) and weight per
  example. Labels and weights never join a node row.
- The degree columns are counted last (subgraph.degrees).

The builder reads only the parsed rows of the queries that build examples. None holds a date
of record (opened_ms, joined_ms or closed_ms) or a label, and the builder reads none.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from ..model.encode import Column, EdgeType, Layout
from ..tigergraph.queries.read_activity_change import ActivityChange
from ..tigergraph.queries.read_community import Community
from ..tigergraph.queries.read_counterparties import Counterparties
from ..tigergraph.queries.read_hubs import Hubs, Leaf
from ..tigergraph.queries.read_money_flow import MoneyFlow, TrackedInflow
from ..tigergraph.queries.read_payment_events import RAILS, PaymentEvent, PaymentGraph
from ..tigergraph.queries.read_shared_identity import IdentityGraph, NodeRow
from . import degrees
from .cache import (
    ACTIVITY,
    COMMUNITY,
    COUNTERPARTIES,
    MONEY_FLOW,
    PAYMENT_EVENTS,
    SHARED_IDENTITY,
    Cache,
)
from .layouts import (
    ACCOUNT,
    ACCOUNT_LAYOUT,
    B1_COLUMNS,
    B2_COLUMNS,
    B3_COLUMNS,
    COMMUNITY_LAYOUT,
    CUSTOMER,
    CUSTOMER_COLUMNS,
    DEVICE,
    ENROLLED_ON_DEVICE,
    ENROLLED_ON_IP,
    FORWARD,
    FORWARD_EDGE_LAYOUTS,
    IP,
    NODE_COLUMNS,
    NODE_LAYOUTS,
    NODE_TYPES,
    OWNED_BY,
    OWNER_ENROLLED_ON_DEVICE,
    OWNER_ENROLLED_ON_IP,
    PAYMENT_LAYOUT,
    PAYS,
    SEEN_ON_DEVICE,
    SEEN_ON_IP,
    reverse,
)

# The durations a query always gives, whose presence bit is always 1 (docs/features.md,
# "Payment edges" and "Seen-on-payment edges").
ALWAYS_PRESENT = frozenset({"age", "since_last_use"})
# The labels of a training batch.
LABELS = (0, 1)
# The payment-edge columns B4 gives besides the amount and the rail: the age, the earlier pair
# payments and the four gaps. B1 gives the last three, dwell_tracked, left_24h and dwell.
B4_TIMING = tuple(
    column
    for column in PAYMENT_LAYOUT.columns
    if column.name
    in {"age", "pair_prior_count", "pair_gap", "pair_age", "sender_gap", "receiver_gap"}
)


@dataclass(frozen=True)
class Example:
    """An example's key: its world's identity, its Monday t and its scored account. Keys stay
    in a list on the CPU and never reach the model."""

    world: str
    t_ms: int
    account_id: str


@dataclass(frozen=True)
class Leaves:
    """A Monday's leaves by id, from read_hubs: each account hub's measure, and each device
    and IP leaf."""

    accounts: Mapping[str, int]
    devices: Mapping[str, Leaf]
    ips: Mapping[str, Leaf]

    @classmethod
    def of(cls, hubs: Hubs) -> Leaves:
        return cls(
            {hub.account_id: hub.customers_90d for hub in hubs.accounts},
            {leaf.node_id: leaf for leaf in hubs.devices},
            {leaf.node_id: leaf for leaf in hubs.ips},
        )


@dataclass(frozen=True)
class ExampleRows:
    """Every row one example is built from: its Monday's leaves, its scored account's B4, B5
    and B6 answers, and, by account, the B1 to B3 rows of every account node that is no hub,
    the scored account always among them; and, by payment id, the tracked inflows of its
    payments."""

    t_ms: int
    leaves: Leaves
    payments: PaymentGraph
    identity: IdentityGraph
    community: Community
    flows: Mapping[str, MoneyFlow]
    inflows: Mapping[str, TrackedInflow]
    counterparties: Mapping[str, Counterparties]
    activity: Mapping[str, ActivityChange]

    @property
    def scored(self) -> str:
        return self.payments.scored_id


@dataclass(frozen=True)
class Batch:
    """A built batch: its examples' keys, in order, and its raw HeteroData."""

    keys: tuple[Example, ...]
    data: HeteroData


def account_nodes(payments: PaymentGraph, identity: IdentityGraph) -> dict[str, bool]:
    """The account nodes of an example, B4's accounts and B5's co-users' accounts, each once,
    with whether it is another bank's; an account the two give unlike is refused."""
    found: dict[str, bool] = {}
    for account in (*payments.accounts, *identity.accounts):
        held = found.setdefault(account.account_id, account.is_external)
        if held != account.is_external:
            raise ValueError(
                f"The subgraph of {payments.scored_id} gives {account.account_id} two values "
                "of is_external"
            )
    return found


def read_rows(cache: Cache, examples: Sequence[tuple[int, str]]) -> list[ExampleRows]:
    """The rows of each example, (Monday t, scored account), from the cache, in the order
    given. Every row must be cached: the Monday's hubs, B4 to B6 of each scored account, and
    B1 to B3 of every account node of its subgraph that is no hub, and of the scored account."""
    by_monday: dict[int, set[str]] = defaultdict(set)
    for t_ms, account in examples:
        by_monday[t_ms].add(account)
    found: dict[tuple[int, str], ExampleRows] = {}
    for t_ms in sorted(by_monday):
        try:
            found.update(_monday_rows(cache, t_ms, sorted(by_monday[t_ms])))
        finally:
            cache.forget(t_ms)
    return [found[key] for key in examples]


def _monday_rows(cache: Cache, t_ms: int, scored: list[str]) -> dict[tuple[int, str], ExampleRows]:
    hubs = cache.hubs(t_ms)
    if hubs is None:
        raise KeyError(f"The cache holds no hubs of the Monday {t_ms}")
    leaves = Leaves.of(hubs)
    payments = {graph.scored_id: graph for graph in cache.rows(PAYMENT_EVENTS, t_ms, scored)}
    identity = {graph.scored_id: graph for graph in cache.rows(SHARED_IDENTITY, t_ms, scored)}
    community = {row.account_id: row for row in cache.rows(COMMUNITY, t_ms, scored)}
    nodes = {
        account: {
            node
            for node in account_nodes(payments[account], identity[account])
            if node == account or node not in leaves.accounts
        }
        for account in scored
    }
    everyone = sorted(set().union(*nodes.values()))
    flows = cache.rows(MONEY_FLOW, t_ms, everyone)
    flow = {row.account_id: row for row in flows.accounts}
    inflow = {row.payment_id: row for row in flows.tracked_inflows}
    counterparties = {row.account_id: row for row in cache.rows(COUNTERPARTIES, t_ms, everyone)}
    activity = {row.account_id: row for row in cache.rows(ACTIVITY, t_ms, everyone)}
    return {
        (t_ms, account): ExampleRows(
            t_ms,
            leaves,
            payments[account],
            identity[account],
            community[account],
            {node: flow[node] for node in nodes[account]},
            {
                payment.payment_id: inflow[payment.payment_id]
                for payment in payments[account].payments
                if payment.payment_id in inflow
            },
            {node: counterparties[node] for node in nodes[account]},
            {node: activity[node] for node in nodes[account]},
        )
        for account in scored
    }


def _read(columns: Iterable[Column], row: object) -> list[float]:
    """The raw values of columns, each read from row's field of its raw name."""
    values: list[float] = []
    for column in columns:
        for name in column.raw_names:
            if name == f"{column.name}_present" and column.name in ALWAYS_PRESENT:
                values.append(1.0)
            else:
                values.append(float(getattr(row, name)))
    return values


def _degree_zeros(layout: Layout, filled: int) -> list[float]:
    """Zeros for the columns after the first filled, the degree columns among them."""
    return [0.0] * (layout.raw_width - filled)


class _Parts:
    """The nodes and edges of a batch as they are gathered, example after example."""

    def __init__(self) -> None:
        self.nodes: dict[str, list[list[float]]] = {kind: [] for kind in NODE_TYPES}
        self.ends: dict[EdgeType, list[tuple[int, int]]] = {relation: [] for relation in FORWARD}
        self.attrs: dict[EdgeType, list[list[float]]] = {relation: [] for relation in FORWARD}
        self.scored: list[int] = []
        self.community: list[list[float]] = []

    def add(self, rows: ExampleRows) -> None:
        """Add one example's subgraph."""
        scored = rows.scored
        identity = rows.identity
        leaves = rows.leaves
        if identity.scored_id != scored or rows.community.account_id != scored:
            raise ValueError(f"The rows of the example of {scored} belong to other accounts")
        accounts = account_nodes(rows.payments, identity)
        index = {
            ACCOUNT: self._place(ACCOUNT, sorted(accounts)),
            CUSTOMER: self._place(CUSTOMER, [row.customer_id for row in identity.customers]),
            DEVICE: self._place(DEVICE, [row.node_id for row in identity.devices]),
            IP: self._place(IP, [row.node_id for row in identity.ips]),
        }
        for account in sorted(accounts):
            self.nodes[ACCOUNT].append(_account_row(rows, account, accounts[account]))
        for customer in identity.customers:
            values = _read(CUSTOMER_COLUMNS, customer)
            self.nodes[CUSTOMER].append(values + _degree_zeros(NODE_LAYOUTS[CUSTOMER], len(values)))
        for kind, nodes, held in (
            (DEVICE, identity.devices, leaves.devices),
            (IP, identity.ips, leaves.ips),
        ):
            for node in nodes:
                self.nodes[kind].append(_node_row(scored, kind, node, held.get(node.node_id)))
        for payment in rows.payments.payments:
            tracked = None
            if payment.receiver_id not in leaves.accounts:
                tracked = rows.inflows.get(payment.payment_id)
            self._edge(
                PAYS, index, payment.sender_id, payment.receiver_id, _payment_row(payment, tracked)
            )
        for edge in rows.payments.owned_by:
            if edge.customer_id not in index[CUSTOMER]:
                raise ValueError(
                    f"The subgraph of {scored} owns {edge.account_id} by {edge.customer_id}, "
                    "which has no customer row"
                )
            self._edge(OWNED_BY, index, edge.account_id, edge.customer_id, None)
        for edge in identity.enrolled:
            relation = ENROLLED_ON_DEVICE if edge.node_id in index[DEVICE] else ENROLLED_ON_IP
            values = _read(FORWARD_EDGE_LAYOUTS[relation].columns, edge)
            self._edge(relation, index, edge.customer_id, edge.node_id, values)
        for kinds, edges in (
            ((OWNER_ENROLLED_ON_DEVICE, OWNER_ENROLLED_ON_IP), identity.owner_enrolled),
            ((SEEN_ON_DEVICE, SEEN_ON_IP), identity.seen),
        ):
            for edge in edges:
                if edge.account_id != scored and edge.account_id in leaves.accounts:
                    raise ValueError(
                        f"The subgraph of {scored} gives the account hub {edge.account_id} an "
                        "edge to a device or an IP, though a hub is never expanded"
                    )
                relation = kinds[0] if edge.node_id in index[DEVICE] else kinds[1]
                values = _read(FORWARD_EDGE_LAYOUTS[relation].columns, edge)
                self._edge(relation, index, edge.account_id, edge.node_id, values)
        self.scored.append(index[ACCOUNT][scored])
        self.community.append(_read(COMMUNITY_LAYOUT.columns, rows.community))

    def _place(self, kind: str, ids: list[str]) -> dict[str, int]:
        """The position in the batch of each node of an example's type, after the nodes
        before it."""
        first = len(self.nodes[kind])
        return {node: first + position for position, node in enumerate(ids)}

    def _edge(
        self,
        relation: EdgeType,
        index: Mapping[str, Mapping[str, int]],
        source: str,
        target: str,
        values: list[float] | None,
    ) -> None:
        start, _, end = relation
        if source not in index[start] or target not in index[end]:
            raise ValueError(
                f"An edge of {relation[1]} joins {source} or {target}, not in its subgraph"
            )
        self.ends[relation].append((index[start][source], index[end][target]))
        if values is not None:
            self.attrs[relation].append(values)


def _account_row(rows: ExampleRows, account: str, is_external: bool) -> list[float]:
    """An account node's raw row: flag-only for an account hub that is not the scored
    account."""
    hub = rows.leaves.accounts.get(account)
    head = [float(is_external), float(hub is not None), float(hub or 0)]
    if hub is not None and account != rows.scored:
        return head + _degree_zeros(ACCOUNT_LAYOUT, len(head))
    try:
        values = (
            head
            + _read(B1_COLUMNS, rows.flows[account])
            + _read(B2_COLUMNS, rows.counterparties[account])
            + _read(B3_COLUMNS, rows.activity[account])
        )
    except KeyError:
        raise ValueError(f"The example of {rows.scored} lacks B1 to B3 rows of {account}") from None
    return values + _degree_zeros(ACCOUNT_LAYOUT, len(values))


def _node_row(scored: str, kind: str, row: NodeRow, leaf: Leaf | None) -> list[float]:
    """A device's or IP's raw row; a leaf's from read_hubs."""
    if row.leaf != (leaf is not None):
        raise ValueError(
            f"The subgraph of {scored} reads the {kind} {row.node_id} as "
            f"{'a leaf' if row.leaf else 'no leaf'}, unlike read_hubs"
        )
    layout = NODE_LAYOUTS[kind]
    if leaf is not None:
        values = [0.0, float(leaf.customers_90d), 0.0, 0.0, 0.0, 0.0]
        values += [float(leaf.observed_only), 1.0, float(leaf.hub)]
    else:
        # B5's columns 1 to 7, then crowd and hub, both 0 on a node that is no leaf.
        values = _read(NODE_COLUMNS[:7], row) + [0.0, 0.0]
    return values + _degree_zeros(layout, len(values))


def _payment_row(payment: PaymentEvent, tracked: TrackedInflow | None) -> list[float]:
    """A payment edge's raw row, with its receiver's tracked inflow when it has one."""
    if tracked is not None and tracked.account_id != payment.receiver_id:
        raise ValueError(
            f"B1 tracked {payment.payment_id} for {tracked.account_id}, not its receiver"
        )
    values = [float(payment.amount), *(float(payment.rail == rail) for rail in RAILS)]
    values += _read(B4_TIMING, payment)
    if tracked is None:
        return values + [0.0] * 4
    return values + [1.0, tracked.left_24h, float(tracked.dwell_present), float(tracked.dwell_ms)]


def _matrix(rows: list[list[float]], layout: Layout, what: str) -> Tensor:
    """Raw rows as a float64 tensor, once their values fit their columns."""
    found: NDArray[np.float64] = np.array(rows, dtype=np.float64).reshape(
        len(rows), layout.raw_width
    )
    layout.check(found, what)
    return torch.from_numpy(found)


def put_relation(
    data: HeteroData, relation: EdgeType, edge_index: Tensor, edge_attr: Tensor | None
) -> None:
    """Set a forward relation's edges and attributes, and its reverse's: the same edges
    turned around, with the same attributes."""
    data[relation].edge_index = edge_index
    data[reverse(relation)].edge_index = edge_index.flip(0)
    if edge_attr is not None:
        data[relation].edge_attr = edge_attr
        data[reverse(relation)].edge_attr = edge_attr


def build(
    rows: Sequence[ExampleRows],
    world: str,
    labels: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
) -> Batch:
    """The batch of these examples of a world, in the order given: a training batch with
    labels and weights, a scoring batch without either, built alike otherwise."""
    if not rows:
        raise ValueError("A batch holds at least one example")
    if (labels is None) != (weights is None):
        raise ValueError("A training batch has labels and weights, a scoring batch neither")
    parts = _Parts()
    for example in rows:
        parts.add(example)
    data = HeteroData()
    for kind in NODE_TYPES:
        data[kind].x = _matrix(parts.nodes[kind], NODE_LAYOUTS[kind], kind)
    for relation in FORWARD:
        ends = torch.tensor(parts.ends[relation], dtype=torch.long).reshape(-1, 2)
        layout = FORWARD_EDGE_LAYOUTS.get(relation)
        attrs = None if layout is None else _matrix(parts.attrs[relation], layout, relation[1])
        put_relation(data, relation, ends.T.contiguous(), attrs)
    data.scored_index = torch.tensor(parts.scored, dtype=torch.long)
    data.community = _matrix(parts.community, COMMUNITY_LAYOUT, "community")
    if labels is not None and weights is not None:
        _training(data, labels, weights, len(rows))
    degrees.count(data)
    keys = tuple(Example(world, example.t_ms, example.scored) for example in rows)
    return Batch(keys, data)


def _training(data: HeteroData, labels: Sequence[int], weights: Sequence[float], n: int) -> None:
    if len(labels) != n or len(weights) != n:
        raise ValueError(f"A training batch of {n} examples needs {n} labels and weights")
    if any(label not in LABELS for label in labels):
        raise ValueError(f"A label is one of {LABELS}, got {sorted(set(labels))}")
    if not all(math.isfinite(weight) and weight > 0 for weight in weights):
        raise ValueError("A weight is a finite number above 0")
    data.label = torch.tensor(list(labels), dtype=torch.long)
    data.weight = torch.tensor(list(weights), dtype=torch.float64)
