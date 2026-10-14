"""Fetched subgraphs as one PyG HeteroData, the model's input (docs/design.md, "The subgraph
query" and "The model").

Each seed's subgraph is a component of its own, its nodes laid out after those of the seeds
before it in the order its lists give them. Within a subgraph a node reached by two paths
is one node, so a cycle such as (Z, device, A, payment, Z) stays a cycle; two seeds that
reach the same account each hold a copy, so a seed's score never depends on the other seeds
of its batch.

- Node types account, party, device, ip and address, each with x, one row per node.
- Each relation of the response, from an account, and its reverse (rev_ and its name),
  each with edge_index and edge_attr, the reverse sharing its relation's edge_attr.
- seed_index, each seed's account node, and as_of_ms, each seed's as-of time, in the order
  of the batches and of their seeds.

A row of x or edge_attr has its Layout's columns: the plain numbers (categories one-hot,
flags 0 or 1, counts and amounts as log1p), a presence bit per duration, then the Fourier
encoding of each duration (model.time_encoding), zero when it is absent. The durations are
encoded on the device the HeteroData is built on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from ..model.time_encoding import WIDTH, encode
from ..tigergraph.queries import (
    NODES,
    RELATIONS,
    AccountNode,
    IdentityNode,
    LinkEdge,
    PartyNode,
    PaymentEdge,
    SentEdge,
    SubgraphBatch,
)

type Row = AccountNode | PartyNode | IdentityNode | PaymentEdge | LinkEdge | SentEdge

# The node type of each node list of the response.
NODE_TYPES = {
    "accounts": "account",
    "parties": "party",
    "devices": "device",
    "ips": "ip",
    "addresses": "address",
}
# PhantomLedger's account types and payment rails; a Zelle_Transfer's rail is zelle.
ACCOUNT_TYPES = ("deposit", "credit", "brokerage", "gl", "unknown")
RAILS = ("ach", "card", "cash", "check", "internal", "unknown", "zelle")
# How _host_columns turns a count into a feature. A model's fingerprint holds it
# (training.trainer), so rename it whenever the transform changes.
COUNT_TRANSFORM = "log1p"


@dataclass(frozen=True)
class Layout:
    """How the rows of a node list or relation become features: categories one-hot over
    their values, flags 0 or 1, counts (amounts among them) as log1p, and durations, each
    with the boolean field that says it is present: "" when it always is, and "not " before
    the field when that field says it is absent."""

    categories: tuple[tuple[str, tuple[str, ...]], ...] = ()
    flags: tuple[str, ...] = ()
    counts: tuple[str, ...] = ()
    durations: tuple[tuple[str, str], ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """The name of each column of x or edge_attr."""
        one_hot = [f"{name}={value}" for name, values in self.categories for value in values]
        present = [f"{name}:present" for name, _ in self.durations]
        encoded = [f"{name}:fourier{i}" for name, _ in self.durations for i in range(WIDTH)]
        return (*one_hot, *self.flags, *self.counts, *present, *encoded)


LAYOUTS: dict[type[Row], Layout] = {
    AccountNode: Layout(
        categories=(("account_type", ACCOUNT_TYPES),),
        flags=("is_external", "closed"),
        counts=(
            "in_7d",
            "in_30d",
            "in_365d",
            "out_7d",
            "out_30d",
            "out_365d",
            "amount_in_30d",
            "amount_out_30d",
            "counterparties_in_30d",
            "counterparties_out_30d",
            "first_inflows_7d",
            "first_inflows_30d",
            "passed_on_30d",
        ),
        durations=(
            ("age_ms", ""),
            ("token_age_ms", "token_present"),
            ("since_last_in_ms", "last_in_present"),
            ("since_last_out_ms", "last_out_present"),
        ),
    ),
    PartyNode: Layout(counts=("open_accounts", "closed_accounts"), durations=(("age_ms", ""),)),
    IdentityNode: Layout(
        flags=("observed_only", "hub"),
        counts=("parties_now", "parties_ever", "accounts", "new_parties_30d"),
        durations=(("age_ms", ""),),
    ),
    PaymentEdge: Layout(
        categories=(("rail", RAILS),),
        flags=("first_in_pair",),
        counts=("amount", "prior_count"),
        durations=(
            ("age_ms", ""),
            ("gap_ms", "gap_present"),
            ("pair_age_ms", "gap_present"),
            ("next_outflow_ms", "next_outflow_present"),
        ),
    ),
    LinkEdge: Layout(
        flags=("active",),
        durations=(("since_start_ms", ""), ("since_end_ms", "not active"), ("held_ms", "")),
    ),
    SentEdge: Layout(counts=("uses",), durations=(("since_first_ms", ""), ("since_last_ms", ""))),
}


def build(batches: Sequence[SubgraphBatch], *, device: torch.device | str = "cpu") -> HeteroData:
    """The subgraphs of batches, each of one as-of time, as one HeteroData on device."""
    seeds = [(subgraph, batch.as_of_ms) for batch in batches for subgraph in batch.subgraphs]
    if not seeds:
        raise ValueError("A HeteroData needs at least one seed")
    rows: dict[str, list[Row]] = {name: [] for name in (*NODES, *RELATIONS)}
    ends: dict[str, list[tuple[int, int]]] = {name: [] for name in RELATIONS}
    seed_index: list[int] = []
    for subgraph, _ in seeds:
        index: dict[str, dict[str, int]] = {}
        for name in NODES:
            nodes: tuple[AccountNode | PartyNode | IdentityNode, ...] = getattr(subgraph, name)
            first = len(rows[name])
            index[name] = {node.id: first + position for position, node in enumerate(nodes)}
            if len(index[name]) != len(nodes):
                raise ValueError(f"The subgraph of {subgraph.seed} repeats a node of {name}")
            rows[name] += nodes
        seed_index.append(index["accounts"][subgraph.seed])
        for name, (_, start, end) in RELATIONS.items():
            edges: tuple[PaymentEdge | LinkEdge | SentEdge, ...] = getattr(subgraph, name)
            ends[name] += [(index[start][edge.from_id], index[end][edge.to_id]) for edge in edges]
            rows[name] += edges
    data = HeteroData()
    for name, kind in NODES.items():
        data[NODE_TYPES[name]].x = _features(LAYOUTS[kind], rows[name], device)
    for name, (kind, start, end) in RELATIONS.items():
        source, target = NODE_TYPES[start], NODE_TYPES[end]
        pairs = torch.tensor(ends[name], dtype=torch.long, device=device).reshape(-1, 2)
        edge_attr = _features(LAYOUTS[kind], rows[name], device)
        data[source, name, target].edge_index = pairs.T.contiguous()
        data[source, name, target].edge_attr = edge_attr
        data[target, f"rev_{name}", source].edge_index = pairs.flip(1).T.contiguous()
        data[target, f"rev_{name}", source].edge_attr = edge_attr
    data.seed_index = torch.tensor(seed_index, dtype=torch.long, device=device)
    data.as_of_ms = torch.tensor([t for _, t in seeds], dtype=torch.long, device=device)
    return data


def _features(layout: Layout, rows: Sequence[Row], device: torch.device | str) -> Tensor:
    """The rows' features, laid out as layout's columns, with their durations encoded on
    device."""
    numbers, durations, present = _host_columns(layout, rows)
    bits = torch.as_tensor(present, device=device)
    encoded = encode(torch.as_tensor(durations, device=device)) * bits[..., None]
    plain = torch.as_tensor(numbers, device=device)
    return torch.cat([plain, bits.to(torch.float32), encoded.flatten(1)], dim=1)


def _host_columns(
    layout: Layout, rows: Sequence[Row]
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.bool_]]:
    """The rows' plain numbers, their durations in milliseconds and the durations' presence
    bits, each a matrix with one row per row."""

    def column(name: str) -> list[Any]:
        return [getattr(row, name) for row in rows]

    numbers: list[NDArray[np.float64]] = []
    for name, values in layout.categories:
        found = column(name)
        unknown = set(found) - set(values)
        if unknown:
            raise ValueError(f"{name} {sorted(unknown)} is not one of {list(values)}")
        numbers += [np.array([item == value for item in found], np.float64) for value in values]
    numbers += [np.array(column(name), np.float64) for name in layout.flags]
    for name in layout.counts:
        counts = np.array(column(name), np.float64)
        if np.any(counts < 0):
            raise ValueError(f"{name} is negative")
        numbers.append(np.log1p(counts))
    durations = [column(name) for name, _ in layout.durations]
    present: list[list[bool]] = []
    for _, flag in layout.durations:
        absent = flag.startswith("not ")
        found = column(flag.removeprefix("not ")) if flag else [True] * len(rows)
        present.append([bool(value) != absent for value in found])
    return (
        _matrix(numbers, np.float32, len(rows)),
        _matrix(durations, np.int64, len(rows)),
        _matrix(present, np.bool_, len(rows)),
    )


def _matrix[D: np.generic](columns: Sequence[Any], dtype: type[D], rows: int) -> NDArray[D]:
    """columns side by side, rows by len(columns), even when either is 0."""
    return np.array(columns, dtype=dtype).T.reshape(rows, len(columns))
