"""The XGBoost vectors, read from the built batches the GNN reads (docs/features.md, "Ablation and
comparison switches", XGBoost inputs), so the trees see exactly what TigerGraph gives the GNN.

- Own behaviour: the scored account's own row as stored, its columns 1 to 46 with each duration
  as its presence bit and seconds: 50 numbers (OWN_COLUMNS).
- Graph features: the same, then its customer's columns 1 to 8, the mean and then the maximum
  of each of the columns 1 to 9 of the rows of the devices its customer is enrolled on (D_s),
  the same of the IPs, zeros when there are none, and the community block: 50 + 8 + 18 + 18 +
  8 = 102 numbers (GRAPH_COLUMNS).

The scored account's customer C_s is the one its ownership edge reaches, and D_s the devices and
IPs of C_s's enrolment edges in the batch, exactly what the GNN sees (the owner, 24 October
2026; docs/features.md says so); a leaf among them has the row read_hubs gives it. B5's closure
around the scored account's nodes also gives C_s an enrolment edge to a node seen on the
account's payments that C_s is linked to beyond its 8 kept links, so with more than 8 visible
links of a kind D_s holds a node beyond the kept links. Only past evidence is read, so it is no
leak. Each value is the raw one the batch holds before encoding: counts, cents, shares and flags as
they are, durations in seconds. No degree column is read, and no node or edge of another
example, since each example is its own copy of its subgraph.

The vectors are float32, as XGBoost reads them anyway, so a whole selection world's examples
take half the memory (the seconds are divided in float64 first). A cent sum above 2**24 cents
loses its last cents, which a tree never reads.

A training batch's labels and weights travel with its vectors, row by row in the order of the
keys, since the batches hold their examples shuffled and a label table in any other order would
train the trees on the wrong labels without a word. The vectors of a scoring batch carry none.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from ..model.encode import Column, Kind
from ..subgraph.build import Batch
from ..subgraph.layouts import (
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
    IP,
    NODE_COLUMNS,
    OWNED_BY,
)

# The scored account's columns 1 to 46, which come first in its row.
OWN_ACCOUNT_COLUMNS = (*ACCOUNT_LAYOUT.columns[:3], *B1_COLUMNS, *B2_COLUMNS, *B3_COLUMNS)
# Milliseconds in a second: a duration reaches the trees in seconds.
SECOND_MS = 1_000.0


def _names(columns: Iterable[Column]) -> tuple[str, ...]:
    """The names of columns as a vector holds them: a duration's presence bit and seconds."""
    found: list[str] = []
    for column in columns:
        if column.kind is Kind.DURATION:
            found += [f"{column.name}_present", f"{column.name}_s"]
        else:
            found.append(column.name)
    return tuple(found)


OWN_COLUMNS = _names(OWN_ACCOUNT_COLUMNS)
NODE_NAMES = _names(NODE_COLUMNS)
GRAPH_COLUMNS = (
    *OWN_COLUMNS,
    *(f"customer.{name}" for name in _names(CUSTOMER_COLUMNS)),
    *(f"device_mean.{name}" for name in NODE_NAMES),
    *(f"device_max.{name}" for name in NODE_NAMES),
    *(f"ip_mean.{name}" for name in NODE_NAMES),
    *(f"ip_max.{name}" for name in NODE_NAMES),
    *(f"community.{name}" for name in _names(COMMUNITY_LAYOUT.columns)),
)
# The raw positions of the durations' milliseconds in the scored account's row.
MILLISECONDS = [
    ACCOUNT_LAYOUT.index(f"{column.name}_ms")
    for column in OWN_ACCOUNT_COLUMNS
    if column.kind is Kind.DURATION
]


@dataclass(frozen=True)
class Vectors:
    """The XGBoost vectors of examples: each example's (Monday, account) key, and its own and
    graph vectors, a row per example in the order of the keys, float32; and, for examples of
    training batches, each one's label (1 identified, 0 unlabelled) and weight in the same
    order, both empty for examples of scoring batches."""

    keys: tuple[tuple[int, str], ...]
    own: NDArray[np.float32]
    graph: NDArray[np.float32]
    labels: NDArray[np.int64]
    weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        examples = len(self.keys)
        if len(self.own) != examples or len(self.graph) != examples:
            raise ValueError(f"{examples} keys for {len(self.own)} and {len(self.graph)} rows")
        if len(self.labels) != len(self.weights) or len(self.labels) not in (0, examples):
            raise ValueError(
                f"{len(self.labels)} labels and {len(self.weights)} weights for {examples} keys"
            )

    @property
    def labelled(self) -> bool:
        """Whether the vectors are of training batches, each with its label and weight."""
        return len(self.labels) > 0


def _numpy(tensor: Tensor) -> NDArray[np.float64]:
    return tensor.detach().cpu().numpy().astype(np.float64)


def _example_of(nodes: int, owner: NDArray[np.int64]) -> NDArray[np.int64]:
    """The example of each node that is some example's customer, -1 for every other node."""
    found = np.full(nodes, -1, dtype=np.int64)
    found[owner] = np.arange(len(owner))
    return found


def _mean_and_max(
    rows: NDArray[np.float64], example: NDArray[np.int64], examples: int
) -> NDArray[np.float64]:
    """Per example, the mean and then the maximum of each column of the rows that belong to it
    (example gives each row's example, -1 for none), zeros for an example with no row."""
    held = example >= 0
    rows, example = rows[held], example[held]
    width = rows.shape[1]
    sums = np.zeros((examples, width))
    np.add.at(sums, example, rows)
    counts = np.bincount(example, minlength=examples).astype(np.float64)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    maxima = np.full((examples, width), -np.inf)
    np.maximum.at(maxima, example, rows)
    maxima[counts == 0] = 0.0
    return np.concatenate([means, maxima], axis=1)


def vectors(batch: Batch) -> Vectors:
    """The own and graph vectors of every example of a raw built batch, before any switch."""
    data = batch.data
    scored = _numpy(data.scored_index).astype(np.int64)
    examples = len(scored)
    own = _numpy(data[ACCOUNT].x)[scored, : len(OWN_COLUMNS)].copy()
    own[:, MILLISECONDS] /= SECOND_MS
    accounts, customers = _numpy(data[OWNED_BY].edge_index).astype(np.int64)
    owner = np.full(data[ACCOUNT].num_nodes, -1, dtype=np.int64)
    owner[accounts] = customers
    owners = owner[scored]
    if (owners < 0).any():
        raise ValueError("A scored account of the batch has no ownership edge to its customer")
    customer_rows = _numpy(data[CUSTOMER].x)[owners, : len(CUSTOMER_COLUMNS)]
    example = _example_of(data[CUSTOMER].num_nodes, owners)
    parts = [own, customer_rows]
    for kind, relation in ((DEVICE, ENROLLED_ON_DEVICE), (IP, ENROLLED_ON_IP)):
        sources, targets = _numpy(data[relation].edge_index).astype(np.int64)
        rows = _numpy(data[kind].x)[targets, : len(NODE_COLUMNS)]
        parts.append(_mean_and_max(rows, example[sources], examples))
    parts.append(_numpy(data.community))
    keys = tuple((key.t_ms, key.account_id) for key in batch.keys)
    if "label" in data:
        labels = data.label.detach().cpu().numpy().astype(np.int64)
        weights = data.weight.detach().cpu().numpy().astype(np.float64)
    else:
        labels, weights = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return Vectors(
        keys,
        own.astype(np.float32),
        np.concatenate(parts, axis=1).astype(np.float32),
        labels,
        weights,
    )


def vectors_of(batches: Iterable[Batch]) -> Vectors:
    """The vectors of every example of batches, in their order; batches are all training
    batches, whose labels and weights come with the vectors, or all scoring batches."""
    found = [vectors(batch) for batch in batches]
    if not found:
        raise ValueError("Vectors need at least one batch")
    if len({part.labelled for part in found}) != 1:
        raise ValueError("The batches are some training batches and some scoring batches")
    return Vectors(
        tuple(key for part in found for key in part.keys),
        np.concatenate([part.own for part in found]),
        np.concatenate([part.graph for part in found]),
        np.concatenate([part.labels for part in found]),
        np.concatenate([part.weights for part in found]),
    )
