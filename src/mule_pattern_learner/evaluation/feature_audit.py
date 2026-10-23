"""The feature audit of mule audit, on development worlds only (docs/design.md, "The feature
audit").

Over the audit sample's account-Mondays (evaluation.sample: every active true mule, and a keyed
sample of non-mule account-Mondays, each standing for its share of them), each column's
univariate separation of active true mules from non-mules: its weighted ROC AUC against
is_mule, with a ring-resampled interval (evaluation.uncertainty). The columns are read from the
built batch, raw before encoding (each duration as its presence bit and its milliseconds):

- the scored account's row, its degree columns among them (account.<column>);
- its customer's row, with its degree columns (customer.<column>);
- the rows of its own devices and its own IPs, those its owner-enrolment and seen edges reach,
  aggregated by mean and by maximum (device_mean.<column>, device_max.<column>, ip_...);
- the fields of its own edges, likewise: the payments it sent and those it received apart,
  its owner-enrolment edges and its seen edges (pays_out_mean.<field>, ...);
- its community block (community.<column>).

An aggregate over no row is 0, as the model sees no edge.

- A column whose interval contains 0.5 has no separation: it is reported, to be researched or
  dropped.
- A column whose ROC AUC is at least audit.separation_auc (0.95), or at most 1 minus it,
  separates almost perfectly: a suspected leak or shortcut, which stops the run.
- A near-pure region stops the run too, since a marker held by a minority of mules and by no
  ordinary account keeps its ROC AUC near 0.5: for each column, each value of a flag (a column
  holding only 0 and 1) and its top and bottom audit.near_pure_tail of the account-Mondays by
  weight (with ties at the edge inside), a region holding at least
  audit.near_pure_min_mule_share of the active true-mule account-Mondays in which true mules
  are more than audit.near_pure_purity of the weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from torch import Tensor

from ..config import AuditConfig
from ..model.encode import EdgeType, Layout
from ..subgraph.build import Batch
from ..subgraph.layouts import (
    ACCOUNT,
    ACCOUNT_LAYOUT,
    COMMUNITY_LAYOUT,
    CUSTOMER,
    CUSTOMER_LAYOUT,
    DEVICE,
    DEVICE_LAYOUT,
    ENROLMENT_LAYOUT,
    IP,
    IP_LAYOUT,
    OWNED_BY,
    OWNER_ENROLLED_ON_DEVICE,
    OWNER_ENROLLED_ON_IP,
    PAYMENT_LAYOUT,
    PAYS,
    SEEN_LAYOUT,
    SEEN_ON_DEVICE,
    SEEN_ON_IP,
)
from .uncertainty import Clusters, Ranked, intervals

# The edges of the scored account whose fields are aggregated: by name, the relations, the
# end the scored account is at (0 the source, 1 the target), and their layout.
EDGES: tuple[tuple[str, tuple[EdgeType, ...], int, Layout], ...] = (
    ("pays_out", (PAYS,), 0, PAYMENT_LAYOUT),
    ("pays_in", (PAYS,), 1, PAYMENT_LAYOUT),
    ("owner_enrolled", (OWNER_ENROLLED_ON_DEVICE, OWNER_ENROLLED_ON_IP), 0, ENROLMENT_LAYOUT),
    ("seen", (SEEN_ON_DEVICE, SEEN_ON_IP), 0, SEEN_LAYOUT),
)
# The kinds of node the scored account reaches by its own edges, and those edges.
NODES: tuple[tuple[str, str, Layout, tuple[EdgeType, ...]], ...] = (
    ("device", DEVICE, DEVICE_LAYOUT, (OWNER_ENROLLED_ON_DEVICE, SEEN_ON_DEVICE)),
    ("ip", IP, IP_LAYOUT, (OWNER_ENROLLED_ON_IP, SEEN_ON_IP)),
)


def _aggregated(prefix: str, layout: Layout) -> list[str]:
    return [f"{prefix}_{how}.{name}" for how in ("mean", "max") for name in layout.raw_names]


# The audited columns, in the order of audit_columns' values.
COLUMNS: tuple[str, ...] = (
    *(f"account.{name}" for name in ACCOUNT_LAYOUT.raw_names),
    *(f"customer.{name}" for name in CUSTOMER_LAYOUT.raw_names),
    *(column for prefix, _, layout, _ in NODES for column in _aggregated(prefix, layout)),
    *(column for prefix, _, _, layout in EDGES for column in _aggregated(prefix, layout)),
    *(f"community.{name}" for name in COMMUNITY_LAYOUT.raw_names),
)


def _numpy(tensor: Tensor) -> NDArray[np.float64]:
    return tensor.detach().cpu().numpy().astype(np.float64)


def _grouped(
    rows: NDArray[np.float64], example: NDArray[np.int64], examples: int
) -> NDArray[np.float64]:
    """Per example, the mean then the maximum of each column of its rows (example gives each
    row's example), zeros for an example with none."""
    width = rows.shape[1]
    sums = np.zeros((examples, width))
    np.add.at(sums, example, rows)
    counts = np.bincount(example, minlength=examples).astype(np.float64)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    maxima = np.full((examples, width), -np.inf)
    np.maximum.at(maxima, example, rows)
    maxima[counts == 0] = 0.0
    return np.concatenate([means, maxima], axis=1)


def audit_columns(batch: Batch) -> NDArray[np.float64]:
    """The audited columns of every example of a raw built batch, a row per example."""
    data = batch.data
    scored = _numpy(data.scored_index).astype(np.int64)
    examples = len(scored)
    accounts = np.full(int(data[ACCOUNT].num_nodes), -1, dtype=np.int64)
    accounts[scored] = np.arange(examples)
    parts = [_numpy(data[ACCOUNT].x)[scored]]
    sources, targets = _numpy(data[OWNED_BY].edge_index).astype(np.int64)
    owner = np.full(len(accounts), -1, dtype=np.int64)
    owner[sources] = targets
    owners = owner[scored]
    if (owners < 0).any():
        raise ValueError("A scored account of the batch has no ownership edge to its customer")
    parts.append(_numpy(data[CUSTOMER].x)[owners])
    for _, kind, _, relations in NODES:
        pairs: set[tuple[int, int]] = set()
        for relation in relations:
            ends = _numpy(data[relation].edge_index).astype(np.int64)
            for source, target in zip(ends[0].tolist(), ends[1].tolist(), strict=True):
                if accounts[source] >= 0:
                    pairs.add((int(accounts[source]), target))
        ordered = sorted(pairs)
        example = np.array([pair[0] for pair in ordered], dtype=np.int64)
        nodes = np.array([pair[1] for pair in ordered], dtype=np.int64)
        parts.append(_grouped(_numpy(data[kind].x)[nodes], example, examples))
    for _, relations, end, layout in EDGES:
        rows_found: list[NDArray[np.float64]] = []
        examples_found: list[NDArray[np.int64]] = []
        for relation in relations:
            ends = _numpy(data[relation].edge_index).astype(np.int64)
            held = accounts[ends[end]] if ends.shape[1] else np.zeros(0, dtype=np.int64)
            chosen = held >= 0
            rows_found.append(_numpy(data[relation].edge_attr)[chosen])
            examples_found.append(held[chosen])
        rows = np.concatenate(rows_found).reshape(-1, layout.raw_width)
        parts.append(_grouped(rows, np.concatenate(examples_found), examples))
    parts.append(_numpy(data.community))
    found = np.concatenate(parts, axis=1)
    if found.shape[1] != len(COLUMNS):
        raise ValueError(f"The audited columns are {found.shape[1]} wide, not {len(COLUMNS)}")
    return found


@dataclass(frozen=True)
class FeatureAudit:
    """The audit's table, a row per column: its ROC AUC and interval, whether it has no
    separation, whether it separates almost perfectly, and its near-pure regions; and whether
    the run may go on."""

    table: pd.DataFrame

    @property
    def passed(self) -> bool:
        """Whether no column separates almost perfectly or holds a near-pure region."""
        return not bool((self.table["separates"] | (self.table["near_pure"] != "")).any())

    @property
    def stopping(self) -> tuple[str, ...]:
        """The columns that stop the run, by name."""
        stops = self.table["separates"] | (self.table["near_pure"] != "")
        return tuple(str(name) for name in self.table.loc[stops, "column"])


def _edge(
    values: NDArray[np.float64], weight: NDArray[np.float64], tail: float, top: bool
) -> float:
    """The value at which the top (or bottom) tail of the weight is reached, its ties inside."""
    order = np.argsort(-values if top else values, kind="stable")
    cumulative = np.cumsum(weight[order])
    reached = int(np.searchsorted(cumulative, tail * cumulative[-1]))
    return float(values[order][min(reached, len(order) - 1)])


def near_pure(
    values: NDArray[np.float64],
    positive: NDArray[np.bool_],
    weight: NDArray[np.float64],
    audit: AuditConfig,
) -> list[str]:
    """The near-pure regions of one column, by name: value=<v> for a flag value, top and
    bottom for its tails."""
    regions: list[tuple[str, NDArray[np.bool_]]] = []
    if np.isin(values, (0.0, 1.0)).all():
        regions += [(f"value={value:g}", values == value) for value in (0.0, 1.0)]
    top = _edge(values, weight, audit.near_pure_tail, top=True)
    bottom = _edge(values, weight, audit.near_pure_tail, top=False)
    regions += [("top", values >= top), ("bottom", values <= bottom)]
    mules = float(weight[positive].sum())
    found: list[str] = []
    for name, region in regions:
        held = float(weight[region & positive].sum())
        total = float(weight[region].sum())
        if (
            mules > 0
            and held >= audit.near_pure_min_mule_share * mules
            and total > 0
            and held / total > audit.near_pure_purity
        ):
            found.append(name)
    return found


def feature_audit(
    matrix: NDArray[np.float64],
    positive: NDArray[np.bool_],
    weight: NDArray[np.float64],
    keys: Sequence[str],
    audit: AuditConfig,
    seed: int,
    columns: Sequence[str] = COLUMNS,
) -> FeatureAudit:
    """The audit of every column of matrix (a row per account-Monday of the audit sample, in
    the order of positive, weight and the cluster names keys)."""
    if matrix.shape != (len(positive), len(columns)):
        raise ValueError(
            f"The audited matrix is {matrix.shape}, not ({len(positive)}, {len(columns)})"
        )
    clusters = Clusters.of(keys, positive)
    ranked = [Ranked(matrix[:, column], positive) for column in range(len(columns))]
    found = intervals(
        [rank.roc_auc for rank in ranked],
        clusters,
        weight,
        replicates=audit.replicates,
        level=audit.interval,
        seed=seed,
    )
    rows: list[dict[str, object]] = []
    for column, (name, rank, bounds) in enumerate(zip(columns, ranked, found, strict=True)):
        auc = rank.roc_auc(weight)
        low, high = bounds if bounds is not None else (np.nan, np.nan)
        rows.append(
            {
                "column": name,
                "roc_auc": auc,
                "low": low,
                "high": high,
                "no_separation": bounds is None or low <= 0.5 <= high,
                "separates": bool(
                    np.isfinite(auc)
                    and (auc >= audit.separation_auc or auc <= 1 - audit.separation_auc)
                ),
                "near_pure": ",".join(near_pure(matrix[:, column], positive, weight, audit)),
            }
        )
    return FeatureAudit(pd.DataFrame(rows))
