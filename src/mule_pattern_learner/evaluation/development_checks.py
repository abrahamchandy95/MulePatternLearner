"""The checks mule audit runs on development worlds beside the feature audit (docs/design.md,
"Shared rules" and "The feature audit"; docs/features.md, "Gates the features add"), read from
the answer key and the cache of the audit sample (evaluation.sample):

- No positive is an account hub: no active true mule's account is an account hub of
  read_hubs in a week it is active. It passes or fails.
- The share of ring devices and IPs that are leaves in their ring's active weeks, reported
  and judged near 0 by the owner. A device or IP is a ring node of a ring in a week when the
  own edges (owner-enrolment or seen) of active true mules of the ring that belong to at least
  two different customers reach it in their subgraphs that week; it is a leaf when B5 reads
  it as one.
- Whether busy ordinary accounts' card spending fills B1's short dwell shares, reported: the
  weighted means of left_15m to left_1d and left_by_card, and the card part of left_1d
  (left_1d times left_by_card), over busy non-mule account-weeks (those at or above the 90th
  percentile of non-mule out_30d by weight), over every non-mule account-week, and over the
  active true mules. If the card part fills them, the design's answer is to count transfer and
  cash outflows only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..subgraph.build import ExampleRows

# The B1 shares the card-spending check reads, and the B3 count that makes an account busy.
SHORT_SHARES = ("left_15m", "left_1h", "left_6h", "left_1d")
BUSY_QUANTILE = 0.9


@dataclass(frozen=True)
class OwnNodes:
    """The devices and IPs an example's scored account reaches by its own owner-enrolment and
    seen edges, each as (kind, id), with whether B5 reads it as a leaf."""

    nodes: Mapping[tuple[str, str], bool]


def own_nodes(rows: ExampleRows) -> OwnNodes:
    """The scored account's own devices and IPs in its subgraph."""
    scored = rows.scored
    identity = rows.identity
    leaves = {("device", node.node_id): node.leaf for node in identity.devices}
    leaves |= {("ip", node.node_id): node.leaf for node in identity.ips}
    found: dict[tuple[str, str], bool] = {}
    for edge in (*identity.owner_enrolled, *identity.seen):
        if edge.account_id != scored:
            continue
        for key in (("device", edge.node_id), ("ip", edge.node_id)):
            if key in leaves:
                found[key] = leaves[key]
    return OwnNodes(found)


def positive_hubs(
    mule_rows: Sequence[tuple[int, str]], hubs: Mapping[int, frozenset[str]]
) -> pd.DataFrame:
    """The active true-mule account-weeks whose account is an account hub that week."""
    found = [(t_ms, account) for t_ms, account in mule_rows if account in hubs[t_ms]]
    return pd.DataFrame(found, columns=["t_ms", "account_id"]).astype(
        {"t_ms": "int64", "account_id": "str"}
    )


@dataclass(frozen=True)
class ActiveMule:
    """An active true mule in a week, as the ring nodes read it: its t, its account, the
    customer that owns the account that week (None when none is known) and its ring (-1 for
    none)."""

    t_ms: int
    account_id: str
    customer_id: str | None
    ring: int


def customer_of(rows: ExampleRows) -> str | None:
    """The customer that owns the scored account, by its ownership edge in the subgraph."""
    for edge in rows.payments.owned_by:
        if edge.account_id == rows.scored:
            return edge.customer_id
    return None


def other_customer(mine: str | None, theirs: str | None) -> bool:
    """Whether two accounts belong to two different customers, both of them known."""
    return mine is not None and theirs is not None and mine != theirs


def ring_nodes(
    mules: Sequence[ActiveMule], nodes: Sequence[OwnNodes]
) -> dict[tuple[int, int, tuple[str, str]], bool]:
    """The ring nodes, as (ring, t, (kind, id)) with whether B5 reads the node as a leaf.
    A device or IP is a ring's in a week when the own edges of active true mules of that
    ring, belonging to at least two different customers, reach it that week. nodes holds each
    mule's own nodes, in the order of mules. A ringless mule, or one with no known customer,
    makes no ring node."""
    customers: dict[tuple[int, int, tuple[str, str]], set[str]] = defaultdict(set)
    leaf: dict[tuple[int, int, tuple[str, str]], bool] = {}
    for mule, own in zip(mules, nodes, strict=True):
        if mule.ring < 0 or mule.customer_id is None:
            continue
        for node, is_leaf in own.nodes.items():
            key = (mule.ring, mule.t_ms, node)
            customers[key].add(mule.customer_id)
            leaf[key] = is_leaf
    return {key: leaf[key] for key, held in customers.items() if len(held) >= 2}


def ring_leaf_share(mules: Sequence[ActiveMule], nodes: Sequence[OwnNodes]) -> pd.DataFrame:
    """The ring nodes (ring_nodes) and the leaves among them, over every ring and week."""
    found = ring_nodes(mules, nodes)
    leaves = sum(found.values())
    share = leaves / len(found) if found else float("nan")
    return pd.DataFrame({"ring_nodes": [len(found)], "leaves": [leaves], "leaf_share": [share]})


def _mean(values: NDArray[np.float64], weight: NDArray[np.float64]) -> float:
    total = float(weight.sum())
    return float(values @ weight) / total if total > 0 else float("nan")


def card_spending(
    shares: NDArray[np.float64],
    out_30d: NDArray[np.float64],
    positive: NDArray[np.bool_],
    weight: NDArray[np.float64],
) -> pd.DataFrame:
    """The card-spending report: shares holds, a row per account-week, left_15m to left_1d
    and left_by_card in that order; out_30d its payments sent in the last 30 days."""
    ordinary = ~positive
    busy = np.zeros(len(positive), dtype=bool)
    if ordinary.any():
        values = out_30d[ordinary]
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weight[ordinary][order])
        edge = values[order][
            min(int(np.searchsorted(cumulative, BUSY_QUANTILE * cumulative[-1])), len(order) - 1)
        ]
        busy = ordinary & (out_30d >= edge)
    card_part = shares[:, 3] * shares[:, 4]
    rows: list[dict[str, object]] = []
    for group, chosen in (("busy_ordinary", busy), ("ordinary", ordinary), ("mules", positive)):
        row: dict[str, object] = {"group": group, "account_weeks": int(chosen.sum())}
        for column, name in enumerate((*SHORT_SHARES, "left_by_card")):
            row[name] = _mean(shares[chosen, column], weight[chosen])
        row["card_part_of_left_1d"] = _mean(card_part[chosen], weight[chosen])
        rows.append(row)
    return pd.DataFrame(rows)


def flow_shares(rows: ExampleRows) -> tuple[list[float], float]:
    """The scored account's left_15m to left_1d and left_by_card, and its out_30d."""
    flow = rows.flows[rows.scored]
    shares = [float(getattr(flow, name)) for name in (*SHORT_SHARES, "left_by_card")]
    return shares, float(rows.activity[rows.scored].out_30d)
