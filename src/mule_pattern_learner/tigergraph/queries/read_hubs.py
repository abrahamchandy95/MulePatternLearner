"""read_hubs: the leaves of one Monday, the accounts, devices and IPs too connected to expand,
each with its measure (docs/design.md, "Shared rules" and "read_hubs")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_CONFIG
from .rows import as_of, parse_row, projected, response_head

SUBGRAPH = DEFAULT_CONFIG.subgraph


@dataclass(frozen=True)
class HubAccount:
    """An account hub: more than the hub threshold of distinct customers owned the other
    accounts of its payments in the 90 days before t (customers_90d)."""

    account_id: str
    customers_90d: int


@dataclass(frozen=True)
class Leaf:
    """A device or IP with more than the crowd threshold of distinct customers in the 90 days
    before t (customers_90d): a crowd, and a hub too above the hub threshold. observed_only
    says no customer has a visible enrolment link to it."""

    node_id: str
    customers_90d: int
    observed_only: bool
    hub: bool


@dataclass(frozen=True)
class Hubs:
    """The leaves of one Monday, each kind sorted by id."""

    accounts: tuple[HubAccount, ...]
    devices: tuple[Leaf, ...]
    ips: tuple[Leaf, ...]


def params(t_ms: int) -> dict[str, Any]:
    """The parameters of the Monday t."""
    return {"t_ms": as_of(t_ms)}


def _sorted[T](found: list[T], ids: list[str], what: str) -> tuple[T, ...]:
    """found, in the order of its ids, once each id is distinct."""
    if len(set(ids)) != len(ids):
        raise ValueError(f"read_hubs printed a repeated id in {what}")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return tuple(found[index] for index in order)


def parse(rows: Sequence[object]) -> Hubs:
    """The leaves of the Monday params(t_ms) asked for. Every one is checked against the
    settings' thresholds, which the query declares too."""
    from . import READ_HUBS, VERSIONS  # the registry imports this module

    response_head(rows, READ_HUBS, VERSIONS[READ_HUBS])
    accounts = [parse_row(HubAccount, row) for row in projected(rows, "accounts", READ_HUBS)]
    for account in accounts:
        if account.customers_90d <= SUBGRAPH.hub_threshold:
            raise ValueError(
                f"{READ_HUBS} printed the account hub {account.account_id} with "
                f"{account.customers_90d} customers"
            )
    nodes: dict[str, tuple[Leaf, ...]] = {}
    for what in ("devices", "ips"):
        found = [parse_row(Leaf, row) for row in projected(rows, what, READ_HUBS)]
        for leaf in found:
            if leaf.customers_90d <= SUBGRAPH.crowd_threshold or leaf.hub != (
                leaf.customers_90d > SUBGRAPH.hub_threshold
            ):
                raise ValueError(f"{READ_HUBS} printed the leaf {leaf} among its {what}")
        nodes[what] = _sorted(found, [leaf.node_id for leaf in found], what)
    return Hubs(
        _sorted(accounts, [account.account_id for account in accounts], "accounts"),
        nodes["devices"],
        nodes["ips"],
    )
