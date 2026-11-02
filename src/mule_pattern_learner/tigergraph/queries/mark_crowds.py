"""mark_crowds: the crowds at one time t, the devices and IPs too shared to expand, stamped on the
graph as crowd_at_ms and printed each with its measure (docs/design.md, "Shared rules")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_CONFIG
from .rows import as_of, parse_row, projected, response_head

SUBGRAPH = DEFAULT_CONFIG.subgraph


@dataclass(frozen=True)
class Crowd:
    """A device or IP with more than the crowd threshold of distinct customers in the 90 days
    before t (customers_90d), and a hub too above the hub threshold. observed_only says no
    customer has a visible enrolment link to it."""

    node_id: str
    customers_90d: int
    observed_only: bool
    hub: bool


@dataclass(frozen=True)
class Crowds:
    """The crowds at one time t, each kind sorted by id."""

    devices: tuple[Crowd, ...]
    ips: tuple[Crowd, ...]


def params(t_ms: int) -> dict[str, Any]:
    """The parameters of the time t."""
    return {"t_ms": as_of(t_ms)}


def parse(rows: Sequence[object]) -> Crowds:
    """The crowds stamped at the time params(t_ms) asked for. Every one is checked against the
    settings' thresholds, which the query declares too."""
    from . import MARK_CROWDS  # the registry imports this module

    response_head(rows, MARK_CROWDS)
    found: dict[str, tuple[Crowd, ...]] = {}
    for what in ("devices", "ips"):
        crowds = [parse_row(Crowd, row) for row in projected(rows, what, MARK_CROWDS)]
        for crowd in crowds:
            if crowd.customers_90d <= SUBGRAPH.crowd_threshold or crowd.hub != (
                crowd.customers_90d > SUBGRAPH.hub_threshold
            ):
                raise ValueError(f"{MARK_CROWDS} printed {crowd} among its {what}")
        if len({crowd.node_id for crowd in crowds}) != len(crowds):
            raise ValueError(f"{MARK_CROWDS} printed a repeated id in {what}")
        found[what] = tuple(sorted(crowds, key=lambda crowd: crowd.node_id))
    return Crowds(found["devices"], found["ips"])
