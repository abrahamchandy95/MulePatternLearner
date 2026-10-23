"""Hop distance from the nearest identified mule (docs/design.md, "Groups"): on each Monday, the
fewest edges between a mule's account and an account with pu_label 1 inside the mule's own
fetched subgraph at that t, of any of the 16 relations and in either direction.

An account node counts as the end; customers, devices and IPs are steps on the way, so a shared
device puts two mule accounts two edges apart. The subgraph is the example's own, as the model
reads it: B4's accounts, payments and ownership edges, and B5's customers, devices, IPs,
co-users' accounts and their enrolment, owner-enrolment and seen edges, from the cache. The
distance is 1, 2 or none within two hops; the scored account itself is never its own nearest
identified mule. It is evaluation's output, never a model input.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence

from ..subgraph.build import ExampleRows, read_rows
from ..subgraph.cache import Cache

# The bins of the distance: one edge, two edges, and not within two hops.
WITHIN = (1, 2)


def hop_distance(rows: ExampleRows, identified: Collection[str]) -> int | None:
    """The fewest edges from the example's scored account to another account the bank
    identified, in its subgraph: 1 or 2, or None when none is within two hops."""
    devices = {node.node_id for node in rows.identity.devices}
    neighbours: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    def join(first: tuple[str, str], second: tuple[str, str]) -> None:
        neighbours[first].add(second)
        neighbours[second].add(first)

    for payment in rows.payments.payments:
        join(("account", payment.sender_id), ("account", payment.receiver_id))
    for owned in rows.payments.owned_by:
        join(("account", owned.account_id), ("customer", owned.customer_id))
    identity = rows.identity

    def node(node_id: str) -> tuple[str, str]:
        return ("device" if node_id in devices else "ip", node_id)

    for enrolled in identity.enrolled:
        join(("customer", enrolled.customer_id), node(enrolled.node_id))
    for edge in (*identity.owner_enrolled, *identity.seen):
        join(("account", edge.account_id), node(edge.node_id))
    start = ("account", rows.scored)
    frontier = {start}
    seen = {start}
    for distance in WITHIN:
        frontier = {near for here in frontier for near in neighbours[here]} - seen
        seen |= frontier
        if any(kind == "account" and name in identified for kind, name in frontier):
            return distance
    return None


def hop_distances(
    cache: Cache, keys: Sequence[tuple[int, str]], identified: Collection[str]
) -> dict[tuple[int, str], int | None]:
    """The hop distance of each (Monday, account) of keys, read from the cache."""
    found: dict[tuple[int, str], int | None] = {}
    for example in read_rows(cache, list(keys)):
        found[(example.t_ms, example.scored)] = hop_distance(example, identified)
    return found
