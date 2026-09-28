"""The cache tiers of a ContextSource: its in-memory LRU, and later a disk tier.

A tier holds context rows by hop and ContextKey (ContextTier). The source asks its
tiers before TigerGraph and gives them the rows it obtained, so every tier has the same
two operations: get the rows it holds of some keys, and put the rows a fetch obtained.
MemoryTier is the bounded LRU of the rows the source serves.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from ..contract.graph_schema import ContextKey


class ContextTier(Protocol):
    """One cache tier of a ContextSource, keyed by (hop, ContextKey).

    get returns the rows the tier holds of keys, and put hands it the rows one fetch
    obtained (``rows``) with every key the fetch asked for, in the fetch's order
    (``keys``). close frees what the tier holds in memory. The source calls a tier from
    one thread at a time unless the tier says otherwise.
    """

    def get(self, hop: int, keys: Iterable[ContextKey]) -> dict[ContextKey, dict[str, Any]]: ...
    def put(
        self, hop: int, keys: Sequence[ContextKey], rows: Mapping[ContextKey, dict[str, Any]]
    ) -> None: ...
    def close(self) -> None: ...


class MemoryTier:
    """The bounded in-memory LRU of the rows a ContextSource serves.

    It is keyed by (hop, ContextKey), because roots and children use different
    candidate pools and feature flags. put keeps the new rows, then marks every key of
    the fetch it holds as the most recent, in the fetch's key order, so recency does
    not depend on thread timing; only then are the least recent rows beyond
    ``capacity`` dropped. A capacity of 0 keeps nothing.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.rows: OrderedDict[tuple[int, ContextKey], dict[str, Any]] = OrderedDict()

    def get(self, hop: int, keys: Iterable[ContextKey]) -> dict[ContextKey, dict[str, Any]]:
        """The rows held of keys; their recency changes only with put."""
        found: dict[ContextKey, dict[str, Any]] = {}
        for key in keys:
            row = self.rows.get((hop, key))
            if row is not None:
                found[key] = row
        return found

    def put(
        self, hop: int, keys: Sequence[ContextKey], rows: Mapping[ContextKey, dict[str, Any]]
    ) -> None:
        for key in keys:
            if key in rows and self.capacity:
                self.rows[(hop, key)] = rows[key]
            if (hop, key) in self.rows:
                self.rows.move_to_end((hop, key))
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def close(self) -> None:
        self.rows.clear()
