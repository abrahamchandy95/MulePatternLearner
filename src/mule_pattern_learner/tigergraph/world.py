"""The loaded world's identity (docs/design.md, "Cache"): the hash of read_world's fields,
which names every cache and store folder, and the fields PhantomLedger's world record shares
with read_world, under the record's key names. Beside it, the rest of a cached row's key: the
folder of the query versions and settings under a world's identity, and the folder of each
scoring time under it.

A world regenerated or changed has another identity, so stale rows are never read. Every field
is an integer, so a world has the same identity on every run.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json

from ..config import SubgraphConfig
from .queries import (
    LIST_ACCOUNTS,
    MARK_CROWDS,
    READ_ACTIVITY_CHANGE,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_MONEY_FLOW,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    VERSIONS,
)
from .queries.read_world import World
from .queries.rows import as_of

# The fields of read_world that PhantomLedger's world record carries too: each map's entries
# keyed "<field>.<key>" (edges holds the 15 forward edge types alone, as the record does), and
# four single values. The degree sums are read_world's own.
SHARED_MAPS = ("vertices", "edges", "account_types", "payment_rails", "seconds", "cents")
SHARED_VALUES = ("external_accounts", "first_joined_ms", "first_event_ms", "last_event_ms")
# The bytes of a blake2b digest that names a world: 32 hex digits (paths.IDENTITY).
IDENTITY_BYTES = 16
# The queries whose rows the cache of a world holds, in a folder per scoring time: the
# population, the hubs, the crowds, and B1 to B6 (docs/design.md, "Cache").
CACHED_QUERIES = (
    LIST_ACCOUNTS,
    READ_HUBS,
    MARK_CROWDS,
    READ_MONEY_FLOW,
    READ_COUNTERPARTIES,
    READ_ACTIVITY_CHANGE,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    READ_COMMUNITY,
)


def identity(world: World) -> str:
    """The blake2b hash, 16 bytes as 32 hex digits, of the canonical JSON of the world's
    fields: sorted keys at every level and no spaces."""
    return canonical_hash(asdict(world))


def shared_fields(world: World) -> dict[str, int]:
    """The fields PhantomLedger's world record shares with read_world, by the record's key
    names, sorted: mule prepare and mule audit compare these and no other."""
    found: dict[str, int] = {name: getattr(world, name) for name in SHARED_VALUES}
    for name in SHARED_MAPS:
        entries: dict[str, int] = getattr(world, name)
        found.update({f"{name}.{key}": value for key, value in entries.items()})
    return dict(sorted(found.items()))


def canonical_hash(value: object) -> str:
    """The blake2b hash, 16 bytes as 32 hex digits, of the canonical JSON of value: sorted keys
    at every level and no spaces."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=IDENTITY_BYTES).hexdigest()


def cache_key(subgraph: SubgraphConfig) -> str:
    """The folder under a world's identity that holds its cached rows: the hash of the versions
    of CACHED_QUERIES and of the subgraph settings. Rows of other query versions or settings
    live in another folder, so they are never read and never overwritten."""
    versions = {name: VERSIONS[name] for name in CACHED_QUERIES}
    return canonical_hash({"versions": versions, "subgraph": asdict(subgraph)})


def time_folder(t_ms: int) -> str:
    """The folder of the scoring time t in a world's cache, its UTC time to the second with
    colons written as hyphens so that it is a filesystem-safe name (2024-04-03T14-37-12Z). Any
    t that is a whole second after the lookback is a folder."""
    return datetime.fromtimestamp(as_of(t_ms) // 1000, UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
