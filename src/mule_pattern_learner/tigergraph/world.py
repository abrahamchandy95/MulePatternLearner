"""The loaded world's identity (docs/design.md, "Cache"): the hash of read_world's fields,
which names every cache and store folder, and the fields PhantomLedger's world record shares
with read_world, under the record's key names.

A world regenerated or changed has another identity, so stale rows are never read. Every field
is an integer, so a world has the same identity on every run.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .queries.read_world import World

# The fields of read_world that PhantomLedger's world record carries too: each map's entries
# keyed "<field>.<key>" (edges holds the 15 forward edge types alone, as the record does), and
# four single values. The degree sums are read_world's own.
SHARED_MAPS = ("vertices", "edges", "account_types", "payment_rails", "seconds", "cents")
SHARED_VALUES = ("external_accounts", "first_joined_ms", "first_event_ms", "last_event_ms")
# The bytes of a blake2b digest that names a world: 32 hex digits (paths.IDENTITY).
IDENTITY_BYTES = 16


def identity(world: World) -> str:
    """The blake2b hash, 16 bytes as 32 hex digits, of the canonical JSON of the world's
    fields: sorted keys at every level and no spaces."""
    text = json.dumps(asdict(world), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=IDENTITY_BYTES).hexdigest()


def shared_fields(world: World) -> dict[str, int]:
    """The fields PhantomLedger's world record shares with read_world, by the record's key
    names, sorted: mule prepare and mule audit compare these and no other."""
    found: dict[str, int] = {name: getattr(world, name) for name in SHARED_VALUES}
    for name in SHARED_MAPS:
        entries: dict[str, int] = getattr(world, name)
        found.update({f"{name}.{key}": value for key, value in entries.items()})
    return dict(sorted(found.items()))
