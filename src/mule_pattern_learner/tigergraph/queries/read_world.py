"""read_world: what identifies the loaded world, all integers (docs/design.md, "Cache").
tigergraph.world hashes it into the identity that keys every cache and store folder, and names
the fields PhantomLedger's world record shares with it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from ...contract import ACCOUNT_TYPES, LABEL_COLUMNS, PAYMENT_RAILS, TABLES, Kind
from .rows import counts, response_head

# The keys of each map read_world prints, from the export contract. A key the world has no row
# for is not printed, and reads 0 here, so every world has the same keys; a key outside these
# is refused, such as an account_type or payment_rail value outside the contract's.
VERTEX_TYPES = tuple(table.name for table in TABLES if table.kind is Kind.VERTEX)
# The 15 forward edge types; a reverse type is its forward type read from the other end.
EDGE_TYPES = tuple(table.name for table in TABLES if table.kind is not Kind.VERTEX)
PAYMENT_TYPES = ("Payment_Transaction", "Zelle_Transfer")
# The times whose second sums are printed, as "<type>.<attribute>": the dates of record and the
# payments' event times, then each link type's start and end. No label column is among them.
SECONDS = (
    *(
        f"{table.name}.{column}"
        for table in TABLES
        if table.kind is Kind.VERTEX
        for column in table.column_names
        if column.endswith("_ms") and column not in LABEL_COLUMNS
    ),
    *(
        f"{table.name}.{column}"
        for table in TABLES
        if table.kind is Kind.LINK
        for column in ("valid_from_ms", "valid_to_ms")
    ),
)
KEYS: dict[str, tuple[str, ...]] = {
    "vertices": VERTEX_TYPES,
    "edges": EDGE_TYPES,
    "account_types": ACCOUNT_TYPES,
    "payment_rails": PAYMENT_RAILS,
    "seconds": SECONDS,
    "cents": PAYMENT_TYPES,
    "squared_degrees": EDGE_TYPES,
    "squared_target_degrees": EDGE_TYPES,
    "degree_products": EDGE_TYPES,
}


@dataclass(frozen=True)
class World:
    """What identifies a loaded world: the count of each vertex type, forward edge type,
    account_type and payment_rail value, and of external accounts; the sums in whole seconds
    of the dates of record, link times and event times, keyed "<type>.<attribute>"; the sums
    of the amounts in cents per payment type; per forward edge type the sums of its sources'
    and its targets' squared degrees and of the products of both ends' degrees over its edges;
    and the first joined_ms and the first and last payment's event_ts_ms."""

    vertices: dict[str, int]
    edges: dict[str, int]
    account_types: dict[str, int]
    payment_rails: dict[str, int]
    external_accounts: int
    seconds: dict[str, int]
    cents: dict[str, int]
    squared_degrees: dict[str, int]
    squared_target_degrees: dict[str, int]
    degree_products: dict[str, int]
    first_joined_ms: int
    first_event_ms: int
    last_event_ms: int


def params() -> dict[str, Any]:
    """read_world takes no parameter."""
    return {}


def parse(rows: Sequence[object]) -> World:
    """The loaded world's identity fields."""
    from . import READ_WORLD  # the registry imports this module

    head = response_head(rows, READ_WORLD)
    names = [field.name for field in fields(World)]
    if set(head) != {"status", *names}:
        raise ValueError(f"{READ_WORLD} printed the fields {sorted(head)}")
    values: dict[str, Any] = {}
    for name in names:
        value = head[name]
        if name not in KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{READ_WORLD} printed {name} {value!r}, not a count or time")
            values[name] = value
            continue
        found = counts(value, name, READ_WORLD)
        unknown = sorted(set(found) - set(KEYS[name]))
        if unknown:
            raise ValueError(f"{READ_WORLD} printed {name} of {unknown}, outside the contract")
        values[name] = {key: found.get(key, 0) for key in KEYS[name]}
    return World(**values)
