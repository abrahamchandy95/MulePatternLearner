"""What every query's parser shares: a response's head with its status and version, a
refusal, a row parsed into a plain dataclass, a printed list of ids, and a set of vertices as a
parameter.

It imports nothing else of the package, so a query module may import it while the registry
imports that module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

OK = "ok"


class QueryRefused(ValueError):
    """A query answered with a status row other than "ok": it refused its parameters, such as
    an invalid setting or a duplicate, unknown or invisible account (docs/design.md, "Query
    plumbing"). status is its word for why, and row the whole row."""

    def __init__(self, query: str, status: str, row: Mapping[str, Any]) -> None:
        super().__init__(f"{query} refused the request: {status} ({dict(row)})")
        self.query = query
        self.status = status
        self.row = dict(row)


def response_head(rows: Sequence[object], query: str, version: str) -> dict[str, Any]:
    """The first row of a query's answer, once it says the query ran and is of version.

    A status other than "ok" raises QueryRefused. An answer of another version is refused
    too: the installed query is not the one this code reads, so install the queries again.
    """
    head = rows[0] if rows else None
    if not isinstance(head, Mapping) or "status" not in head:
        raise ValueError(f"{query} answered without a status first: {head!r}")
    head = {str(key): value for key, value in head.items()}
    status = head["status"]
    if status != OK:
        raise QueryRefused(query, str(status), head)
    if head.get("version") != version:
        raise ValueError(
            f"{query} is version {head.get('version')!r}, not {version}: run `mule install`"
        )
    return head


def parse_row[T: DataclassInstance](
    kind: type[T], row: object, *, signed: frozenset[str] = frozenset()
) -> T:
    """A response row as the dataclass kind, whose fields name the row's keys exactly.

    A value must have its field's type (a float field also takes an integer), and an
    integer must not be negative unless its field is in signed.
    """
    name = kind.__name__
    if not isinstance(row, Mapping):
        raise ValueError(f"A {name} row is {row!r}, not an object")
    expected = [field.name for field in fields(kind)]
    if set(row) != set(expected):
        raise ValueError(f"A {name} row has the fields {sorted(row)}, not {sorted(expected)}")
    values: dict[str, Any] = {}
    for field in fields(kind):
        value = row[field.name]
        ok = {
            "str": isinstance(value, str),
            "bool": isinstance(value, bool),
            "int": isinstance(value, int)
            and not isinstance(value, bool)
            and (value >= 0 or field.name in signed),
            "float": isinstance(value, int | float) and not isinstance(value, bool),
        }[str(field.type)]
        if not ok:
            raise ValueError(f"{name}.{field.name} is {value!r}, not a valid {field.type}")
        values[field.name] = float(value) if field.type == "float" else value
    return kind(**values)


def ids(value: object, what: str) -> tuple[str, ...]:
    """A printed list of distinct ids, sorted."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{what} is {value!r}, not a list of ids")
    found: list[str] = value
    if len(set(found)) != len(found):
        raise ValueError(f"{what} repeats an id")
    return tuple(sorted(found))


def vertex_set(vertex_ids: Iterable[object]) -> list[dict[str, str]]:
    """A SET<VERTEX<T>> parameter, as TigerGraph's JSON takes it: each id once."""
    items = list(vertex_ids)
    given = [item for item in items if isinstance(item, str) and item]
    if len(given) != len(items):
        raise ValueError(f"A vertex set holds ids, got {items!r}")
    if len(set(given)) != len(given):
        raise ValueError("A vertex set repeats an id")
    return [{"id": vertex_id} for vertex_id in given]
