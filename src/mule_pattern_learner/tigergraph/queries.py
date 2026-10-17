"""The installed queries: the graph's name, every query's file, and the parser of a response
row into a plain dataclass.

The queries of docs/design.md ("The queries") are added one step at a time, each with its
row dataclasses; until then no query is installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

GRAPH_NAME = "Mule_Pattern_Learner"

# Every query the code installs, and its file under gsql/.
QUERY_FILES: dict[str, str] = {}


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
