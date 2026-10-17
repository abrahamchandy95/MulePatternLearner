"""The query registry and the parser of response rows, on rows written by hand.

No query is registered until the query steps add them, so the registry is checked against
the files it names, whatever they are, and parse_row on a dataclass of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph.installer import repository_queries
from mule_pattern_learner.tigergraph.queries import GRAPH_NAME, QUERY_FILES, parse_row


@dataclass(frozen=True)
class Row:
    account_id: str
    is_external: bool
    payments: int
    gap_ms: int
    amount: float


ROW: dict[str, Any] = {
    "account_id": "a_01",
    "is_external": False,
    "payments": 3,
    "gap_ms": -5,
    "amount": 12,
}


def test_the_queries_are_for_the_graph_the_schema_creates() -> None:
    schema = (GSQL_DIR / "schema" / "schema.gsql").read_text()
    assert f"CREATE GRAPH {GRAPH_NAME} ()" in schema


def test_every_registered_query_has_its_file_under_queries_or_evaluation() -> None:
    for name, relative in QUERY_FILES.items():
        assert relative in (f"queries/{name}.gsql", f"evaluation/{name}.gsql"), name
    assert list(repository_queries()) == list(QUERY_FILES)


def test_a_row_parses_into_its_dataclass() -> None:
    row = parse_row(Row, ROW, signed=frozenset({"gap_ms"}))
    assert row == Row("a_01", False, 3, -5, 12.0)
    assert isinstance(row.amount, float)


# Changes to a row, and what the refusal names.
MALFORMED: list[tuple[Callable[[dict[str, Any]], object], str]] = [
    (lambda row: row.pop("amount"), "fields"),
    (lambda row: row.update(colour="red"), "fields"),
    (lambda row: row.update(account_id=7), "Row.account_id"),
    (lambda row: row.update(is_external=0), "Row.is_external"),
    (lambda row: row.update(payments=True), "Row.payments"),
    (lambda row: row.update(payments=-1), "Row.payments"),
    (lambda row: row.update(payments=2.0), "Row.payments"),
    (lambda row: row.update(amount="12"), "Row.amount"),
    (lambda row: row.update(amount=False), "Row.amount"),
]


@pytest.mark.parametrize(("change", "message"), MALFORMED)
def test_a_malformed_row_is_refused(
    change: Callable[[dict[str, Any]], object], message: str
) -> None:
    row = dict(ROW)
    change(row)
    with pytest.raises(ValueError, match=message):
        parse_row(Row, row, signed=frozenset({"gap_ms"}))


def test_a_negative_integer_is_refused_unless_its_field_is_signed() -> None:
    with pytest.raises(ValueError, match="Row.gap_ms"):
        parse_row(Row, ROW)
    with pytest.raises(ValueError, match="not an object"):
        parse_row(Row, [ROW])
