"""What every query's parser shares, on rows written by hand: a response's head with its
status and version, a refusal, a row parsed into its dataclass, a printed list of ids and a
set of vertices as a parameter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries.rows import (
    QueryRefused,
    ids,
    parse_row,
    response_head,
    vertex_set,
)


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
VERSION = "read_example-0123456789ab"


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


def test_the_head_of_an_answer_says_it_ran_and_its_version() -> None:
    head = {"status": "ok", "version": VERSION, "received": 3}
    assert response_head([head, {"accounts": []}], "read_example", VERSION) == head


def test_a_refusal_raises_with_its_status() -> None:
    refusal = {"status": "invisible_account", "accounts": ["a_02"]}
    with pytest.raises(QueryRefused) as refused:
        response_head([refusal], "read_example", VERSION)
    assert (refused.value.query, refused.value.status) == ("read_example", "invisible_account")
    assert refused.value.row == refusal
    assert "read_example refused the request: invisible_account" in str(refused.value)
    # A refusal is a ValueError, which the executor never retries.
    assert isinstance(refused.value, ValueError)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "without a status first"),
        ([["ok"]], "without a status first"),
        ([{"version": VERSION}], "without a status first"),
        ([{"status": "ok", "version": "read_example-ba9876543210"}], "run `mule install`"),
        ([{"status": "ok"}], "is version None"),
    ],
)
def test_an_answer_without_a_status_or_of_another_version_is_refused(
    rows: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        response_head(rows, "read_example", VERSION)


def test_printed_ids_are_distinct_and_sorted() -> None:
    assert ids(["b", "a"], "hubs") == ("a", "b")
    for value in (["a", "a"], ["a", 1], "a", None):
        with pytest.raises(ValueError, match="hubs"):
            ids(value, "hubs")


def test_a_vertex_set_parameter_holds_each_id_once() -> None:
    assert vertex_set(["a_01", "a_02"]) == [{"id": "a_01"}, {"id": "a_02"}]
    assert vertex_set([]) == []
    with pytest.raises(ValueError, match="repeats"):
        vertex_set(["a_01", "a_01"])
    with pytest.raises(ValueError, match="holds ids"):
        vertex_set(["a_01", ""])
    with pytest.raises(ValueError, match="holds ids"):
        vertex_set(["a_01", 2])
