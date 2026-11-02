"""What every query's parser shares: a response's head with its status and version, a
refusal, a row parsed into a plain dataclass, what a PRINT printed, a page of accounts in id
order, a printed list of ids, and the parameters every query checks alike: t, a page's size, a
set of vertices and a request's accounts.

It imports nothing of the package but the settings, so a query module may import it while the
registry imports that module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import Field, fields
from typing import TYPE_CHECKING, Any

from ...config import DEFAULT_CONFIG

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

OK = "ok"
# The types a row's field may have, by name.
FIELD_TYPES: dict[str, type] = {"str": str, "bool": bool, "int": int, "float": float}
# A query with t takes any whole second, in epoch milliseconds, after the lookback's length, so
# that t - lookback never goes below 0 (docs/design.md, "Time"). The queries refuse any other t,
# and so do their builders.
DAY_MS = 86_400_000
LOOKBACK_MS = DEFAULT_CONFIG.subgraph.lookback_days * DAY_MS
# The most accounts a page of list_accounts, read_labels or read_answer_key holds, which they
# page by strictly increasing id.
PAGE_LIMIT = 10_000


class QueryRefused(ValueError):
    """A query answered with a status row other than "ok": it refused its parameters, such as
    an invalid setting or a duplicate, unknown or invisible account (docs/design.md, "Query
    plumbing"). status is its word for why, and row the whole row."""

    def __init__(self, query: str, status: str, row: Mapping[str, Any]) -> None:
        super().__init__(f"{query} refused the request: {status} ({dict(row)})")
        self.query = query
        self.status = status
        self.row = dict(row)


def response_head(rows: Sequence[object], query: str) -> dict[str, Any]:
    """The first row of a query's answer, once it says the query ran.

    A status other than "ok" raises QueryRefused. No query prints its version: every command
    checks that the installed queries are the repository's before it runs (installer.verify).
    """
    head = rows[0] if rows else None
    if not isinstance(head, Mapping) or "status" not in head:
        raise ValueError(f"{query} answered without a status first: {head!r}")
    head = {str(key): value for key, value in head.items()}
    status = head["status"]
    if status != OK:
        raise QueryRefused(query, str(status), head)
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
        of = _field_type(field)
        if of is None:
            raise ValueError(
                f"{name}.{field.name} has the type {field.type!r}, "
                f"not one of {', '.join(FIELD_TYPES)}"
            )
        ok = {
            str: isinstance(value, str),
            bool: isinstance(value, bool),
            int: isinstance(value, int)
            and not isinstance(value, bool)
            and (value >= 0 or field.name in signed),
            float: isinstance(value, int | float) and not isinstance(value, bool),
        }[of]
        if not ok:
            raise ValueError(f"{name}.{field.name} is {value!r}, not a valid {of.__name__}")
        values[field.name] = float(value) if of is float else value
    return kind(**values)


def _field_type(field: Field[Any]) -> type | None:
    """A row field's type: its class, or its name where annotations are postponed (`from
    __future__ import annotations`); None for any other type, such as `int | None`."""
    declared: object = field.type
    if isinstance(declared, str):
        return FIELD_TYPES.get(declared)
    return next((kind for kind in FIELD_TYPES.values() if declared is kind), None)


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


def account_ids(accounts: Iterable[object]) -> list[str]:
    """The accounts of a per-account query's request, as the LIST<STRING> parameter it takes:
    at least one, each a distinct id.

    An installed query drops an id the graph lacks from a SET<VERTEX> parameter without a
    word, so a per-account query takes its accounts as ids and refuses a duplicate or unknown
    one itself, with a status row (docs/design.md, "Query plumbing").
    """
    found = list(accounts)
    given = [item for item in found if isinstance(item, str) and item]
    if len(given) != len(found):
        raise ValueError(f"A request holds account ids, got {found!r}")
    if not given:
        raise ValueError("A request holds at least one account")
    if len(set(given)) != len(given):
        raise ValueError("A request repeats an account")
    return given


def as_of(t_ms: object) -> int:
    """t, when it is a whole second after the lookback's length."""
    if (
        isinstance(t_ms, bool)
        or not isinstance(t_ms, int)
        or t_ms <= LOOKBACK_MS
        or t_ms % 1000 != 0
    ):
        raise ValueError(
            f"t must be a whole second in epoch milliseconds after the lookback, got {t_ms!r}"
        )
    return t_ms


def after(after_id: object) -> str:
    """after_id, when it is an account id or the empty string, which starts the first page."""
    if not isinstance(after_id, str):
        raise ValueError(f"after_id is an account id or the empty string, got {after_id!r}")
    return after_id


def page_size(page: object) -> int:
    """page, when it is a number of accounts from 1 to PAGE_LIMIT."""
    if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= PAGE_LIMIT:
        raise ValueError(f"A page holds 1 to {PAGE_LIMIT} accounts, got {page!r}")
    return page


def printed(rows: Sequence[object], key: str, query: str) -> object:
    """The value a query printed under key, from the one row of its answer that holds it."""
    found = [row[key] for row in rows if isinstance(row, Mapping) and key in row]
    if len(found) != 1:
        raise ValueError(f"{query} printed {key} {len(found)} times, not once")
    return found[0]


def projected(rows: Sequence[object], key: str, query: str) -> list[Mapping[str, Any]]:
    """The rows of a PRINT of a vertex set's projection under key: the fields projected for
    each vertex, which TigerGraph prints under "attributes" beside its id and type."""
    value = printed(rows, key, query)
    if not isinstance(value, list):
        raise ValueError(f"{query} printed {key} as {value!r}, not a list of vertices")
    found: list[Mapping[str, Any]] = []
    for item in value:
        attributes = item.get("attributes") if isinstance(item, Mapping) else None
        if not isinstance(attributes, Mapping):
            raise ValueError(f"{query} printed in {key} {item!r}, not a vertex's attributes")
        found.append(attributes)
    return found


def paged[T: DataclassInstance](
    kind: type[T],
    rows: Sequence[object],
    query: str,
    after_id: str,
    page: int,
    *,
    signed: frozenset[str] = frozenset(),
) -> tuple[T, ...]:
    """The accounts of a page of a paged query, printed under "accounts", as kind, whose first
    field is account_id: at most page of them, each id after after_id and after the one before
    it, as the query's strictly increasing order gives them."""
    found = tuple(parse_row(kind, row, signed=signed) for row in projected(rows, "accounts", query))
    if len(found) > page:
        raise ValueError(f"{query} answered {len(found)} accounts to a page of {page}")
    previous = after_id
    for row in found:
        account: object = getattr(row, "account_id")
        if not isinstance(account, str) or account <= previous:
            raise ValueError(
                f"{query} answered {account!r} after {previous!r}, not in strictly increasing id "
                "order"
            )
        previous = account
    return found


def counts(value: object, what: str, query: str) -> dict[str, int]:
    """A printed map of names to integers that are not negative, such as a count per type."""
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for key, count in value.items()
    ):
        raise ValueError(f"{query} printed {what} {value!r}, not counts by name")
    found: Mapping[str, int] = value
    return dict(found)
