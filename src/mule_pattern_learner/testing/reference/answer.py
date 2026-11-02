"""How the reference graph prints an answer as TigerGraph prints it, and the checks of the
parameters that queries share (docs/design.md, "Query plumbing").

- An answer starts with a status row holding the query's version; a refusal is the status row
  the query prints, alone.
- A PRINT of a vertex set's projection is a list of vertices, each with its projected fields
  under "attributes".
- A query with t takes a whole second after the lookback's length; a paged query takes
  1 to PAGE_LIMIT accounts a page.
- A per-account query takes its accounts as a list of ids, at least one, each once, each in
  the graph and existing at t, and refuses any other request with a status row. A query that
  takes the account hubs of t takes them as a SET<VERTEX<Account>>, from which TigerGraph
  drops an id the graph lacks without a word, and refuses a hub that does not exist at t.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ...tigergraph.queries.rows import LOOKBACK_MS, PAGE_LIMIT

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from .graph import ReferenceGraph

OK = "ok"
INVALID_PARAMETERS = "invalid_parameters"
DUPLICATE_ACCOUNT = "duplicate_account"
UNKNOWN_ACCOUNT = "unknown_account"
INVISIBLE_ACCOUNT = "invisible_account"
INVISIBLE_HUB = "invisible_hub"


def head(name: str) -> dict[str, Any]:
    """The status row of an answer: the query ran."""
    del name
    return {"status": OK}


def refused(status: str, **fields: Any) -> list[dict[str, Any]]:
    """A refusal: the status row alone, with its fields."""
    return [{"status": status, **fields}]


def vertices(kind: str, rows: Iterable[DataclassInstance]) -> list[dict[str, Any]]:
    """A PRINT of a vertex set's projection: each vertex, keyed by its row's first field, the
    vertex's id, with the row as its attributes."""
    printed: list[dict[str, Any]] = []
    for row in rows:
        attributes = asdict(row)
        vertex_id = next(iter(attributes.values()))
        printed.append({"v_id": vertex_id, "v_type": kind, "attributes": attributes})
    return printed


def valid_time(t_ms: int) -> bool:
    """t is a whole second after the lookback's length, which every query with t requires."""
    return t_ms > LOOKBACK_MS and t_ms % 1000 == 0


def valid_page(size: int) -> bool:
    """A page holds 1 to PAGE_LIMIT accounts."""
    return 1 <= size <= PAGE_LIMIT


def request_refusal(graph: ReferenceGraph, t_ms: int, account_ids: Sequence[str]) -> str | None:
    """Why a per-account query refuses a request, in the order the queries check it, or None
    when it answers: t no whole second after the lookback or no account, an account named twice,
    an account the graph lacks, an account that does not exist at t."""
    if not valid_time(t_ms) or not account_ids:
        return INVALID_PARAMETERS
    if len(set(account_ids)) != len(account_ids):
        return DUPLICATE_ACCOUNT
    if any(account_id not in graph.accounts for account_id in account_ids):
        return UNKNOWN_ACCOUNT
    if any(graph.accounts[account_id].opened_ms >= t_ms for account_id in account_ids):
        return INVISIBLE_ACCOUNT
    return None


def hub_set(
    graph: ReferenceGraph, t_ms: int, hub_accounts: Iterable[Mapping[str, str]]
) -> frozenset[str] | None:
    """The account hubs a SET<VERTEX<Account>> parameter names that the graph holds, since
    TigerGraph drops the others without a word; None when one of them does not exist at t,
    which the query refuses as invisible_hub."""
    named = {vertex["id"] for vertex in hub_accounts}
    found = frozenset(account_id for account_id in named if account_id in graph.accounts)
    if any(graph.accounts[account_id].opened_ms >= t_ms for account_id in found):
        return None
    return found
