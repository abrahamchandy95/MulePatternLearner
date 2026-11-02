"""list_accounts: the scorable accounts at one time t with their owner, a page at a time by
strictly increasing id (docs/design.md, "The scored population" and "list_accounts")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .rows import after, as_of, page_size, paged, response_head


@dataclass(frozen=True)
class ScorableAccount:
    """An internal deposit account open at t with a visible ownership link, and its owner. No
    date of record is printed: one in every cache is one edit away from a feature."""

    account_id: str
    owner_id: str


def params(t_ms: int, after_id: str, page: int) -> dict[str, Any]:
    """The parameters of one page: the accounts after after_id, at most page of them."""
    return {"t_ms": as_of(t_ms), "after_id": after(after_id), "page": page_size(page)}


def parse(rows: Sequence[object], after_id: str, page: int) -> tuple[ScorableAccount, ...]:
    """The accounts of the page that params(t_ms, after_id, page) asked for, each with its
    owner."""
    from . import LIST_ACCOUNTS  # the registry imports this module

    response_head(rows, LIST_ACCOUNTS)
    accounts = paged(ScorableAccount, rows, LIST_ACCOUNTS, after_id, page)
    for account in accounts:
        if not account.owner_id:
            raise ValueError(f"{LIST_ACCOUNTS} answered {account.account_id} without an owner")
    return accounts
