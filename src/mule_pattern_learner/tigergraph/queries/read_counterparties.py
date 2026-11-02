"""read_counterparties, B2: who pays it and whom it pays. For each account of a request at one
time t, how many accounts paid it and were paid by it, how new they are, and where its money
went (docs/design.md, "read_counterparties (B2): who pays it and whom it pays", and
docs/features.md, "Account node", columns 22 to 33)."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_CONFIG
from .rows import account_ids, as_of, parse_row, projected, response_head, vertex_set

# The most first-payment times max_new_payers_24h reads: the newest of the time-ordered cap.
RECENT = DEFAULT_CONFIG.subgraph.recent_payments


@dataclass(frozen=True)
class Counterparties:
    """An account's counterparty columns at t (docs/features.md, "Account node", columns 22
    to 33).

    - payers_30d, payees_30d: the distinct accounts that paid it, and that it paid, in the
      last 30 days.
    - new_internal_payers_*, new_external_payers_*: the distinct payers, of this bank
      (is_external 0) or of another, whose earliest payment to it in the 90-day lookback is
      in the last 7 or 30 days; new_payees_*, the same for the accounts it paid.
    - new_payer_share_30d: the share of the cents received in the last 30 days that came from
      payers new over 30 days.
    - max_new_payers_24h: the most first-payment times of payers new over 30 days in any 24
      hours starting at one of them, over the newest 256 kept by the whole-second rule.
    - top_payee_share_30d, hub_payee_share_30d: the shares of the cents sent in the last 30
      days that went to its largest payee, and to account hubs.

    Every share is 0 when its denominator is: payers_30d shows nothing received, payees_30d
    nothing sent.
    """

    account_id: str
    payers_30d: int
    payees_30d: int
    new_internal_payers_7d: int
    new_internal_payers_30d: int
    new_external_payers_7d: int
    new_external_payers_30d: int
    new_payees_7d: int
    new_payees_30d: int
    new_payer_share_30d: float
    max_new_payers_24h: int
    top_payee_share_30d: float
    hub_payee_share_30d: float


SHARES = ("new_payer_share_30d", "top_payee_share_30d", "hub_payee_share_30d")


def params(t_ms: int, accounts: Iterable[str], hubs: Iterable[str]) -> dict[str, Any]:
    """The parameters of a request: the time t, the accounts, at least one, each once, and
    the account hubs of t, read_hubs' accounts of the same t, each once."""
    return {
        "t_ms": as_of(t_ms),
        "account_ids": account_ids(accounts),
        "hub_accounts": vertex_set(hubs),
    }


def _check(row: Counterparties) -> None:
    """Refuse a row whose values the query cannot give."""
    from . import READ_COUNTERPARTIES  # the registry imports this module

    def refuse(why: str) -> ValueError:
        return ValueError(f"{READ_COUNTERPARTIES} answered {row.account_id} with {why}: {row}")

    for kind in ("internal_payers", "external_payers", "payees"):
        if getattr(row, f"new_{kind}_7d") > getattr(row, f"new_{kind}_30d"):
            raise refuse(f"more new {kind.replace('_', ' ')} over 7 days than over 30")
    new_payers = row.new_internal_payers_30d + row.new_external_payers_30d
    if new_payers > row.payers_30d:
        raise refuse("more new payers than payers over 30 days")
    if row.new_payees_30d > row.payees_30d:
        raise refuse("more new payees than payees over 30 days")
    if not all(0.0 <= getattr(row, name) <= 1.0 for name in SHARES):
        raise refuse("a share outside [0, 1]")
    if new_payers == 0 and row.new_payer_share_30d:
        raise refuse("a share received from new payers without one")
    if row.payees_30d == 0 and (row.top_payee_share_30d or row.hub_payee_share_30d):
        raise refuse("shares of cents sent without a payee")
    if row.max_new_payers_24h > min(new_payers, RECENT):
        raise refuse("more new payers in 24 hours than it kept")
    if 0 < new_payers <= RECENT and row.max_new_payers_24h == 0:
        raise refuse("no new payer in 24 hours though it kept some")


def parse(rows: Sequence[object], accounts: Collection[str]) -> tuple[Counterparties, ...]:
    """The answer to the request params(t_ms, accounts, hubs) made: a row for every account
    asked for and no other, sorted by id."""
    from . import READ_COUNTERPARTIES  # the registry imports this module

    response_head(rows, READ_COUNTERPARTIES)
    found = [
        parse_row(Counterparties, row) for row in projected(rows, "accounts", READ_COUNTERPARTIES)
    ]
    asked = set(accounts)
    answered = [row.account_id for row in found]
    if len(set(answered)) != len(answered) or set(answered) != asked:
        raise ValueError(
            f"{READ_COUNTERPARTIES} answered the accounts {sorted(answered)}, not {sorted(asked)}"
        )
    for row in found:
        _check(row)
    return tuple(sorted(found, key=lambda row: row.account_id))
