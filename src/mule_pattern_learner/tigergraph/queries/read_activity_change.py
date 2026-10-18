"""read_activity_change, B3: the account against its own past. For each account of a request
at one Monday, how many payments it received and sent over the last 7, 30 and 90 days, its
busiest hour and day of the last 30 days each way, the shortest gap between its inflows, and
how long it has been quiet each way (docs/design.md, "read_activity_change (B3): the account
against its own past", and docs/features.md, "Account node", columns 34 to 46)."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_CONFIG
from .rows import DAY_MS, LOOKBACK_MS, account_ids, as_of, parse_row, projected, response_head

# The most payments a burst reads each way: the newest of the time-ordered cap.
RECENT = DEFAULT_CONFIG.subgraph.recent_payments
# The windows of the counts, in milliseconds.
WINDOW_7D_MS = 7 * DAY_MS
WINDOW_30D_MS = 30 * DAY_MS


@dataclass(frozen=True)
class ActivityChange:
    """An account's activity columns at t (docs/features.md, "Account node", columns 34 to
    46).

    - in_7d, in_30d, in_90d, out_7d, out_30d, out_90d: the payments it received and sent in
      the last 7, 30 and 90 days.
    - burst_in_1h, burst_in_24h, burst_out_1h, burst_out_24h: over its newest 256 payments
      received, or sent, in the last 30 days, kept by the whole-second rule, the most in an
      hour and in 24 hours from one of them.
    - shortest_in_gap: over the same kept inflows, the smallest gap from one to the latest
      strictly before it, as presence and milliseconds; absent when no kept inflow has a
      strictly earlier one.
    - since_last_in, since_last_out: t minus its latest payment received, or sent, in the
      90-day lookback, as presence and milliseconds; absent with none.
    """

    account_id: str
    in_7d: int
    in_30d: int
    in_90d: int
    out_7d: int
    out_30d: int
    out_90d: int
    burst_in_1h: int
    burst_in_24h: int
    burst_out_1h: int
    burst_out_24h: int
    shortest_in_gap_present: bool
    shortest_in_gap_ms: int
    since_last_in_present: bool
    since_last_in_ms: int
    since_last_out_present: bool
    since_last_out_ms: int


def params(t_ms: int, accounts: Iterable[str]) -> dict[str, Any]:
    """The parameters of a request: the Monday t and the accounts, at least one, each once."""
    return {"t_ms": as_of(t_ms), "account_ids": account_ids(accounts)}


def _check(row: ActivityChange) -> None:
    """Refuse a row whose values the query cannot give."""
    from . import READ_ACTIVITY_CHANGE  # the registry imports this module

    def refuse(why: str) -> ValueError:
        return ValueError(f"{READ_ACTIVITY_CHANGE} answered {row.account_id} with {why}: {row}")

    for way in ("in", "out"):
        counts = [getattr(row, f"{way}_{days}d") for days in (7, 30, 90)]
        if counts != sorted(counts):
            raise refuse(f"{way} counts that shrink over a longer window")
        recent = counts[1]
        hour, day = getattr(row, f"burst_{way}_1h"), getattr(row, f"burst_{way}_24h")
        if not hour <= day <= min(recent, RECENT):
            raise refuse(f"{way} bursts beyond each other or the payments kept")
        if 0 < recent <= RECENT and hour == 0:
            raise refuse(f"no {way} burst though it kept payments")
        present = getattr(row, f"since_last_{way}_present")
        since = getattr(row, f"since_last_{way}_ms")
        if present != (counts[2] > 0):
            raise refuse(f"a time since the last payment {way} unlike its count over 90 days")
        if not present and since:
            raise refuse(f"an absent time since the last payment {way} that is not 0")
        if present and not (
            0 < since <= LOOKBACK_MS
            and (since <= WINDOW_7D_MS) == (counts[0] > 0)
            and (since <= WINDOW_30D_MS) == (recent > 0)
        ):
            raise refuse(f"a time since the last payment {way} unlike its counts")
    if row.shortest_in_gap_present and not (
        row.in_30d >= 2 and 0 < row.shortest_in_gap_ms < WINDOW_30D_MS
    ):
        raise refuse("a gap between inflows it cannot have")
    if not row.shortest_in_gap_present and row.shortest_in_gap_ms:
        raise refuse("an absent gap between inflows that is not 0")


def parse(rows: Sequence[object], accounts: Collection[str]) -> tuple[ActivityChange, ...]:
    """The answer to the request params(t_ms, accounts) made: a row for every account asked
    for and no other, sorted by id."""
    from . import READ_ACTIVITY_CHANGE, VERSIONS  # the registry imports this module

    response_head(rows, READ_ACTIVITY_CHANGE, VERSIONS[READ_ACTIVITY_CHANGE])
    found = [
        parse_row(ActivityChange, row) for row in projected(rows, "accounts", READ_ACTIVITY_CHANGE)
    ]
    asked = set(accounts)
    answered = [row.account_id for row in found]
    if len(set(answered)) != len(answered) or set(answered) != asked:
        raise ValueError(
            f"{READ_ACTIVITY_CHANGE} answered the accounts {sorted(answered)}, not {sorted(asked)}"
        )
    for row in found:
        _check(row)
    return tuple(sorted(found, key=lambda row: row.account_id))
