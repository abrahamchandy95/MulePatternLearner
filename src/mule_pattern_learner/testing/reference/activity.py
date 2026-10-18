"""The reference answer of read_activity_change, B3: the account against its own past
(docs/design.md, "read_activity_change (B3): the account against its own past", and
docs/features.md, "Account node", columns 34 to 46), written from the features apart from the
GSQL.

- The counts read every payment received or sent in [t - w, t); a self-payment is read
  nowhere.
- The bursts and the shortest gap read the payments received, or sent, in the last 30 days,
  each list cut on its own to its newest RECENT by the whole-second rule. A burst is the most
  payments in [x, x + w) with x one of their times, so payments of one time all count; the
  shortest gap is the smallest x - x' over kept inflows, x' the latest kept inflow strictly
  before x.
- The time since the last payment each way reads the whole lookback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...tigergraph.queries import READ_ACTIVITY_CHANGE
from ...tigergraph.queries.read_activity_change import ActivityChange
from ...tigergraph.queries.rows import DAY_MS
from .answer import head, refused, request_refusal, vertices
from .graph import kept_times, most_within

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .graph import AsOf, Payment, ReferenceGraph

# The days of the bursts' lists, and the shorter burst window.
BURST_DAYS = 30
HOUR_MS = 3_600_000


def _shortest_gap(times: Sequence[int]) -> int | None:
    """The smallest gap from a time to the latest time strictly before it, which is the
    smallest difference of two distinct times; None with fewer than two distinct times."""
    distinct = sorted(set(times))
    return min((b - a for a, b in zip(distinct, distinct[1:])), default=None)


def activity_change(view: AsOf, account_id: str) -> ActivityChange:
    """The account's activity row at t."""
    received, sent = view.received(account_id), view.sent(account_id)

    def count(payments: Sequence[Payment], days: int) -> int:
        return sum(1 for p in payments if p.ts >= view.t - days * DAY_MS)

    def kept(payments: Sequence[Payment]) -> list[int]:
        return kept_times([p.ts for p in payments if p.ts >= view.t - BURST_DAYS * DAY_MS])

    kept_in, kept_out = kept(received), kept(sent)
    gap = _shortest_gap(kept_in)
    last_in = max((p.ts for p in received), default=None)
    last_out = max((p.ts for p in sent), default=None)
    return ActivityChange(
        account_id=account_id,
        in_7d=count(received, 7),
        in_30d=count(received, 30),
        in_90d=count(received, 90),
        out_7d=count(sent, 7),
        out_30d=count(sent, 30),
        out_90d=count(sent, 90),
        burst_in_1h=most_within(kept_in, HOUR_MS),
        burst_in_24h=most_within(kept_in, DAY_MS),
        burst_out_1h=most_within(kept_out, HOUR_MS),
        burst_out_24h=most_within(kept_out, DAY_MS),
        shortest_in_gap_present=gap is not None,
        shortest_in_gap_ms=0 if gap is None else gap,
        since_last_in_present=last_in is not None,
        since_last_in_ms=0 if last_in is None else view.t - last_in,
        since_last_out_present=last_out is not None,
        since_last_out_ms=0 if last_out is None else view.t - last_out,
    )


def read_activity_change(
    graph: ReferenceGraph, t_ms: int, account_ids: list[str]
) -> list[dict[str, Any]]:
    """The activity rows of the request's accounts at t, sorted by id."""
    status = request_refusal(graph, t_ms, account_ids)
    if status is not None:
        return refused(status)
    view = graph.as_of(t_ms)
    rows = [activity_change(view, account_id) for account_id in sorted(account_ids)]
    return [head(READ_ACTIVITY_CHANGE), {"accounts": vertices("Account", rows)}]
