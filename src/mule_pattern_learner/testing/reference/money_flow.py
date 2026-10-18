"""The reference answer of read_money_flow, B1: velocity (docs/design.md, "read_money_flow
(B1): velocity", and B1's walk in docs/features.md, "Account node"). The walk is an explicit
stack of lots written from the features, an implementation apart from the GSQL's walk
without a stack, so parity tests the GSQL's algebra as well as its plumbing.

The walk, as the features define it:

- The lists: the payments received in [t - 30 days, t) with positive cents, and the payments
  sent then, each cut to its newest RECENT. When neither list is cut the span is the whole 30
  days; otherwise b is the latest second, over the cut lists, of the newest payment a list left
  out, and every payment of either list at or before the second b is dropped. Both lists are
  then complete inside the span.
- A lot is the inflows of one second inside the span; it comes before every outflow of its
  second. The outflows of one second are walked by the name of their rail, then by payment
  id.
- Last in, first out: an outflow of c cents spends from the newest lot with cents left, then
  the next, until c is spent or no lot is left; what is left of it spent money from before the
  span and is recorded nowhere. Each match is a piece: its lot, its outflow and rail, its
  cents, and its dwell, the outflow's second minus the lot's second.

A self-payment is read nowhere, and only payments before t exist, so an outflow at or after t
never matches. The walk's work counter, scan_steps, is printed as 0: the reference graph does
not mirror work counters, and parity ignores them (docs/design.md, "Query plumbing").
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ...tigergraph.queries import READ_MONEY_FLOW
from ...tigergraph.queries.read_money_flow import MoneyFlow, TrackedInflow
from ...tigergraph.queries.rows import DAY_MS
from .answer import head, refused, request_refusal, vertices
from .graph import RECENT, SECOND_MS, Payment, cut_second, second, share

if TYPE_CHECKING:
    from .graph import AsOf, ReferenceGraph

# The span of the walk's lists, in days.
WALK_DAYS = 30
# The dwell windows of left_15m to left_7d, in seconds, and a day, within which the rail
# shares and a lot's left_24h are read.
DAY_S = 86_400
WINDOWS_S = (900, 3_600, 21_600, DAY_S, 3 * DAY_S, 7 * DAY_S)
# The rails of an onward transfer; "unknown" counts in the rail shares' denominator only.
TRANSFER_RAILS = frozenset({"ach", "check", "internal", "zelle"})


@dataclass(frozen=True, slots=True)
class Lot:
    """The inflows of one second inside the span: the second, their ids and their cents."""

    second: int
    payments: tuple[str, ...]
    cents: int


@dataclass(frozen=True, slots=True)
class Piece:
    """The cents an outflow spent from a lot: the lot's index in the walk, the outflow's id
    and rail, the cents and the dwell in seconds."""

    lot: int
    outflow: str
    rail: str
    cents: int
    dwell: int


@dataclass(frozen=True)
class Walk:
    """The lots of the span, oldest first, and the pieces in the order the outflows took
    them."""

    lots: tuple[Lot, ...]
    pieces: tuple[Piece, ...]


def walk(received: Iterable[Payment], sent: Iterable[Payment], cap: int = RECENT) -> Walk:
    """The walk of an account's payments received and sent in the last 30 days, its lists cut
    at cap payments each."""
    inflows = [payment for payment in received if payment.cents > 0]
    outflows = list(sent)
    cuts = [cut_second([p.ts for p in found], cap) for found in (inflows, outflows)]
    cut = max((found for found in cuts if found is not None), default=None)
    if cut is not None:
        inflows = [payment for payment in inflows if second(payment.ts) > cut]
        outflows = [payment for payment in outflows if second(payment.ts) > cut]
    by_second: dict[int, list[Payment]] = defaultdict(list)
    for payment in inflows:
        by_second[second(payment.ts)].append(payment)
    lots = tuple(
        Lot(at, tuple(sorted(p.id for p in found)), sum(p.cents for p in found))
        for at, found in sorted(by_second.items())
    )
    # Each lot before the outflows of its second, which go by rail, then id.
    events: list[tuple[int, int, str, str, int]] = [
        (lot.second, 0, "", "", index) for index, lot in enumerate(lots)
    ]
    events += [(second(p.ts), 1, p.rail, p.id, index) for index, p in enumerate(outflows)]
    stack: list[list[int]] = []
    pieces: list[Piece] = []
    for at, kind, _, _, index in sorted(events):
        if kind == 0:
            stack.append([index, lots[index].cents])
            continue
        outflow = outflows[index]
        need = outflow.cents
        while need > 0 and stack:
            top = stack[-1]
            take = min(need, top[1])
            pieces.append(Piece(top[0], outflow.id, outflow.rail, take, at - lots[top[0]].second))
            top[1] -= take
            need -= take
            if top[1] == 0:
                stack.pop()
    return Walk(lots, tuple(pieces))


def _half(pieces: Sequence[Piece], whole: int) -> int | None:
    """The dwell, in seconds, of the piece at which the pieces, in the order given, first add
    up to half of whole; None when they never do."""
    gone = 0
    for piece in pieces:
        gone += piece.cents
        if 2 * gone >= whole:
            return piece.dwell
    return None


def money_flow(view: AsOf, account_id: str) -> tuple[MoneyFlow, list[TrackedInflow]]:
    """The account's velocity row at t and its tracked inflows."""
    received, sent = view.received(account_id), view.sent(account_id)

    def amount(payments: Sequence[Payment], days: int) -> int:
        return sum(p.cents for p in payments if p.ts >= view.t - days * DAY_MS)

    since = view.t - WALK_DAYS * DAY_MS
    found = walk([p for p in received if p.ts >= since], [p for p in sent if p.ts >= since])
    total = sum(lot.cents for lot in found.lots)
    matched = sum(piece.cents for piece in found.pieces)
    left = [share(sum(p.cents for p in found.pieces if p.dwell < w), total) for w in WINDOWS_S]
    median = _half(sorted(found.pieces, key=lambda piece: piece.dwell), total)
    day = [piece for piece in found.pieces if piece.dwell < DAY_S]
    day_cents = sum(p.cents for p in day)
    row = MoneyFlow(
        account_id=account_id,
        in_amount_7d=amount(received, 7),
        in_amount_30d=amount(received, 30),
        in_amount_90d=amount(received, 90),
        out_amount_7d=amount(sent, 7),
        out_amount_30d=amount(sent, 30),
        out_amount_90d=amount(sent, 90),
        flow_present=total > 0,
        left_15m=left[0],
        left_1h=left[1],
        left_6h=left[2],
        left_1d=left[3],
        left_3d=left[4],
        left_7d=left[5],
        held_share=share(total - matched, total),
        median_dwell_present=median is not None,
        median_dwell_ms=0 if median is None else median * SECOND_MS,
        left_by_transfer=share(sum(p.cents for p in day if p.rail in TRANSFER_RAILS), day_cents),
        left_by_card=share(sum(p.cents for p in day if p.rail == "card"), day_cents),
        left_by_cash=share(sum(p.cents for p in day if p.rail == "cash"), day_cents),
        scan_steps=0,
    )
    tracked: list[TrackedInflow] = []
    by_lot: dict[int, list[Piece]] = defaultdict(list)
    for piece in found.pieces:
        by_lot[piece.lot].append(piece)
    for index, lot in enumerate(found.lots):
        pieces = by_lot[index]
        left_24h = sum(p.cents for p in pieces if p.dwell < DAY_S) / lot.cents
        half = _half(pieces, lot.cents)
        for payment_id in lot.payments:
            tracked.append(
                TrackedInflow(
                    account_id,
                    payment_id,
                    left_24h,
                    half is not None,
                    0 if half is None else half * SECOND_MS,
                )
            )
    return row, tracked


def read_money_flow(
    graph: ReferenceGraph, t_ms: int, account_ids: list[str]
) -> list[dict[str, Any]]:
    """The velocity rows of the request's accounts at t, sorted by id, and every tracked
    inflow, sorted by account and payment id."""
    status = request_refusal(graph, t_ms, account_ids)
    if status is not None:
        return refused(status)
    view = graph.as_of(t_ms)
    rows: list[MoneyFlow] = []
    tracked: list[TrackedInflow] = []
    for account_id in sorted(account_ids):
        row, inflows = money_flow(view, account_id)
        rows.append(row)
        tracked.extend(sorted(inflows, key=lambda inflow: inflow.payment_id))
    return [
        head(READ_MONEY_FLOW),
        {"accounts": vertices("Account", rows)},
        {"tracked_inflows": [asdict(inflow) for inflow in tracked]},
    ]
