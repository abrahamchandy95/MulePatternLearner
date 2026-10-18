"""read_money_flow, B1: velocity. For each account of a request at one Monday, the money it
received and sent over the last 7, 30 and 90 days, and how fast the money it received in the
last 30 days left again, matched last in, first out (docs/design.md, "read_money_flow (B1):
velocity", and B1's walk in docs/features.md, "Account node")."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .rows import account_ids, as_of, parse_row, printed, projected, response_head

# The walk counts dwell in whole seconds, so every dwell it prints is a whole number of them.
SECOND_MS = 1_000
# Shares are each one division of two sums of cents; three of them can add up to their
# denominator's share with a rounding error of a few units in the last place.
ROUNDING = 1e-9


@dataclass(frozen=True)
class MoneyFlow:
    """An account's velocity columns at t (docs/features.md, "Account node", columns 4 to
    21), and its walk's steps.

    - in_amount_* and out_amount_*: the cents received and sent over the last 7, 30 and 90
      days, every payment counted.
    - flow_present: the walk tracked cents, R, received in its span.
    - left_*: the shares of R matched to outflows with a dwell under 15 minutes, 1 hour, 6
      hours, 1 day, 3 days and 7 days; held_share, the share of R matched to no outflow.
    - median_dwell: the dwell at which the matched cents, by ascending dwell, first reach half
      of R, as presence and milliseconds; absent when less than half left before t.
    - left_by_*: of the cents matched with a dwell under a day, the shares that left by
      transfer, card and cash.
    - scan_steps: the steps of the walk's scan, a work counter for mule bench, which no
      feature reads and parity ignores.

    Every share is 0 when its denominator is: flow_present shows R empty, left_1d the cents
    that left within a day.
    """

    account_id: str
    in_amount_7d: int
    in_amount_30d: int
    in_amount_90d: int
    out_amount_7d: int
    out_amount_30d: int
    out_amount_90d: int
    flow_present: bool
    left_15m: float
    left_1h: float
    left_6h: float
    left_1d: float
    left_3d: float
    left_7d: float
    held_share: float
    median_dwell_present: bool
    median_dwell_ms: int
    left_by_transfer: float
    left_by_card: float
    left_by_cash: float
    scan_steps: int


@dataclass(frozen=True)
class TrackedInflow:
    """A payment that account_id received and that belongs to a lot of its walk, with its
    lot's share matched to outflows within 24 hours, and its lot's dwell: the time until half
    of the lot's cents had left, as presence and milliseconds, absent when less than half left
    before t. It goes on the payment's edge as dwell_tracked, left_24h and dwell
    (docs/features.md, "Payment edges"); a payment received and not tracked has no row."""

    account_id: str
    payment_id: str
    left_24h: float
    dwell_present: bool
    dwell_ms: int


@dataclass(frozen=True)
class MoneyFlows:
    """The answer to one request: each account's row, sorted by id, and every tracked inflow,
    sorted by account and payment id."""

    accounts: tuple[MoneyFlow, ...]
    tracked_inflows: tuple[TrackedInflow, ...]


# The shares of a row, and the dwell windows among them, from the shortest.
WINDOWS = ("left_15m", "left_1h", "left_6h", "left_1d", "left_3d", "left_7d")
RAILS = ("left_by_transfer", "left_by_card", "left_by_cash")
SHARES = (*WINDOWS, "held_share", *RAILS)


def params(t_ms: int, accounts: Iterable[str]) -> dict[str, Any]:
    """The parameters of a request: the Monday t and the accounts, at least one, each once."""
    return {"t_ms": as_of(t_ms), "account_ids": account_ids(accounts)}


def _check_account(row: MoneyFlow) -> None:
    """Refuse a row whose values the walk cannot give."""
    from . import READ_MONEY_FLOW  # the registry imports this module

    def refuse(why: str) -> ValueError:
        return ValueError(f"{READ_MONEY_FLOW} answered {row.account_id} with {why}: {row}")

    for way in ("in", "out"):
        amounts = [getattr(row, f"{way}_amount_{days}d") for days in (7, 30, 90)]
        if amounts != sorted(amounts):
            raise refuse(f"{way} amounts that shrink over a longer window")
    shares = {name: getattr(row, name) for name in SHARES}
    if not all(0.0 <= share <= 1.0 for share in shares.values()):
        raise refuse("a share outside [0, 1]")
    windows = [shares[name] for name in WINDOWS]
    if windows != sorted(windows):
        raise refuse("a dwell share that shrinks over a longer window")
    if sum(shares[name] for name in RAILS) > 1.0 + ROUNDING:
        raise refuse("rail shares above 1")
    if row.left_1d == 0.0 and any(shares[name] for name in RAILS):
        raise refuse("rail shares of no cents that left within a day")
    if not row.flow_present and (any(shares.values()) or row.median_dwell_present):
        raise refuse("velocity without cents received in the span")
    if not row.median_dwell_present and row.median_dwell_ms:
        raise refuse("an absent median dwell that is not 0")
    if row.median_dwell_ms % SECOND_MS:
        raise refuse("a median dwell of no whole second")


def _check_inflow(row: TrackedInflow) -> None:
    """Refuse a tracked inflow whose values the walk cannot give."""
    from . import READ_MONEY_FLOW  # the registry imports this module

    if not 0.0 <= row.left_24h <= 1.0:
        raise ValueError(f"{READ_MONEY_FLOW} answered a share outside [0, 1]: {row}")
    if (not row.dwell_present and row.dwell_ms) or row.dwell_ms % SECOND_MS:
        raise ValueError(f"{READ_MONEY_FLOW} answered a dwell it cannot give: {row}")


def parse(rows: Sequence[object], accounts: Collection[str]) -> MoneyFlows:
    """The answer to the request params(t_ms, accounts) made: a row for every account asked
    for and no other, and the tracked inflows of those accounts, each payment once. An account
    with cents received in its span has a tracked inflow, and one without has none."""
    from . import READ_MONEY_FLOW, VERSIONS  # the registry imports this module

    response_head(rows, READ_MONEY_FLOW, VERSIONS[READ_MONEY_FLOW])
    found = [parse_row(MoneyFlow, row) for row in projected(rows, "accounts", READ_MONEY_FLOW)]
    asked = set(accounts)
    answered = [row.account_id for row in found]
    if len(set(answered)) != len(answered) or set(answered) != asked:
        raise ValueError(
            f"{READ_MONEY_FLOW} answered the accounts {sorted(answered)}, not {sorted(asked)}"
        )
    for row in found:
        _check_account(row)
    value = printed(rows, "tracked_inflows", READ_MONEY_FLOW)
    if not isinstance(value, list):
        raise ValueError(f"{READ_MONEY_FLOW} printed tracked_inflows as {value!r}, not a list")
    listed: list[object] = value
    inflows = [parse_row(TrackedInflow, row) for row in listed]
    payments = [row.payment_id for row in inflows]
    if len(set(payments)) != len(payments):
        raise ValueError(f"{READ_MONEY_FLOW} tracked a payment twice")
    for row in inflows:
        if row.account_id not in asked:
            raise ValueError(f"{READ_MONEY_FLOW} tracked an inflow of {row.account_id}")
        _check_inflow(row)
    tracking = {row.account_id for row in inflows}
    for row in found:
        if row.flow_present != (row.account_id in tracking):
            raise ValueError(
                f"{READ_MONEY_FLOW} answered {row.account_id} with flow_present "
                f"{row.flow_present} and {'some' if row.account_id in tracking else 'no'} "
                "tracked inflows"
            )
    return MoneyFlows(
        tuple(sorted(found, key=lambda row: row.account_id)),
        tuple(sorted(inflows, key=lambda row: (row.account_id, row.payment_id))),
    )
