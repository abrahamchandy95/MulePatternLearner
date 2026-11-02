"""read_payment_events, B4: the payment graph. For each account of a request at one time t, the
accounts and payments of its subgraph, one edge per payment with its own timing, and its
ownership edges (docs/design.md, "read_payment_events (B4): the payment graph", and
docs/features.md, "What one example's subgraph holds" and "Payment edges")."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...config import DEFAULT_CONFIG
from ...contract import PAYMENT_RAILS
from .rows import (
    LOOKBACK_MS,
    account_ids,
    as_of,
    ids,
    parse_row,
    printed,
    projected,
    response_head,
    vertex_set,
)

# A payment's rail: its payment_rail, or zelle for a Zelle transfer.
RAILS = (*PAYMENT_RAILS, "zelle")
SUBGRAPH = DEFAULT_CONFIG.subgraph
# The most payments a subgraph holds: the scored account's newest each way, and the newest
# each way of every hop-1 account, of which there is at most one per payment of the scored
# account.
MOST_PAYMENTS = 2 * SUBGRAPH.scored_payments * (1 + 2 * SUBGRAPH.counterparty_payments)


@dataclass(frozen=True)
class EventAccount:
    """An account of a subgraph and whether it is another bank's (docs/features.md,
    "Account node", column 1)."""

    account_id: str
    is_external: bool


@dataclass(frozen=True)
class PaymentEvent:
    """A payment edge, from its sender to its receiver (docs/features.md, "Payment edges",
    attributes 1 to 14); B1's dwell joins it by payment_id.

    - amount: its cents; rail: its payment_rail, or zelle for a Zelle transfer; age_ms: t
      minus its time, never absent.
    - pair_prior_count: the payments from its sender to its receiver in the lookback before
      it; pair_gap and pair_age: its time minus the latest and the earliest of them.
    - sender_gap: its time minus the sender's latest payment sent in the lookback before it,
      absent at a hub sender; receiver_gap, the same for the receiver's latest payment
      received, absent at a hub receiver.

    Each duration that may be absent is presence and milliseconds, 0 when absent.
    """

    payment_id: str
    sender_id: str
    receiver_id: str
    amount: int
    rail: str
    age_ms: int
    pair_prior_count: int
    pair_gap_present: bool
    pair_gap_ms: int
    pair_age_present: bool
    pair_age_ms: int
    sender_gap_present: bool
    sender_gap_ms: int
    receiver_gap_present: bool
    receiver_gap_ms: int


@dataclass(frozen=True)
class OwnedBy:
    """An ownership edge of a subgraph: an account and its customer (docs/features.md,
    "Ownership edges")."""

    account_id: str
    customer_id: str


@dataclass(frozen=True)
class PaymentGraph:
    """One scored account's payment graph: its accounts, the scored account among them, and
    its payments, each sorted by id, and its ownership edges, sorted by account."""

    scored_id: str
    accounts: tuple[EventAccount, ...]
    payments: tuple[PaymentEvent, ...]
    owned_by: tuple[OwnedBy, ...]


def params(t_ms: int, accounts: Iterable[str], hubs: Iterable[str]) -> dict[str, Any]:
    """The parameters of a request: the time t, the accounts, at least one, each once, and
    the account hubs of t, read_hubs' accounts of the same t, each once."""
    return {
        "t_ms": as_of(t_ms),
        "account_ids": account_ids(accounts),
        "hub_accounts": vertex_set(hubs),
    }


def _check_payment(row: PaymentEvent) -> None:
    """Refuse a payment row whose values the query cannot give."""
    from . import READ_PAYMENT_EVENTS  # the registry imports this module

    def refuse(why: str) -> ValueError:
        return ValueError(f"{READ_PAYMENT_EVENTS} answered {row.payment_id} with {why}: {row}")

    if row.sender_id == row.receiver_id:
        raise refuse("one account at both ends, a self-payment")
    if row.rail not in RAILS:
        raise refuse(f"the rail {row.rail!r}, not one of {RAILS}")
    if not 0 < row.age_ms <= LOOKBACK_MS:
        raise refuse("an age outside the lookback")
    present = row.pair_prior_count > 0
    if row.pair_gap_present != present or row.pair_age_present != present:
        raise refuse("pair gaps that disagree with its earlier pair payments")
    for name in ("pair_gap", "pair_age", "sender_gap", "receiver_gap"):
        shown: bool = getattr(row, f"{name}_present")
        gap: int = getattr(row, f"{name}_ms")
        if not shown and gap:
            raise refuse(f"an absent {name} that is not 0")
        if gap + row.age_ms > LOOKBACK_MS:
            raise refuse(f"a {name} before the lookback")
    if row.pair_gap_ms > row.pair_age_ms:
        raise refuse("a latest pair payment before its earliest")
    for end in ("sender", "receiver"):
        if getattr(row, f"{end}_gap_present") and present:
            if getattr(row, f"{end}_gap_ms") > row.pair_gap_ms:
                raise refuse(f"a {end} gap longer than its pair gap")


def _members(rows: Sequence[object], key: str, asked: set[str]) -> dict[str, tuple[str, ...]]:
    """A printed map of scored accounts to the ids of their accounts or payments."""
    from . import READ_PAYMENT_EVENTS  # the registry imports this module

    value = printed(rows, key, READ_PAYMENT_EVENTS)
    if not isinstance(value, Mapping):
        raise ValueError(f"{READ_PAYMENT_EVENTS} printed {key} as {value!r}, not a map")
    found: Mapping[object, object] = value
    if not set(found) <= asked:
        raise ValueError(f"{READ_PAYMENT_EVENTS} printed {key} of {sorted(map(str, found))}")
    return {str(scored): ids(members, f"{key} of {scored}") for scored, members in found.items()}


def _owned_by(rows: Sequence[object], asked: set[str]) -> dict[str, tuple[OwnedBy, ...]]:
    """The printed ownership edges of each scored account."""
    from . import READ_PAYMENT_EVENTS  # the registry imports this module

    value = printed(rows, "owned_by", READ_PAYMENT_EVENTS)
    if not isinstance(value, Mapping):
        raise ValueError(f"{READ_PAYMENT_EVENTS} printed owned_by as {value!r}, not a map")
    found: Mapping[object, object] = value
    if not set(found) <= asked:
        raise ValueError(f"{READ_PAYMENT_EVENTS} printed owned_by of {sorted(map(str, found))}")
    edges: dict[str, tuple[OwnedBy, ...]] = {}
    for scored, listed in found.items():
        if not isinstance(listed, list):
            raise ValueError(f"{READ_PAYMENT_EVENTS} printed owned_by of {scored} as {listed!r}")
        items: list[object] = listed
        parsed = sorted(
            (parse_row(OwnedBy, item) for item in items), key=lambda edge: edge.account_id
        )
        owned = [edge.account_id for edge in parsed]
        if len(set(owned)) != len(owned):
            raise ValueError(f"{READ_PAYMENT_EVENTS} gave an account of {scored} two owners")
        edges[str(scored)] = tuple(parsed)
    return edges


def parse(rows: Sequence[object], accounts: Collection[str]) -> tuple[PaymentGraph, ...]:
    """The answer to the request params(t_ms, accounts, hubs) made: the payment graph of
    every account asked for and no other, sorted by the scored account's id.

    Each subgraph holds its scored account, both ends of each of its payments and the
    account of each of its ownership edges, and at most the payments its caps allow; every
    account and payment printed belongs to a subgraph.
    """
    from . import READ_PAYMENT_EVENTS  # the registry imports this module

    response_head(rows, READ_PAYMENT_EVENTS)
    asked = set(accounts)
    found_accounts = [
        parse_row(EventAccount, row) for row in projected(rows, "accounts", READ_PAYMENT_EVENTS)
    ]
    found_payments = [
        parse_row(PaymentEvent, row) for row in projected(rows, "payments", READ_PAYMENT_EVENTS)
    ]
    by_account = {row.account_id: row for row in found_accounts}
    by_payment = {row.payment_id: row for row in found_payments}
    if len(by_account) != len(found_accounts) or len(by_payment) != len(found_payments):
        raise ValueError(f"{READ_PAYMENT_EVENTS} printed an account or a payment twice")
    for row in found_payments:
        _check_payment(row)
    members = _members(rows, "subgraph_accounts", asked)
    events = _members(rows, "subgraph_payments", asked)
    owned_by = _owned_by(rows, asked)
    if set(members) != asked:
        raise ValueError(
            f"{READ_PAYMENT_EVENTS} answered the accounts {sorted(members)}, not {sorted(asked)}"
        )
    graphs: list[PaymentGraph] = []
    for scored in sorted(asked):
        held = set(members[scored])
        paid = events.get(scored, ())
        edges = owned_by.get(scored, ())
        if scored not in held:
            raise ValueError(f"{READ_PAYMENT_EVENTS} left {scored} out of its own subgraph")
        if not held <= by_account.keys() or not set(paid) <= by_payment.keys():
            raise ValueError(f"{READ_PAYMENT_EVENTS} gave {scored} an account or payment it lacks")
        if len(paid) > MOST_PAYMENTS:
            raise ValueError(f"{READ_PAYMENT_EVENTS} gave {scored} more payments than its caps")
        for payment in paid:
            ends = {by_payment[payment].sender_id, by_payment[payment].receiver_id}
            if not ends <= held:
                raise ValueError(f"{READ_PAYMENT_EVENTS} gave {scored} {payment} without its ends")
        if not {edge.account_id for edge in edges} <= held:
            raise ValueError(f"{READ_PAYMENT_EVENTS} gave {scored} an owner of no account of it")
        graphs.append(
            PaymentGraph(
                scored,
                tuple(by_account[account] for account in sorted(held)),
                tuple(by_payment[payment] for payment in sorted(paid)),
                edges,
            )
        )
    used_accounts = {account for scored in asked for account in members[scored]}
    used_payments = {payment for scored in asked for payment in events.get(scored, ())}
    if used_accounts != by_account.keys() or used_payments != by_payment.keys():
        raise ValueError(f"{READ_PAYMENT_EVENTS} printed an account or a payment of no subgraph")
    return tuple(graphs)
