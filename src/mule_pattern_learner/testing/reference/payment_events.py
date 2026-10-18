"""The reference answer of read_payment_events, B4: the payment graph (docs/design.md,
"read_payment_events (B4): the payment graph", and docs/features.md, "What one example's
subgraph holds", items 1 to 5 and 10, and "Payment edges"), written from the features apart
from the GSQL.

The subgraph of a scored account s:

- s's newest SCORED payments sent and its newest SCORED received in the lookback, and the
  hop-1 accounts at their other ends;
- each hop-1 account that is no account hub is expanded: its own newest COUNTERPARTY payments
  sent and received in the lookback, and the hop-2 accounts at their other ends. A hop-1 hub
  is not expanded;
- up to OWN other accounts of s's customer that exist at t through a visible ownership link,
  open or closed, most recently active first by their latest payment in the lookback, then
  those with none, ties by account id;
- an ownership edge from s and from each expanded hop-1 account to its customer, and from
  every account of the subgraph that s's customer owns to that customer, each through a
  visible ownership link.

Every cap orders newest first by a payment's second, since payments of one second have no
order (docs/design.md, "Time", under Ties), and breaks a tie by the smaller id, a payment's or
an account's. A payment or an account reached twice is one. A self-payment is read nowhere.

A payment's row depends on the payment, t and the account hubs alone: its cents, its rail, its
age t - ts; the payments from its sender to its receiver in [t - 90 days, ts), how many, and ts
minus the latest and the earliest of them; ts minus its sender's latest payment sent in
[t - 90 days, ts), absent at a hub sender, and ts minus its receiver's latest payment received
then, absent at a hub receiver. Each duration that may be absent is presence and milliseconds,
0 when absent. The account hubs are those the query's parameter names: the query reads no
other.

subgraph and Lookback are shared with read_shared_identity, which finds B4's accounts again.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ...config import DEFAULT_CONFIG
from ...tigergraph.queries import READ_PAYMENT_EVENTS
from ...tigergraph.queries.read_payment_events import EventAccount, OwnedBy, PaymentEvent
from .answer import INVISIBLE_HUB, head, hub_set, refused, request_refusal, vertices
from .graph import second

if TYPE_CHECKING:
    from .graph import AsOf, Payment, ReferenceGraph

SUBGRAPH = DEFAULT_CONFIG.subgraph
SCORED = SUBGRAPH.scored_payments
COUNTERPARTY = SUBGRAPH.counterparty_payments
OWN = SUBGRAPH.own_accounts


class Lookback:
    """The payments of the lookback by account, each way, but self-payments, each read once."""

    def __init__(self, view: AsOf) -> None:
        self.view = view
        self._sent: dict[str, list[Payment]] = {}
        self._received: dict[str, list[Payment]] = {}

    def sent(self, account_id: str) -> list[Payment]:
        """The account's payments sent in the lookback, oldest first."""
        if account_id not in self._sent:
            self._sent[account_id] = self.view.sent(account_id)
        return self._sent[account_id]

    def received(self, account_id: str) -> list[Payment]:
        """The account's payments received in the lookback, oldest first."""
        if account_id not in self._received:
            self._received[account_id] = self.view.received(account_id)
        return self._received[account_id]

    def both(self, account_id: str) -> list[Payment]:
        """The account's payments either way in the lookback."""
        return [*self.sent(account_id), *self.received(account_id)]


def newest(payments: Iterable[Payment], cap: int) -> list[Payment]:
    """The newest cap payments, by second, ties by the smaller id."""
    return sorted(payments, key=lambda payment: (-second(payment.ts), payment.id))[:cap]


def other_end(payment: Payment, account_id: str) -> str:
    """The account at the other end of a payment of account_id."""
    return payment.receiver if payment.sender == account_id else payment.sender


@dataclass(frozen=True)
class Subgraph:
    """B4's subgraph of a scored account: its customer, if it has one; its hop-1 accounts and
    those expanded; its customer's other accounts kept; every account and payment, by id; and
    its ownership edges."""

    scored: str
    customer: str | None
    hop1: frozenset[str]
    expanded: frozenset[str]
    own_accounts: tuple[str, ...]
    accounts: frozenset[str]
    payments: dict[str, Payment]
    owned_by: frozenset[OwnedBy]


def own_accounts(lookback: Lookback, scored: str, customer: str | None) -> tuple[str, ...]:
    """The other accounts of the scored account's customer that the subgraph keeps."""
    view = lookback.view
    if customer is None:
        return ()
    others = [
        link.target
        for link in view.graph.owned.get(customer, [])
        if link.target != scored and view.visible(link) and view.exists(link.target)
    ]

    def order(account_id: str) -> tuple[int, int, str]:
        latest = max((p.ts for p in lookback.both(account_id)), default=None)
        return (1, 0, account_id) if latest is None else (0, -second(latest), account_id)

    return tuple(sorted(others, key=order)[:OWN])


def subgraph(lookback: Lookback, scored: str, hubs: Collection[str]) -> Subgraph:
    """B4's subgraph of the scored account at t, with hubs the account hubs of t."""
    view = lookback.view
    picked = [*newest(lookback.sent(scored), SCORED), *newest(lookback.received(scored), SCORED)]
    payments = {payment.id: payment for payment in picked}
    hop1 = frozenset(other_end(payment, scored) for payment in picked)
    expanded = frozenset(account for account in hop1 if account not in hubs)
    accounts = {scored, *hop1}
    for account in expanded:
        for payment in (
            *newest(lookback.sent(account), COUNTERPARTY),
            *newest(lookback.received(account), COUNTERPARTY),
        ):
            payments[payment.id] = payment
            accounts.add(other_end(payment, account))
    customer = view.owner(scored)
    kept = own_accounts(lookback, scored, customer)
    accounts.update(kept)
    owned: set[OwnedBy] = set()
    for account in (scored, *expanded):
        owner = view.owner(account)
        if owner is not None:
            owned.add(OwnedBy(account, owner))
    if customer is not None:
        owned.update(
            OwnedBy(account, customer) for account in accounts if view.owner(account) == customer
        )
    return Subgraph(
        scored, customer, hop1, expanded, kept, frozenset(accounts), payments, frozenset(owned)
    )


def _gap(ts: int, earlier: list[int]) -> tuple[bool, int]:
    """ts minus the latest of earlier as presence and milliseconds, absent with none."""
    return (True, ts - max(earlier)) if earlier else (False, 0)


def payment_event(lookback: Lookback, payment: Payment, hubs: Collection[str]) -> PaymentEvent:
    """A payment's row at t, with hubs the account hubs of t."""
    view = lookback.view
    ts = payment.ts
    sent = [p.ts for p in lookback.sent(payment.sender) if p.ts < ts]
    received = [p.ts for p in lookback.received(payment.receiver) if p.ts < ts]
    pair = [
        p.ts for p in lookback.sent(payment.sender) if p.ts < ts and p.receiver == payment.receiver
    ]
    pair_gap = _gap(ts, pair)
    sender_gap = (False, 0) if payment.sender in hubs else _gap(ts, sent)
    receiver_gap = (False, 0) if payment.receiver in hubs else _gap(ts, received)
    return PaymentEvent(
        payment_id=payment.id,
        sender_id=payment.sender,
        receiver_id=payment.receiver,
        amount=payment.cents,
        rail=payment.rail,
        age_ms=view.t - ts,
        pair_prior_count=len(pair),
        pair_gap_present=pair_gap[0],
        pair_gap_ms=pair_gap[1],
        pair_age_present=bool(pair),
        pair_age_ms=ts - min(pair) if pair else 0,
        sender_gap_present=sender_gap[0],
        sender_gap_ms=sender_gap[1],
        receiver_gap_present=receiver_gap[0],
        receiver_gap_ms=receiver_gap[1],
    )


def read_payment_events(
    graph: ReferenceGraph,
    t_ms: int,
    account_ids: list[str],
    hub_accounts: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """The payment graphs of the request's accounts at t: every account and payment of them,
    each once and sorted by id, and per scored account, sorted, the ids of its accounts and
    payments and its ownership edges. A scored account with no payment or no ownership edge
    has no key in that map, as a TigerGraph MapAccum prints it."""
    status = request_refusal(graph, t_ms, account_ids)
    if status is not None:
        return refused(status)
    hubs = hub_set(graph, t_ms, hub_accounts)
    if hubs is None:
        return refused(INVISIBLE_HUB)
    lookback = Lookback(graph.as_of(t_ms))
    graphs = [subgraph(lookback, scored, hubs) for scored in sorted(account_ids)]
    accounts = sorted({account for found in graphs for account in found.accounts})
    payments = {pid: p for found in graphs for pid, p in found.payments.items()}
    rows = [payment_event(lookback, payments[pid], hubs) for pid in sorted(payments)]
    return [
        head(READ_PAYMENT_EVENTS),
        {
            "accounts": vertices(
                "Account",
                (EventAccount(a, graph.accounts[a].is_external) for a in accounts),
            )
        },
        {
            "payments": [
                {
                    "v_id": row.payment_id,
                    "v_type": payments[row.payment_id].kind,
                    "attributes": asdict(row),
                }
                for row in rows
            ]
        },
        {"subgraph_accounts": {found.scored: sorted(found.accounts) for found in graphs}},
        {
            "subgraph_payments": {
                found.scored: sorted(found.payments) for found in graphs if found.payments
            }
        },
        {
            "owned_by": {
                found.scored: [
                    asdict(edge) for edge in sorted(found.owned_by, key=lambda e: e.account_id)
                ]
                for found in graphs
                if found.owned_by
            }
        },
    ]
