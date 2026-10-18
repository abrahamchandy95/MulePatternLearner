"""The reference answer of read_counterparties, B2: who pays it and whom it pays
(docs/design.md, "read_counterparties (B2): who pays it and whom it pays", and
docs/features.md, "Account node", columns 22 to 33), written from the features apart from the
GSQL.

- A payer or a payee is the account at the other end of a payment of the lookback; a
  self-payment is read nowhere.
- A payer is new over w when its earliest payment to the account in the lookback is in the
  last w, and new payers split by the payer's is_external; a payee likewise, by its earliest
  payment from the account.
- max_new_payers_24h reads the first-payment times of the payers new over 30 days, one per
  payer, cut to the newest RECENT by the whole-second rule.
- The account hubs are those the query's parameter names: the query reads no other.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from ...tigergraph.queries import READ_COUNTERPARTIES
from ...tigergraph.queries.read_counterparties import Counterparties
from ...tigergraph.queries.rows import DAY_MS
from .answer import INVISIBLE_HUB, head, hub_set, refused, request_refusal, vertices
from .graph import kept_times, most_within, share

if TYPE_CHECKING:
    from .graph import AsOf, ReferenceGraph

# The windows over which a payer or a payee is new, and the window of the counts and shares.
WEEK_MS = 7 * DAY_MS
MONTH_MS = 30 * DAY_MS


def counterparties(view: AsOf, account_id: str, hubs: Collection[str]) -> Counterparties:
    """The account's counterparty row at t, with hubs the account hubs of t."""
    received, sent = view.received(account_id), view.sent(account_id)
    month, week = view.t - MONTH_MS, view.t - WEEK_MS
    payer_first: dict[str, int] = {}
    for payment in received:
        payer_first[payment.sender] = min(payer_first.get(payment.sender, payment.ts), payment.ts)
    payee_first: dict[str, int] = {}
    for payment in sent:
        payee_first[payment.receiver] = min(
            payee_first.get(payment.receiver, payment.ts), payment.ts
        )
    recent_in = [payment for payment in received if payment.ts >= month]
    recent_out = [payment for payment in sent if payment.ts >= month]
    new_payers = {payer: first for payer, first in payer_first.items() if first >= month}

    def new(firsts: Mapping[str, int], since: int, external: bool | None = None) -> int:
        return sum(
            1
            for account, first in firsts.items()
            if first >= since
            and (external is None or view.graph.accounts[account].is_external == external)
        )

    received_cents = sum(payment.cents for payment in recent_in)
    new_cents = sum(payment.cents for payment in recent_in if payment.sender in new_payers)
    sent_cents = sum(payment.cents for payment in recent_out)
    by_payee: dict[str, int] = defaultdict(int)
    for payment in recent_out:
        by_payee[payment.receiver] += payment.cents
    top = max([0, *by_payee.values()])
    to_hubs = sum(payment.cents for payment in recent_out if payment.receiver in hubs)
    return Counterparties(
        account_id=account_id,
        payers_30d=len({payment.sender for payment in recent_in}),
        payees_30d=len(by_payee),
        new_internal_payers_7d=new(payer_first, week, external=False),
        new_internal_payers_30d=new(payer_first, month, external=False),
        new_external_payers_7d=new(payer_first, week, external=True),
        new_external_payers_30d=new(payer_first, month, external=True),
        new_payees_7d=new(payee_first, week),
        new_payees_30d=new(payee_first, month),
        new_payer_share_30d=share(new_cents, received_cents),
        max_new_payers_24h=most_within(kept_times(list(new_payers.values())), DAY_MS),
        top_payee_share_30d=share(top, sent_cents),
        hub_payee_share_30d=share(to_hubs, sent_cents),
    )


def read_counterparties(
    graph: ReferenceGraph,
    t_ms: int,
    account_ids: list[str],
    hub_accounts: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """The counterparty rows of the request's accounts at t, sorted by id."""
    status = request_refusal(graph, t_ms, account_ids)
    if status is not None:
        return refused(status)
    hubs = hub_set(graph, t_ms, hub_accounts)
    if hubs is None:
        return refused(INVISIBLE_HUB)
    view = graph.as_of(t_ms)
    rows = [counterparties(view, account_id, hubs) for account_id in sorted(account_ids)]
    return [head(READ_COUNTERPARTIES), {"accounts": vertices("Account", rows)}]
