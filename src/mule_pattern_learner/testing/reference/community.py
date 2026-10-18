"""The reference answer of read_community, B6: how the account's network formed
(docs/design.md, "read_community (B6): how the mule network formed", and docs/features.md,
"The community block"), written from the features apart from the GSQL, under the rules of
sharing of shared_identity.Sharing.

The members, as of t, each in the first round that reaches it:

- round 0 is the scored account's owner, through a visible ownership link; an account with
  none has no member, and every number of its block is 0;
- round 1 adds the customers that share a node that is no leaf with it, and round 2 those that
  share one with a member kept in round 1; a candidate's time is the latest, over the nodes it
  shares with a member kept in the round before, of the later of the two sharing starts;
- the payment round adds the owners, through a visible ownership link, of the internal
  accounts that are no hub at the other end of a payment of the last 30 days, but
  self-payments, of an account of a member kept in rounds 0 to 2; a candidate's time is its
  latest such payment.

A member's accounts are those that exist at t through a visible ownership link, open or
closed, but account hubs. Each round's candidates are kept newest first, ties by customer id,
while the community holds fewer than COMMUNITY; capped says that some candidate was left out.

The block: the members kept; capped; the nodes that are no leaf and that two members or more
share, and the links, a link being a member sharing one of them, over members times those
nodes; the share of the links whose member's sharing start there is in the last 30 days; and
over the payments of the last 30 days, but self-payments, of the members' accounts, the cents
of those between accounts of two different members (internal), and the internal ones over the
internal and external ones, external meaning with an account of no member, an account hub
included, and their count. A payment between two accounts of one member is neither. Every
share is 0 with an empty denominator. The payment round adds members, never links: a link is
sharing alone. The account hubs are those the query's parameter names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from ...config import DEFAULT_CONFIG
from ...tigergraph.queries import READ_COMMUNITY
from ...tigergraph.queries.read_community import Community
from .answer import INVISIBLE_HUB, head, hub_set, refused, request_refusal, vertices
from .graph import CURRENT_MS
from .payment_events import Lookback, other_end
from .shared_identity import Sharing

if TYPE_CHECKING:
    from .graph import Payment, ReferenceGraph

COMMUNITY = DEFAULT_CONFIG.subgraph.community


class _Members:
    """A community as its rounds keep it: each member with its round, and whether a candidate
    was left out."""

    def __init__(self, owner: str) -> None:
        self.round: dict[str, int] = {owner: 0}
        self.capped = False

    def keep(self, number: int, candidates: Mapping[str, int]) -> list[str]:
        """The round's candidates, each with its time, kept newest first, ties by id, while
        the community holds fewer than COMMUNITY. Those kept."""
        ordered = sorted(candidates, key=lambda customer: (-candidates[customer], customer))
        room = COMMUNITY - len(self.round)
        if len(ordered) > room:
            self.capped = True
        kept = ordered[:room]
        self.round.update((customer, number) for customer in kept)
        return kept


def members(sharing: Sharing, owner: str, hubs: Collection[str]) -> _Members:
    """The community of the owner at t, with hubs the account hubs of t."""
    view = sharing.view
    found = _Members(owner)
    frontier = [owner]
    for number in (1, 2):
        times: dict[str, int] = {}
        for member in frontier:
            for node in sharing.shared_nodes(member):
                own = sharing.sharing_start(member, node)
                for other in sharing.sharers(node):
                    if other in found.round:
                        continue
                    time = max(own, sharing.sharing_start(other, node))
                    times[other] = max(times.get(other, time), time)
        frontier = found.keep(number, times)
    month = view.t - CURRENT_MS
    times = {}
    for member in list(found.round):
        for account in sharing.accounts(member):
            if account in hubs:
                continue
            for payment in sharing.lookback.both(account):
                other = other_end(payment, account)
                if (
                    payment.ts < month
                    or view.graph.accounts[other].is_external
                    or other in hubs
                    or not view.exists(other)
                ):
                    continue
                owner_of = view.owner(other)
                if owner_of is not None and owner_of not in found.round:
                    times[owner_of] = max(times.get(owner_of, payment.ts), payment.ts)
    found.keep(3, times)
    return found


def community(sharing: Sharing, scored: str, hubs: Collection[str]) -> Community:
    """The scored account's community block at t, with hubs the account hubs of t."""
    view = sharing.view
    owner = view.owner(scored)
    if owner is None:
        return Community(scored, 0, False, 0, 0.0, 0.0, 0, 0.0, 0)
    found = members(sharing, owner, hubs)
    kept = list(found.round)
    links: Counter[tuple[str, str]] = Counter()
    new_links: Counter[tuple[str, str]] = Counter()
    month = view.t - CURRENT_MS
    for member in kept:
        for node in sharing.shared_nodes(member):
            links[node] += 1
            if sharing.sharing_start(member, node) >= month:
                new_links[node] += 1
    shared = [node for node, sharers in links.items() if sharers >= 2]
    link_count = sum(links[node] for node in shared)
    new_count = sum(new_links[node] for node in shared)
    member_of = {
        account: member
        for member in kept
        for account in sharing.accounts(member)
        if account not in hubs
    }
    payments: dict[str, Payment] = {
        payment.id: payment
        for account in member_of
        for payment in sharing.lookback.both(account)
        if payment.ts >= month
    }
    internal = external = internal_cents = 0
    for payment in payments.values():
        sender, receiver = member_of.get(payment.sender), member_of.get(payment.receiver)
        if sender is None or receiver is None:
            external += 1
        elif sender != receiver:
            internal += 1
            internal_cents += payment.cents
    counted = internal + external
    return Community(
        account_id=scored,
        members=len(kept),
        capped=found.capped,
        shared_nodes=len(shared),
        density=link_count / (len(kept) * len(shared)) if shared else 0.0,
        new_link_share_30d=new_count / link_count if link_count else 0.0,
        internal_amount_30d=internal_cents,
        isolation=internal / counted if counted else 0.0,
        member_payments_30d=counted,
    )


def read_community(
    graph: ReferenceGraph,
    t_ms: int,
    account_ids: list[str],
    hub_accounts: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """The community blocks of the request's accounts at t, sorted by id."""
    status = request_refusal(graph, t_ms, account_ids)
    if status is not None:
        return refused(status)
    hubs = hub_set(graph, t_ms, hub_accounts)
    if hubs is None:
        return refused(INVISIBLE_HUB)
    sharing = Sharing(Lookback(graph.as_of(t_ms)))
    rows = [community(sharing, scored, hubs) for scored in sorted(account_ids)]
    return [head(READ_COMMUNITY), {"accounts": vertices("Account", rows)}]
