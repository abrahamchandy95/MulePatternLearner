"""The reference answer of read_shared_identity, B5: the device and IP graph (docs/design.md,
"read_shared_identity (B5): the device and IP graph", "Time" and "Shared rules", and
docs/features.md, "What one example's subgraph holds" and the customer, device, IP, enrolment
and seen tables), written from the features apart from the GSQL. Sharing holds the rules of
time and sharing that read_community (B6) reads too.

The rules, as of t:

- A customer's accounts are those it owns through a visible ownership link that exist at t,
  open or closed; its own payments are those its accounts sent in the lookback, but
  self-payments. A customer's visible enrolment links to one node are one link, whose start is
  the earliest of theirs; it holds the link in the last 30 days when one of them is current.
- A start s of a customer on a node, a link's start or its first use seen in the lookback, is
  known when s > t - 90 days, s > its joined_ms, and one of its own payments in
  [t - 90 days, s) was not seen on that node.
- A node is a leaf when the crowd rule of read_hubs gives it more than CROWD customers.
- A customer shares a node that is no leaf when it holds a link to it in the last 30 days, or
  a deposit account of it sent a payment seen on it in the last 30 days, but self-payments. Its
  sharing start there is the earliest of its visible link starts and its first use from a
  deposit account in the lookback, each t - 90 days when not known. Two customers began
  sharing a node at the later of their starts, and began sharing at the earliest of those over
  the nodes they share.

The subgraph of a scored account s holds B4's accounts (payment_events.subgraph) and:

- D_s, the kept links of C_s, s's customer: up to NODES visible links of each kind by their
  newest evidence, the later of the link's known start and C_s's latest own payment seen on
  the node, links with neither last, ties by node id. Each expanded hop-1 account's customer
  keeps its links by the same rule.
- S_s, the kept seen nodes of s: up to NODES of each kind seen on its payments sent in the
  lookback, latest use first, ties by node id; each expanded hop-1 account keeps its own.
- The co-users: the customers other than C_s counted by the crowd rule on a node of D_s or
  S_s that is no leaf, the newest CO_USERS by their newest evidence on those nodes (the later
  of a known link start and the latest use from a deposit account, t - 90 days with neither),
  ties by customer id. Each brings one of its deposit accounts: the one last seen sending on
  those nodes, else the one with the latest payment in the lookback, else one with none, ties
  by account id. A co-user with a visible link to one of those nodes is a customer of the
  subgraph.
- Its customers: C_s, each expanded hop-1 account's customer and those co-users. Its nodes:
  D_s, S_s, and each expanded hop-1 account's customer's kept links and its own kept seen
  nodes.
- Enrolment edges from C_s to D_s; owner-enrolment edges from every account of C_s in the
  subgraph to D_s and from each expanded hop-1 account to its customer's kept links; seen
  edges from s to S_s and from each expanded hop-1 account to its own. Around every node of
  D_s and S_s that is no leaf, every enrolment, owner-enrolment and seen edge between it and a
  customer or an account of the subgraph. An account hub other than s has no edge, being
  flag-only (docs/design.md, "Shared rules": a leaf is never expanded and never counted as
  sharing).

An owner-enrolment edge carries its account's customer's link, so every edge of a link reads
the same: the time since its known start, and since the customer's latest own payment seen
on the node. A seen edge carries the account's payments sent in the lookback seen on the
node: how many, the time since the first when it is a known start of the account's customer,
and since the last. A leaf's row is its flag alone, every count 0: its measure and kind come
from read_hubs. The account hubs are those the query's parameter names.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable
from dataclasses import asdict, astuple, dataclass
from typing import TYPE_CHECKING, Any

from ...config import DEFAULT_CONFIG
from ...tigergraph.queries import READ_SHARED_IDENTITY
from ...tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityAccount,
    NodeRow,
    OwnerEnrolled,
    Seen,
)
from ...tigergraph.queries.rows import DAY_MS
from .answer import INVISIBLE_HUB, head, hub_set, refused, request_refusal, vertices
from .graph import CURRENT_MS, DEVICE, IP, NODE_KINDS
from .infrastructure import crowd_customers
from .payment_events import Lookback, subgraph

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from .graph import AsOf, Payment, ReferenceGraph

SUBGRAPH = DEFAULT_CONFIG.subgraph
NODES = SUBGRAPH.devices_ips
CO_USERS = SUBGRAPH.co_users
CROWD = SUBGRAPH.crowd_threshold
# The windows over which a customer of a node is new, 7 and 30 days.
NEW_WINDOWS_MS = tuple(days * DAY_MS for days in SUBGRAPH.new_windows)
# A device or an IP: its vertex type and its id.
type Node = tuple[str, str]


def seen_node(payment: Payment, kind: str) -> str | None:
    """The device or IP of a kind seen on a payment, if any."""
    return payment.device if kind == DEVICE else payment.ip


class Sharing:
    """The customers, nodes and links of the graph as of t under the rules of sharing, each
    worked out once."""

    def __init__(self, lookback: Lookback) -> None:
        self.lookback = lookback
        self.view: AsOf = lookback.view
        self.recent = self.view.t - CURRENT_MS
        self._crowd: dict[Node, tuple[set[str], set[str]]] = {}
        self._own: dict[str, list[Payment]] = {}

    # A customer's accounts and own payments.

    def accounts(self, customer: str) -> tuple[str, ...]:
        """The accounts the customer owns that exist at t through a visible ownership link,
        open or closed, by id."""
        view = self.view
        return tuple(
            link.target
            for link in view.graph.owned.get(customer, [])
            if view.visible(link) and view.exists(link.target)
        )

    def deposits(self, customer: str) -> tuple[str, ...]:
        """The customer's accounts that are deposit accounts."""
        accounts = self.view.graph.accounts
        return tuple(a for a in self.accounts(customer) if accounts[a].account_type == "deposit")

    def own_sent(self, customer: str) -> list[Payment]:
        """The customer's own payments: those its accounts sent in the lookback."""
        if customer not in self._own:
            self._own[customer] = [
                payment
                for account in self.accounts(customer)
                for payment in self.lookback.sent(account)
            ]
        return self._own[customer]

    def uses(self, customer: str, node: Node, *, deposit: bool) -> list[int]:
        """The times of the customer's own payments seen on the node, from its deposit
        accounts only when deposit is set."""
        kind, node_id = node
        accounts = self.view.graph.accounts
        return [
            payment.ts
            for payment in self.own_sent(customer)
            if seen_node(payment, kind) == node_id
            and (not deposit or accounts[payment.sender].account_type == "deposit")
        ]

    # A customer's links and their starts.

    def link_starts(self, customer: str, kind: str) -> dict[str, list[int]]:
        """The starts of the customer's visible enrolment links, by node of a kind."""
        starts: dict[str, list[int]] = defaultdict(list)
        for link in self.view.graph.enrolled[kind].get(customer, []):
            if self.view.visible(link):
                starts[link.target].append(link.start)
        return dict(starts)

    def linked(self, customer: str, node: Node) -> bool:
        """The customer has a visible enrolment link to the node."""
        return node[1] in self.link_starts(customer, node[0])

    def current(self, customer: str, kind: str) -> set[str]:
        """The nodes of a kind the customer holds a link to in the last 30 days."""
        return {
            link.target
            for link in self.view.graph.enrolled[kind].get(customer, [])
            if self.view.current(link)
        }

    def known(self, customer: str, node: Node, start: int) -> bool:
        """A start of the customer on the node is known: after the lookback's start, after it
        joined, and after one of its own payments not seen on the node."""
        view = self.view
        kind, node_id = node
        return (
            start > view.start
            and start > view.graph.joined_ms[customer]
            and any(
                payment.ts < start and seen_node(payment, kind) != node_id
                for payment in self.own_sent(customer)
            )
        )

    def known_link_start(self, customer: str, node: Node) -> int | None:
        """The start of the customer's link to the node, the earliest of its visible links,
        when it is known."""
        starts = self.link_starts(customer, node[0]).get(node[1])
        if not starts or not self.known(customer, node, min(starts)):
            return None
        return min(starts)

    def link_fields(self, customer: str, node: Node) -> tuple[bool, int, bool, int]:
        """The fields of the customer's link to the node: the time since its known start, and
        since the customer's latest own payment seen on it, each as presence and
        milliseconds."""
        t = self.view.t
        start = self.known_link_start(customer, node)
        last = max(self.uses(customer, node, deposit=False), default=None)
        return (
            start is not None,
            0 if start is None else t - start,
            last is not None,
            0 if last is None else t - last,
        )

    # Nodes: leaves, sharing and the sharing start.

    def crowd(self, node: Node) -> tuple[set[str], set[str]]:
        """The node's customers by the crowd rule, and those enrolled on it."""
        if node not in self._crowd:
            self._crowd[node] = crowd_customers(self.view, *node)
        return self._crowd[node]

    def leaf(self, node: Node) -> bool:
        """The node is a leaf: more than CROWD customers by the crowd rule."""
        return len(self.crowd(node)[0]) > CROWD

    def shares(self, customer: str, node: Node) -> bool:
        """The customer shares the node."""
        return not self.leaf(node) and (
            node[1] in self.current(customer, node[0])
            or any(ts >= self.recent for ts in self.uses(customer, node, deposit=True))
        )

    def shared_nodes(self, customer: str) -> list[Node]:
        """The nodes the customer shares, sorted."""
        found: set[Node] = {
            (kind, node_id) for kind in NODE_KINDS for node_id in self.current(customer, kind)
        }
        for payment in self.own_sent(customer):
            if payment.ts >= self.recent:
                for kind in NODE_KINDS:
                    node_id = seen_node(payment, kind)
                    if node_id is not None:
                        found.add((kind, node_id))
        return sorted(node for node in found if self.shares(customer, node))

    def sharers(self, node: Node) -> set[str]:
        """The customers sharing the node: none on a leaf. Every sharer is in the node's crowd,
        since a current link is a visible one and a payment that counts for sharing is one the
        crowd counts, so the crowd's customers are tested one by one."""
        return {customer for customer in self.crowd(node)[0] if self.shares(customer, node)}

    def sharing_start(self, customer: str, node: Node) -> int:
        """The customer's sharing start on the node: the earliest of its visible link starts
        and its first use from a deposit account, each t - 90 days when not known."""
        uses = self.uses(customer, node, deposit=True)
        starts = [
            *self.link_starts(customer, node[0]).get(node[1], []),
            *([min(uses)] if uses else []),
        ]
        return min(
            start if self.known(customer, node, start) else self.view.start for start in starts
        )

    def co_users(self, customer: str) -> dict[str, tuple[int, frozenset[str]]]:
        """The customer's co-users: each customer sharing a node with it, when the pair began
        sharing, and the kinds of node they share."""
        starts: dict[str, int] = {}
        kinds: dict[str, set[str]] = defaultdict(set)
        for node in self.shared_nodes(customer):
            own = self.sharing_start(customer, node)
            for other in self.sharers(node) - {customer}:
                began = max(own, self.sharing_start(other, node))
                starts[other] = min(starts.get(other, began), began)
                kinds[other].add(node[0])
        return {other: (began, frozenset(kinds[other])) for other, began in starts.items()}

    # Rows.

    def customer_row(self, customer: str) -> CustomerRow:
        """The customer's row."""
        t = self.view.t
        new = {
            kind: sum(
                1
                for node_id in self.link_starts(customer, kind)
                if (start := self.known_link_start(customer, (kind, node_id))) is not None
                and start >= t - CURRENT_MS
            )
            for kind in NODE_KINDS
        }
        co = self.co_users(customer)
        return CustomerRow(
            customer_id=customer,
            accounts_owned=len(self.accounts(customer)),
            devices_30d=len(self.current(customer, DEVICE)),
            ips_30d=len(self.current(customer, IP)),
            new_devices_30d=new[DEVICE],
            new_ips_30d=new[IP],
            co_users=len(co),
            new_co_users_30d=sum(1 for began, _ in co.values() if began >= t - CURRENT_MS),
            co_users_both=sum(1 for _, kinds in co.values() if len(kinds) == len(NODE_KINDS)),
        )

    def node_row(self, node: Node) -> NodeRow:
        """The node's row: its leaf flag alone on a leaf."""
        if self.leaf(node):
            return NodeRow(node[1], True, 0, 0, 0, 0, 0, 0, False)
        view = self.view
        customers, enrolled = self.crowd(node)
        sharers = self.sharers(node)
        starts = [self.sharing_start(customer, node) for customer in sharers]
        week, month = (view.t - window for window in NEW_WINDOWS_MS)
        recent = [
            payment
            for payment in view.seen(*node)
            if payment.ts >= self.recent and not payment.self_payment
        ]
        return NodeRow(
            node_id=node[1],
            leaf=False,
            customers_30d=len(sharers),
            customers_90d=len(customers),
            new_customers_7d=sum(1 for start in starts if start >= week),
            new_customers_30d=sum(1 for start in starts if start >= month),
            accounts_seen_30d=len({payment.sender for payment in recent}),
            payments_seen_30d=len(recent),
            observed_only=not enrolled,
        )

    def seen_edge(self, account_id: str, node: Node) -> Seen:
        """The seen edge of an account that sent a payment in the lookback seen on the node."""
        t = self.view.t
        times = [p.ts for p in self.lookback.sent(account_id) if seen_node(p, node[0]) == node[1]]
        first, last = min(times), max(times)
        owner = self.view.owner(account_id)
        known = owner is not None and self.known(owner, node, first)
        return Seen(account_id, node[1], len(times), known, t - first if known else 0, t - last)

    # Caps.

    def kept_links(self, customer: str | None) -> list[Node]:
        """The customer's kept links: up to NODES of each kind by their newest evidence."""
        if customer is None:
            return []
        kept: list[Node] = []
        for kind in NODE_KINDS:

            def evidence(node_id: str, kind: str = kind) -> int:
                node = (kind, node_id)
                start = self.known_link_start(customer, node) or 0
                return max([start, *self.uses(customer, node, deposit=False)])

            starts = self.link_starts(customer, kind)
            ordered = sorted(starts, key=lambda node_id: (-evidence(node_id), node_id))
            kept.extend((kind, node_id) for node_id in ordered[:NODES])
        return kept

    def kept_seen(self, account_id: str) -> list[Node]:
        """The account's kept seen nodes: up to NODES of each kind, latest use first."""
        kept: list[Node] = []
        for kind in NODE_KINDS:
            last: dict[str, int] = {}
            for payment in self.lookback.sent(account_id):
                node_id = seen_node(payment, kind)
                if node_id is not None:
                    last[node_id] = max(last.get(node_id, payment.ts), payment.ts)
            ordered = sorted(last, key=lambda node_id: (-last[node_id], node_id))
            kept.extend((kind, node_id) for node_id in ordered[:NODES])
        return kept

    def last_seen_on(self, account_id: str, nodes: Iterable[Node]) -> int | None:
        """The time of the account's latest payment sent in the lookback seen on one of the
        nodes."""
        wanted = set(nodes)
        return max(
            (
                payment.ts
                for payment in self.lookback.sent(account_id)
                for kind in NODE_KINDS
                if (kind, seen_node(payment, kind)) in wanted
            ),
            default=None,
        )


@dataclass(frozen=True)
class IdentityParts:
    """One scored account's device and IP graph: its customers, nodes and co-users' accounts,
    and its edges."""

    scored: str
    customers: frozenset[str]
    nodes: frozenset[Node]
    co_user_accounts: frozenset[str]
    enrolled: frozenset[Enrolled]
    owner_enrolled: frozenset[OwnerEnrolled]
    seen: frozenset[Seen]


def kept_co_users(sharing: Sharing, customer: str | None, nodes: Iterable[Node]) -> list[str]:
    """The kept co-users on the nodes, those of s that are no leaf: the customers of their
    crowds other than the scored account's own, the newest CO_USERS by evidence."""
    view = sharing.view
    evidence: dict[str, int] = {}
    for node in nodes:
        for other in sharing.crowd(node)[0] - {customer}:
            found = max(
                view.start,
                sharing.known_link_start(other, node) or 0,
                *sharing.uses(other, node, deposit=True),
            )
            evidence[other] = max(evidence.get(other, found), found)
    return sorted(evidence, key=lambda other: (-evidence[other], other))[:CO_USERS]


def brought_account(sharing: Sharing, customer: str, nodes: Iterable[Node]) -> str | None:
    """The deposit account a co-user brings: the one last seen sending on the nodes, else the
    one with the latest payment in the lookback, else one with none, ties by account id."""
    on = list(nodes)

    def order(account_id: str) -> tuple[int, int, str]:
        seen = sharing.last_seen_on(account_id, on)
        if seen is not None:
            return (0, -seen, account_id)
        latest = max((p.ts for p in sharing.lookback.both(account_id)), default=None)
        return (1, 0, account_id) if latest is None else (1, -latest, account_id)

    deposits = sorted(sharing.deposits(customer), key=order)
    return deposits[0] if deposits else None


def identity_graph(sharing: Sharing, scored: str, hubs: Collection[str]) -> IdentityParts:
    """The scored account's device and IP graph at t, with hubs the account hubs of t."""
    view = sharing.view
    payments = subgraph(sharing.lookback, scored, hubs)
    customer = payments.customer
    d_s = sharing.kept_links(customer)
    s_s = sharing.kept_seen(scored)
    own_nodes = [*d_s, *s_s]
    open_nodes = sorted({node for node in own_nodes if not sharing.leaf(node)})
    kept = kept_co_users(sharing, customer, open_nodes)
    brought = {
        other: account
        for other in kept
        if (account := brought_account(sharing, other, open_nodes)) is not None
    }
    customers = {customer} if customer is not None else set()
    nodes = set(own_nodes)
    for account in payments.expanded:
        owner = view.owner(account)
        if owner is not None:
            customers.add(owner)
        nodes.update(sharing.kept_links(owner), sharing.kept_seen(account))
    customers.update(
        other for other in kept if any(sharing.linked(other, node) for node in open_nodes)
    )
    # The accounts that may carry an edge: B4's and the co-users', but account hubs other
    # than s.
    edged = {
        account
        for account in (*payments.accounts, *brought.values())
        if account == scored or account not in hubs
    }

    def enrolment(other: str, node: Node) -> Enrolled:
        return Enrolled(other, node[1], *sharing.link_fields(other, node))

    def owner_enrolment(account: str, node: Node) -> OwnerEnrolled:
        owner = view.owner(account)
        assert owner is not None
        return OwnerEnrolled(account, node[1], *sharing.link_fields(owner, node))

    enrolled = {enrolment(customer, node) for node in d_s if customer is not None}
    owner_enrolled = {
        owner_enrolment(account, node)
        for account in edged
        if customer is not None and view.owner(account) == customer
        for node in d_s
    }
    seen = {sharing.seen_edge(scored, node) for node in s_s}
    for account in payments.expanded:
        owner = view.owner(account)
        owner_enrolled.update(owner_enrolment(account, node) for node in sharing.kept_links(owner))
        seen.update(sharing.seen_edge(account, node) for node in sharing.kept_seen(account))
    for node in open_nodes:
        enrolled.update(
            enrolment(other, node) for other in customers if sharing.linked(other, node)
        )
        for account in edged:
            owner = view.owner(account)
            if owner is not None and sharing.linked(owner, node):
                owner_enrolled.add(owner_enrolment(account, node))
            if any(seen_node(p, node[0]) == node[1] for p in sharing.lookback.sent(account)):
                seen.add(sharing.seen_edge(account, node))
    return IdentityParts(
        scored,
        frozenset(customers),
        frozenset(nodes),
        frozenset(brought.values()),
        frozenset(enrolled),
        frozenset(owner_enrolled),
        frozenset(seen),
    )


def _by_scored[Item](
    parts: list[IdentityParts], items: Callable[[IdentityParts], list[Item]]
) -> dict[str, list[Item]]:
    """A map of scored accounts to their sorted ids or edges of one kind, a scored account
    with none left out, as a TigerGraph MapAccum prints it."""
    return {part.scored: found for part in parts if (found := items(part))}


def _edges(edges: frozenset[Any]) -> list[dict[str, Any]]:
    """Edges of one kind sorted by their two ends, each as its printed fields."""
    return [asdict(edge) for edge in sorted(edges, key=_ends)]


def _ends(edge: DataclassInstance) -> tuple[Any, ...]:
    """An edge's two ends, its first two fields, by which edges are sorted."""
    return astuple(edge)[:2]


def read_shared_identity(
    graph: ReferenceGraph,
    t_ms: int,
    account_ids: list[str],
    hub_accounts: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """The device and IP graphs of the request's accounts at t: every customer, device, IP and
    co-user's account of them, each once and sorted by id, and per scored account, sorted, the
    ids of its customers, nodes and co-users' accounts and its three kinds of edge."""
    status = request_refusal(graph, t_ms, account_ids)
    if status is not None:
        return refused(status)
    hubs = hub_set(graph, t_ms, hub_accounts)
    if hubs is None:
        return refused(INVISIBLE_HUB)
    sharing = Sharing(Lookback(graph.as_of(t_ms)))
    parts = [identity_graph(sharing, scored, hubs) for scored in sorted(account_ids)]
    customers = sorted({customer for part in parts for customer in part.customers})
    nodes = sorted({node for part in parts for node in part.nodes})
    accounts = sorted({account for part in parts for account in part.co_user_accounts})
    return [
        head(READ_SHARED_IDENTITY),
        {"customers": vertices("Party", (sharing.customer_row(c) for c in customers))},
        {"devices": vertices(DEVICE, (sharing.node_row(n) for n in nodes if n[0] == DEVICE))},
        {"ips": vertices(IP, (sharing.node_row(n) for n in nodes if n[0] == IP))},
        {
            "accounts": vertices(
                "Account", (IdentityAccount(a, graph.accounts[a].is_external) for a in accounts)
            )
        },
        {"subgraph_customers": _by_scored(parts, lambda part: sorted(part.customers))},
        {
            "subgraph_nodes": _by_scored(
                parts, lambda part: sorted(node_id for _, node_id in part.nodes)
            )
        },
        {"co_user_accounts": _by_scored(parts, lambda part: sorted(part.co_user_accounts))},
        {"enrolled": _by_scored(parts, lambda part: _edges(part.enrolled))},
        {"owner_enrolled": _by_scored(parts, lambda part: _edges(part.owner_enrolled))},
        {"seen": _by_scored(parts, lambda part: _edges(part.seen))},
    ]
