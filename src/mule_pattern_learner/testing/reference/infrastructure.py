"""The reference answers of the infrastructure queries, written from docs/design.md ("The
scored population", "Shared rules", and the sections of list_accounts, read_labels,
read_answer_key, read_hubs, read_world and profile_hubs), apart from their GSQL, and of
mark_crowds, which prints the crowds of read_hubs.

Each query's rows are computed as the dataclasses its parser reads, then printed as TigerGraph
prints its PRINT statements: a status row with the version first; a vertex set's projection as
a list of vertices, each with its projected fields under "attributes"; a map with a key for
each type or value that has a row, and its integer keys as strings. A refusal is the status
row the query prints, alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ...config import DEFAULT_CONFIG
from ...tigergraph.queries import (
    LIST_ACCOUNTS,
    MARK_CROWDS,
    PROFILE_HUBS,
    READ_ANSWER_KEY,
    READ_HUBS,
    READ_LABELS,
    READ_WORLD,
)
from ...tigergraph.queries import read_world as world_rows
from ...tigergraph.queries.list_accounts import ScorableAccount
from ...tigergraph.queries.mark_crowds import Crowd, Crowds
from ...tigergraph.queries.profile_hubs import HubProfile
from ...tigergraph.queries.read_answer_key import AccountTruth
from ...tigergraph.queries.read_hubs import HubAccount, Hubs, Leaf
from ...tigergraph.queries.read_labels import IdentifiedMule
from ...tigergraph.queries.read_world import PAYMENT_TYPES, SECONDS, World
from ...tigergraph.world import identity
from .answer import INVALID_PARAMETERS, head, refused, valid_page, valid_time, vertices
from .graph import BY_NAME, DEVICE, EDGE_TABLES, IP, NODE_KINDS, VERTEX_TABLES, Account, AsOf, cents

if TYPE_CHECKING:
    from .graph import ReferenceGraph

SUBGRAPH = DEFAULT_CONFIG.subgraph
INVALID_ACTIVITY = "invalid_activity"
# What TigerGraph's MinAccum<UINT> holds before it reads a value: the largest UINT.
UINT_MAX = 2**64 - 1


# list_accounts.


def scorable(view: AsOf, account: Account) -> bool:
    """An internal deposit account open at t with a visible ownership link (docs/design.md,
    "The scored population")."""
    return (
        not account.is_external
        and account.account_type == "deposit"
        and view.open(account.id)
        and view.owner(account.id) is not None
    )


def list_accounts(
    graph: ReferenceGraph, t_ms: int, after_id: str, page: int
) -> list[dict[str, Any]]:
    """A page of the scorable accounts of t, each with its owner."""
    if not valid_time(t_ms) or not valid_page(page):
        return refused(INVALID_PARAMETERS)
    view = graph.as_of(t_ms)
    rows = [
        ScorableAccount(account.id, str(view.owner(account.id)))
        for account in graph.page(after_id, page, lambda account: scorable(view, account))
    ]
    return [head(LIST_ACCOUNTS), {"accounts": vertices("Account", rows)}]


# read_labels and read_answer_key.


def read_labels(graph: ReferenceGraph, after_id: str, page: int) -> list[dict[str, Any]]:
    """A page of the identified mules, pu_label 1, with their activity period. The page is
    refused, naming its accounts, when one of them lacks 0 < mule_onset_ms <= mule_last_ms."""
    if not valid_page(page):
        return refused(INVALID_PARAMETERS)
    identified = graph.page(after_id, page, lambda account: account.pu_label == 1)
    invalid = [
        account.id
        for account in identified
        if not 0 < account.mule_onset_ms <= account.mule_last_ms
    ]
    if invalid:
        return refused(INVALID_ACTIVITY, accounts=invalid)
    rows = [
        IdentifiedMule(account.id, account.mule_onset_ms, account.mule_last_ms)
        for account in identified
    ]
    return [head(READ_LABELS), {"accounts": vertices("Account", rows)}]


def read_answer_key(graph: ReferenceGraph, after_id: str, page: int) -> list[dict[str, Any]]:
    """A page of every internal account's truth, and whether it has an ownership link at
    all."""
    if not valid_page(page):
        return refused(INVALID_PARAMETERS)
    rows = [
        AccountTruth(
            account.id,
            account.is_mule,
            account.mule_ring_id,
            account.mule_onset_ms,
            account.mule_last_ms,
            account.opened_ms,
            account.account_type,
            account.id in graph.ownership,
        )
        for account in graph.page(after_id, page, lambda account: not account.is_external)
    ]
    return [head(READ_ANSWER_KEY), {"accounts": vertices("Account", rows)}]


# read_hubs, mark_crowds and profile_hubs.


def hub_customers(view: AsOf, account_id: str) -> set[str]:
    """The account hub measure: the distinct customers that own, through a visible ownership
    link, the other accounts of the account's payments in the lookback, each of which exists
    at t. An ownerless counterparty is not counted, and a self-payment has no other account."""
    customers: set[str] = set()
    for payment in view.payments(account_id):
        for other in (payment.sender, payment.receiver):
            owner = view.owner(other) if other != account_id and view.exists(other) else None
            if owner is not None:
                customers.add(owner)
    return customers


def crowd_customers(view: AsOf, kind: str, node: str) -> tuple[set[str], set[str]]:
    """The crowd measure of a device or IP and its registered customers. The measure counts
    the distinct customers with a visible enrolment link to it, and by the one account rule the
    owners, through a visible ownership link, of deposit accounts that exist at t and sent a
    payment seen on it in the lookback. A self-payment counts here, and nowhere else."""
    enrolled = {link.source for link in view.enrolled(kind, node)}
    customers = set(enrolled)
    for payment in view.seen(kind, node):
        sender = view.graph.accounts[payment.sender]
        owner = view.owner(sender.id) if view.exists(sender.id) else None
        if sender.account_type == "deposit" and owner is not None:
            customers.add(owner)
    return customers, enrolled


def crowds(graph: ReferenceGraph, t_ms: int) -> Crowds:
    """The crowds of t: the devices and IPs with more than the crowd threshold of customers by
    the crowd measure, each a hub too above the hub threshold."""
    view = graph.as_of(t_ms)
    found: dict[str, tuple[Crowd, ...]] = {}
    for kind in NODE_KINDS:
        nodes: list[Crowd] = []
        for node in graph.ids[kind]:
            customers, enrolled = crowd_customers(view, kind, node)
            if len(customers) > SUBGRAPH.crowd_threshold:
                hub = len(customers) > SUBGRAPH.hub_threshold
                nodes.append(Crowd(node, len(customers), not enrolled, hub))
        found[kind] = tuple(nodes)
    return Crowds(found[DEVICE], found[IP])


def hubs(graph: ReferenceGraph, t_ms: int) -> Hubs:
    """The leaves of t: the account hubs, and the device and IP crowds. The query skips the
    account test for an account with at most 200 payments in all time, which cannot pass it:
    each payment adds at most one customer. So this graph tests every account that exists at
    t, with the same answer."""
    view = graph.as_of(t_ms)
    accounts: list[HubAccount] = []
    for account_id in graph.ids["Account"]:
        if view.exists(account_id):
            customers = len(hub_customers(view, account_id))
            if customers > SUBGRAPH.hub_threshold:
                accounts.append(HubAccount(account_id, customers))
    found = crowds(graph, t_ms)

    def leaves(nodes: tuple[Crowd, ...]) -> tuple[Leaf, ...]:
        return tuple(Leaf(**asdict(node)) for node in nodes)

    return Hubs(tuple(accounts), leaves(found.devices), leaves(found.ips))


def read_hubs(graph: ReferenceGraph, t_ms: int) -> list[dict[str, Any]]:
    """The leaves of t with their measures."""
    if not valid_time(t_ms):
        return refused(INVALID_PARAMETERS)
    found = hubs(graph, t_ms)
    return [
        head(READ_HUBS),
        {"accounts": vertices("Account", found.accounts)},
        {"devices": vertices(DEVICE, found.devices)},
        {"ips": vertices(IP, found.ips)},
    ]


def mark_crowds(graph: ReferenceGraph, t_ms: int) -> list[dict[str, Any]]:
    """The crowds of t with their measures. No stamp is kept: no read of this graph tests one."""
    if not valid_time(t_ms):
        return refused(INVALID_PARAMETERS)
    found = crowds(graph, t_ms)
    return [
        head(MARK_CROWDS),
        {"devices": vertices(DEVICE, found.devices)},
        {"ips": vertices(IP, found.ips)},
    ]


def hub_profile(graph: ReferenceGraph, t_ms: int) -> HubProfile:
    """The histograms of t: every account that exists at t by the account hub measure, and
    every device and IP with a customer by the crowd measure, registered when a customer has a
    visible enrolment link to it and observed-only otherwise."""
    view = graph.as_of(t_ms)
    accounts = Counter(
        len(hub_customers(view, account_id))
        for account_id in graph.ids["Account"]
        if view.exists(account_id)
    )
    nodes: dict[tuple[str, bool], Counter[int]] = {
        (kind, registered): Counter() for kind in NODE_KINDS for registered in (True, False)
    }
    for kind in NODE_KINDS:
        for node in graph.ids[kind]:
            customers, enrolled = crowd_customers(view, kind, node)
            if customers:
                nodes[kind, bool(enrolled)][len(customers)] += 1

    def histogram(counts: Counter[int]) -> dict[int, int]:
        return dict(sorted(counts.items()))

    return HubProfile(
        histogram(accounts),
        histogram(nodes[DEVICE, True]),
        histogram(nodes[DEVICE, False]),
        histogram(nodes[IP, True]),
        histogram(nodes[IP, False]),
    )


def profile_hubs(graph: ReferenceGraph, t_ms: int) -> list[dict[str, Any]]:
    """The histograms of t, each measure printed as a string, as TigerGraph prints a map's
    integer keys."""
    if not valid_time(t_ms):
        return refused(INVALID_PARAMETERS)
    profile = asdict(hub_profile(graph, t_ms))
    printed = {
        name: {str(measure): count for measure, count in histogram.items()}
        for name, histogram in profile.items()
    }
    return [{**head(PROFILE_HUBS), **printed}]


# read_world.


def read_world(graph: ReferenceGraph) -> list[dict[str, Any]]:
    """What identifies the world (docs/design.md, "Cache"), all integers. Each forward edge
    type is counted from its rows: a link of one pair repeated with another start is another
    edge, and a reverse edge type, its forward type read from the other end, is not printed."""
    rows = graph.rows
    edges: dict[str, int] = {}
    squared: dict[str, int] = {}
    squared_target: dict[str, int] = {}
    products: dict[str, int] = {}
    for table in EDGE_TABLES:
        pairs = [(row[0], row[1]) for row in rows[table.name]]
        if not pairs:
            continue
        out = Counter(source for source, _ in pairs)
        into = Counter(target for _, target in pairs)
        edges[table.name] = len(pairs)
        squared[table.name] = sum(degree * degree for degree in out.values())
        squared_target[table.name] = sum(degree * degree for degree in into.values())
        products[table.name] = sum(out[source] * into[target] for source, target in pairs)
    seconds: dict[str, int] = {}
    for key in SECONDS:
        kind, column = key.split(".")
        if rows[kind]:
            index = BY_NAME[kind].column_names.index(column)
            seconds[key] = sum(row[index] // 1000 for row in rows[kind])
    events = [row[-1] for kind in PAYMENT_TYPES for row in rows[kind]]
    printed = {
        "vertices": {
            table.name: len(rows[table.name]) for table in VERTEX_TABLES if rows[table.name]
        },
        "edges": edges,
        "account_types": dict(Counter(row[1] for row in rows["Account"])),
        "payment_rails": dict(Counter(row[2] for row in rows["Payment_Transaction"])),
        "external_accounts": sum(row[2] for row in rows["Account"]),
        "seconds": seconds,
        "cents": {
            kind: sum(cents(row[1]) for row in rows[kind]) for kind in PAYMENT_TYPES if rows[kind]
        },
        "squared_degrees": squared,
        "squared_target_degrees": squared_target,
        "degree_products": products,
        "first_joined_ms": min((row[1] for row in rows["Party"]), default=UINT_MAX),
        "first_event_ms": min(events, default=UINT_MAX),
        "last_event_ms": max(events, default=0),
    }
    return [{**head(READ_WORLD), **printed}]


def world(graph: ReferenceGraph) -> World:
    """The world's identity fields, as read_world's parser reads them from this graph."""
    return world_rows.parse(read_world(graph))


def world_identity(graph: ReferenceGraph) -> str:
    """The world's identity: the hash that names its cache and store folders."""
    return identity(world(graph))
