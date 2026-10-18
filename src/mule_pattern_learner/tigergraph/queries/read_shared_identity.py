"""read_shared_identity, B5: the device and IP graph. For each account of a request at one
Monday, the customers, devices and IPs of its subgraph and its enrolment, owner-enrolment and
seen edges (docs/design.md, "read_shared_identity (B5): the device and IP graph", and
docs/features.md, "What one example's subgraph holds" and the customer, device, IP, enrolment
and seen tables)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...config import DEFAULT_CONFIG
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

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

SUBGRAPH = DEFAULT_CONFIG.subgraph


@dataclass(frozen=True)
class CustomerRow:
    """A customer's row at t (docs/features.md, "Customer node", columns 1 to 8).

    - accounts_owned: the accounts that exist at t with a visible ownership link to it, open
      or closed.
    - devices_30d, ips_30d: the devices and IPs it has a link to held in the last 30 days.
    - new_devices_30d, new_ips_30d: those whose link has a known start in the last 30 days.
    - co_users: the customers sharing a node with it that is no leaf; new_co_users_30d, those
      whose pair began sharing in the last 30 days; co_users_both, those sharing a device and
      an IP with it.
    """

    customer_id: str
    accounts_owned: int
    devices_30d: int
    ips_30d: int
    new_devices_30d: int
    new_ips_30d: int
    co_users: int
    new_co_users_30d: int
    co_users_both: int


@dataclass(frozen=True)
class NodeRow:
    """A device's or IP's row at t (docs/features.md, "Device and IP nodes", columns 1 to 7).
    A leaf has its flag alone and every other field 0: its measure and kind come from
    read_hubs.

    - customers_30d: the customers sharing it; customers_90d: the customers of the crowd rule,
      read_hubs' measure; new_customers_7d and _30d: those whose sharing start on it is in the
      last 7 or 30 days.
    - accounts_seen_30d, payments_seen_30d: the accounts seen sending payments on it, and the
      payments, in the last 30 days.
    - observed_only: no customer has a visible enrolment link to it.
    """

    node_id: str
    leaf: bool
    customers_30d: int
    customers_90d: int
    new_customers_7d: int
    new_customers_30d: int
    accounts_seen_30d: int
    payments_seen_30d: int
    observed_only: bool


@dataclass(frozen=True)
class IdentityAccount:
    """A co-user's account and whether it is another bank's."""

    account_id: str
    is_external: bool


@dataclass(frozen=True)
class Enrolled:
    """An enrolment edge, from a customer to a device or an IP (docs/features.md, "Enrolment
    edges"): the time since its link's known start, and since the latest payment seen on the
    node from an account the customer owns, each as presence and milliseconds."""

    customer_id: str
    node_id: str
    since_start_present: bool
    since_start_ms: int
    since_last_seen_present: bool
    since_last_seen_ms: int


@dataclass(frozen=True)
class OwnerEnrolled:
    """An owner-enrolment edge, from an account to a device or an IP: its customer's link,
    with the enrolment edge's fields."""

    account_id: str
    node_id: str
    since_start_present: bool
    since_start_ms: int
    since_last_seen_present: bool
    since_last_seen_ms: int


@dataclass(frozen=True)
class Seen:
    """A seen edge, from an account to a device or an IP (docs/features.md, "Seen-on-payment
    edges"): its payments sent in the lookback seen on the node, the time since the first
    when it is a known start, and since the last."""

    account_id: str
    node_id: str
    uses_90d: int
    since_first_use_present: bool
    since_first_use_ms: int
    since_last_use_ms: int


@dataclass(frozen=True)
class IdentityGraph:
    """One scored account's device and IP graph: its customers, devices, IPs and co-users'
    accounts, each sorted by id, and its edges, sorted by their ends."""

    scored_id: str
    customers: tuple[CustomerRow, ...]
    devices: tuple[NodeRow, ...]
    ips: tuple[NodeRow, ...]
    accounts: tuple[IdentityAccount, ...]
    enrolled: tuple[Enrolled, ...]
    owner_enrolled: tuple[OwnerEnrolled, ...]
    seen: tuple[Seen, ...]


def params(t_ms: int, accounts: Iterable[str], hubs: Iterable[str]) -> dict[str, Any]:
    """The parameters of a request: the Monday t, the accounts, at least one, each once, and
    the account hubs of t, read_hubs' accounts of the same Monday, each once."""
    return {
        "t_ms": as_of(t_ms),
        "account_ids": account_ids(accounts),
        "hub_accounts": vertex_set(hubs),
    }


def _refuse(what: object, why: str) -> ValueError:
    from . import READ_SHARED_IDENTITY  # the registry imports this module

    return ValueError(f"{READ_SHARED_IDENTITY} answered {why}: {what}")


def _check_customer(row: CustomerRow) -> None:
    """Refuse a customer row the query cannot give."""
    if row.new_devices_30d > row.devices_30d or row.new_ips_30d > row.ips_30d:
        raise _refuse(row, "more new links than links held in the last 30 days")
    if row.new_co_users_30d > row.co_users or row.co_users_both > row.co_users:
        raise _refuse(row, "more new co-users, or co-users of both kinds, than co-users")


def _check_node(row: NodeRow) -> None:
    """Refuse a device or IP row the query cannot give."""
    counts = (
        row.customers_30d,
        row.customers_90d,
        row.new_customers_7d,
        row.new_customers_30d,
        row.accounts_seen_30d,
        row.payments_seen_30d,
    )
    if row.leaf:
        if any(counts) or row.observed_only:
            raise _refuse(row, "a leaf with more than its flag")
        return
    if row.customers_90d > SUBGRAPH.crowd_threshold:
        raise _refuse(row, "a crowd that is no leaf")
    if not row.new_customers_7d <= row.new_customers_30d <= row.customers_30d:
        raise _refuse(row, "more new customers than customers sharing it")
    if row.customers_30d > row.customers_90d:
        raise _refuse(row, "more customers sharing it than customers")
    if row.accounts_seen_30d > row.payments_seen_30d:
        raise _refuse(row, "more accounts seen than payments")


def _since(present: bool, ms: int, what: object, name: str, *, below: bool) -> None:
    """Refuse a duration to t that is absent but not 0, or present but outside the lookback:
    a known start is after the lookback's start, so strictly below it."""
    top = LOOKBACK_MS - 1 if below else LOOKBACK_MS
    if (not present and ms) or (present and not 0 < ms <= top):
        raise _refuse(what, f"a {name} it cannot give")


def _check_link(edge: Enrolled | OwnerEnrolled) -> None:
    _since(edge.since_start_present, edge.since_start_ms, edge, "since_start", below=True)
    _since(
        edge.since_last_seen_present,
        edge.since_last_seen_ms,
        edge,
        "since_last_seen",
        below=False,
    )


def _check_seen(edge: Seen) -> None:
    if edge.uses_90d < 1:
        raise _refuse(edge, "a seen edge of no payment")
    _since(True, edge.since_last_use_ms, edge, "since_last_use", below=False)
    _since(
        edge.since_first_use_present,
        edge.since_first_use_ms,
        edge,
        "since_first_use",
        below=True,
    )
    if edge.since_first_use_present and edge.since_first_use_ms < edge.since_last_use_ms:
        raise _refuse(edge, "a first use after its last")


def _by_scored(rows: Sequence[object], key: str, asked: set[str]) -> dict[str, list[object]]:
    """A printed map of scored accounts to lists, each scored account one of those asked."""
    from . import READ_SHARED_IDENTITY  # the registry imports this module

    value = printed(rows, key, READ_SHARED_IDENTITY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{READ_SHARED_IDENTITY} printed {key} as {value!r}, not a map")
    found: Mapping[object, object] = value
    if not set(found) <= asked:
        raise ValueError(f"{READ_SHARED_IDENTITY} printed {key} of {sorted(map(str, found))}")
    lists: dict[str, list[object]] = {}
    for scored, listed in found.items():
        if not isinstance(listed, list):
            raise ValueError(f"{READ_SHARED_IDENTITY} printed {key} of {scored} as {listed!r}")
        items: list[object] = listed
        lists[str(scored)] = items
    return lists


def _rows[T: DataclassInstance](
    kind: type[T], rows: Sequence[object], key: str, id_of: Callable[[T], str]
) -> dict[str, T]:
    """The rows of a printed vertex set by id, each id once."""
    from . import READ_SHARED_IDENTITY  # the registry imports this module

    found = [parse_row(kind, row) for row in projected(rows, key, READ_SHARED_IDENTITY)]
    by_id = {id_of(row): row for row in found}
    if len(by_id) != len(found):
        raise ValueError(f"{READ_SHARED_IDENTITY} printed a row of {key} twice")
    return by_id


def _edges[T: DataclassInstance](
    kind: type[T], items: list[object], ends: Callable[[T], tuple[str, str]], scored: str
) -> tuple[T, ...]:
    """A scored account's edges of one kind, sorted by their ends, each pair of ends once."""
    from . import READ_SHARED_IDENTITY  # the registry imports this module

    found = sorted((parse_row(kind, item) for item in items), key=ends)
    if len({ends(edge) for edge in found}) != len(found):
        raise ValueError(f"{READ_SHARED_IDENTITY} gave {scored} an edge twice")
    return tuple(found)


def parse(rows: Sequence[object], accounts: Collection[str]) -> tuple[IdentityGraph, ...]:
    """The answer to the request params(t_ms, accounts, hubs) made: the device and IP graph
    of every account asked for, sorted by the scored account's id; one with no customer, node
    or edge is empty.

    Every edge of a subgraph joins its own customers, accounts and nodes: an enrolment edge
    one of its customers to one of its nodes; an owner-enrolment or seen edge one of its
    nodes to an account; every co-user's account is one printed. Every customer, node and
    account printed belongs to a subgraph.
    """
    from . import READ_SHARED_IDENTITY, VERSIONS  # the registry imports this module

    response_head(rows, READ_SHARED_IDENTITY, VERSIONS[READ_SHARED_IDENTITY])
    asked = set(accounts)
    customers = _rows(CustomerRow, rows, "customers", lambda row: row.customer_id)
    devices = _rows(NodeRow, rows, "devices", lambda row: row.node_id)
    ips = _rows(NodeRow, rows, "ips", lambda row: row.node_id)
    co_user_rows = _rows(IdentityAccount, rows, "accounts", lambda row: row.account_id)
    if devices.keys() & ips.keys():
        raise ValueError(f"{READ_SHARED_IDENTITY} printed a node as a device and an IP")
    for row in customers.values():
        _check_customer(row)
    for row in (*devices.values(), *ips.values()):
        _check_node(row)
    members = {
        key: {
            scored: ids(listed, f"{key} of {scored}")
            for scored, listed in _by_scored(rows, key, asked).items()
        }
        for key in ("subgraph_customers", "subgraph_nodes", "co_user_accounts")
    }
    edges = {key: _by_scored(rows, key, asked) for key in ("enrolled", "owner_enrolled", "seen")}
    graphs: list[IdentityGraph] = []
    for scored in sorted(asked):
        held = members["subgraph_customers"].get(scored, ())
        nodes = members["subgraph_nodes"].get(scored, ())
        co_users = members["co_user_accounts"].get(scored, ())
        if not set(held) <= customers.keys() or not set(co_users) <= co_user_rows.keys():
            raise ValueError(f"{READ_SHARED_IDENTITY} gave {scored} a customer or account it lacks")
        if not set(nodes) <= devices.keys() | ips.keys():
            raise ValueError(f"{READ_SHARED_IDENTITY} gave {scored} a node it lacks")
        enrolled = _edges(
            Enrolled,
            edges["enrolled"].get(scored, []),
            lambda edge: (edge.customer_id, edge.node_id),
            scored,
        )
        owner_enrolled = _edges(
            OwnerEnrolled,
            edges["owner_enrolled"].get(scored, []),
            lambda edge: (edge.account_id, edge.node_id),
            scored,
        )
        seen = _edges(
            Seen,
            edges["seen"].get(scored, []),
            lambda edge: (edge.account_id, edge.node_id),
            scored,
        )
        for edge in enrolled:
            if edge.customer_id not in held or edge.node_id not in nodes:
                raise _refuse(edge, f"an enrolment edge of {scored} with an end outside it")
            _check_link(edge)
        for edge in owner_enrolled:
            if edge.node_id not in nodes:
                raise _refuse(edge, f"an owner-enrolment edge of {scored} to a node outside it")
            _check_link(edge)
        for edge in seen:
            if edge.node_id not in nodes:
                raise _refuse(edge, f"a seen edge of {scored} to a node outside it")
            _check_seen(edge)
        graphs.append(
            IdentityGraph(
                scored,
                tuple(customers[customer] for customer in held),
                tuple(devices[node] for node in nodes if node in devices),
                tuple(ips[node] for node in nodes if node in ips),
                tuple(co_user_rows[account] for account in co_users),
                enrolled,
                owner_enrolled,
                seen,
            )
        )
    used = {
        key: {item for scored in asked for item in members[key].get(scored, ())} for key in members
    }
    if (
        used["subgraph_customers"] != customers.keys()
        or used["subgraph_nodes"] != devices.keys() | ips.keys()
        or used["co_user_accounts"] != co_user_rows.keys()
    ):
        raise ValueError(
            f"{READ_SHARED_IDENTITY} printed a customer, node or account of no subgraph"
        )
    return tuple(graphs)
