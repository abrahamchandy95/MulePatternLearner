"""The installed queries: their names, files and versions, the parameters of a request,
and their responses parsed into plain dataclasses.

docs/design.md is the contract: "Response format" under "The subgraph query" for
fetch_subgraphs, "Date tables" for read_leaves and read_sharing, and "Other queries" for
the rest. read_answer_key is parsed in data.answer_key, the only code that reads the
answer key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from ..config import SubgraphConfig
from .executor import QueryExecutor, checked_rows, printed

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

GRAPH_NAME = "Mule_Pattern_Learner"

FETCH_SUBGRAPHS = "fetch_subgraphs"
READ_LEAVES = "read_leaves"
READ_SHARING = "read_sharing"
LIST_POPULATION = "list_population"
READ_SUPERVISION = "read_supervision"
READ_WORLD = "read_world"
PROFILE_HUBS = "profile_hubs"
READ_ANSWER_KEY = "read_answer_key"

# Every query the code installs, and its file under gsql/.
QUERY_FILES = {
    FETCH_SUBGRAPHS: "queries/fetch_subgraphs.gsql",
    READ_LEAVES: "queries/read_leaves.gsql",
    READ_SHARING: "queries/read_sharing.gsql",
    LIST_POPULATION: "queries/list_population.gsql",
    READ_SUPERVISION: "queries/read_supervision.gsql",
    READ_WORLD: "queries/read_world.gsql",
    PROFILE_HUBS: "queries/profile_hubs.gsql",
    READ_ANSWER_KEY: "evaluation/read_answer_key.gsql",
}

# The version each query prints: its name and 12 hex digits of the hash of its text
# (installer.text_version). A test keeps the two equal, so a change to a query needs a new
# value here. The parity check records the versions that passed, the cache is keyed by
# them, and a model stores the one of fetch_subgraphs.
VERSIONS = {
    FETCH_SUBGRAPHS: "fetch_subgraphs-863c2712b2cb",
    READ_LEAVES: "read_leaves-8cb6142273f7",
    READ_SHARING: "read_sharing-74d20a32f238",
    LIST_POPULATION: "list_population-32bcf503f34d",
    READ_SUPERVISION: "read_supervision-599751cd6a84",
    READ_WORLD: "read_world-01bb14aa371d",
    PROFILE_HUBS: "profile_hubs-6a163fb0e896",
    READ_ANSWER_KEY: "read_answer_key-478d3430ad3c",
}
SUBGRAPH_VERSION = VERSIONS[FETCH_SUBGRAPHS]

# The category values (docs/design.md, "Features"), and how a discovered mule was found.
ACCOUNT_TYPES = ("deposit", "credit", "brokerage", "gl", "unknown")
RAILS = ("ach", "card", "cash", "check", "internal", "unknown", "zelle")
DISCOVERY_CHANNELS = frozenset({"victim_report", "monitoring", "trace"})

# The SubgraphConfig fields read_leaves takes; fetch_subgraphs takes every other one.
LEAF_SETTINGS = ("hub_threshold", "observed_threshold")


def parse_row[T: DataclassInstance](
    kind: type[T], row: object, *, signed: frozenset[str] = frozenset()
) -> T:
    """A response row as the dataclass kind, whose fields name the row's keys exactly.

    A value must have its field's type (a float field also takes an integer), and an
    integer must not be negative unless its field is in signed.
    """
    name = kind.__name__
    if not isinstance(row, Mapping):
        raise ValueError(f"A {name} row is {row!r}, not an object")
    expected = [field.name for field in fields(kind)]
    if set(row) != set(expected):
        raise ValueError(f"A {name} row has the fields {sorted(row)}, not {sorted(expected)}")
    values: dict[str, Any] = {}
    for field in fields(kind):
        value = row[field.name]
        ok = {
            "str": isinstance(value, str),
            "bool": isinstance(value, bool),
            "int": isinstance(value, int)
            and not isinstance(value, bool)
            and (value >= 0 or field.name in signed),
            "float": isinstance(value, int | float) and not isinstance(value, bool),
        }[str(field.type)]
        if not ok:
            raise ValueError(f"{name}.{field.name} is {value!r}, not a valid {field.type}")
        values[field.name] = float(value) if field.type == "float" else value
    return kind(**values)


def response_head(rows: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """The first row of a successful response of query.

    A response of another version is refused: the installed query is not the one this code
    reads, so install the queries again.
    """
    head = checked_rows(rows)[0]
    if head.get("version") != VERSIONS[query]:
        raise ValueError(
            f"{query} is version {head.get('version')!r}, not {VERSIONS[query]}: run `mule install`"
        )
    return head


def _ids(value: object, what: str) -> tuple[str, ...]:
    """A printed list of distinct ids, sorted."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{what} is {value!r}, not a list of ids")
    ids: list[str] = value
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} repeats an id")
    return tuple(sorted(ids))


def _vertices(ids: Iterable[str]) -> list[dict[str, str]]:
    """A SET<VERTEX<T>> parameter, as TigerGraph's JSON takes it."""
    return [{"id": vertex_id} for vertex_id in ids]


def _as_of(as_of_ms: int) -> int:
    if as_of_ms <= 0:
        raise ValueError("A query needs a positive as-of time")
    return as_of_ms


# read_leaves.


@dataclass(frozen=True)
class Leaves:
    """The leaves of one date (docs/design.md, "Hubs, crowds and leaves"), each list sorted."""

    hub_accounts: tuple[str, ...]
    hub_devices: tuple[str, ...]
    crowd_devices: tuple[str, ...]
    hub_ips: tuple[str, ...]
    crowd_ips: tuple[str, ...]


def leaves_params(as_of_ms: int, config: SubgraphConfig) -> dict[str, Any]:
    return {
        "as_of_ms": _as_of(as_of_ms),
        "lookback_days": config.lookback_days,
        "hub_threshold": config.hub_threshold,
        "observed_threshold": config.observed_threshold,
    }


def parse_leaves(rows: list[dict[str, Any]]) -> Leaves:
    head = response_head(rows, READ_LEAVES)
    names = [field.name for field in fields(Leaves)]
    if set(head) != {"status", "version", *names}:
        raise ValueError(f"read_leaves printed the fields {sorted(head)}")
    leaves = Leaves(*(_ids(head[name], name) for name in names))
    if set(leaves.hub_devices) & set(leaves.crowd_devices) or set(leaves.hub_ips) & set(
        leaves.crowd_ips
    ):
        raise ValueError("read_leaves returned a node that is both a hub and a crowd")
    return leaves


# read_sharing.


@dataclass(frozen=True)
class SharingRow:
    """A customer's co-users on devices and on IPs, and how many of them are new."""

    party_id: str
    device_co_users: int
    ip_co_users: int
    new_device_co_users_30d: int
    new_ip_co_users_30d: int


@dataclass(frozen=True)
class SharingPage:
    """One page of read_sharing: the page's last party id, "" for an empty page, and the
    rows of its customers with a co-user."""

    last_id: str
    rows: tuple[SharingRow, ...]


def sharing_params(as_of_ms: int, config: SubgraphConfig, leaves: Leaves) -> dict[str, Any]:
    """The read_sharing parameters but the page (sharing_rows adds it)."""
    return {
        "as_of_ms": _as_of(as_of_ms),
        "lookback_days": config.lookback_days,
        "hub_devices": _vertices(leaves.hub_devices),
        "crowd_devices": _vertices(leaves.crowd_devices),
        "hub_ips": _vertices(leaves.hub_ips),
        "crowd_ips": _vertices(leaves.crowd_ips),
    }


def parse_sharing(rows: list[dict[str, Any]], after_id: str) -> SharingPage:
    """A read_sharing page that began after after_id."""
    head = response_head(rows, READ_SHARING)
    if set(head) != {"status", "version", "last_id", "rows"}:
        raise ValueError(f"read_sharing printed the fields {sorted(head)}")
    last_id, items = head["last_id"], head["rows"]
    if not isinstance(last_id, str) or not isinstance(items, list):
        raise ValueError("read_sharing printed a malformed page")
    page = SharingPage(last_id, tuple(parse_row(SharingRow, item) for item in items))
    if last_id and last_id <= after_id:
        raise ValueError("read_sharing pagination is not strictly increasing")
    ids = [row.party_id for row in page.rows]
    if len(set(ids)) != len(ids) or not all(after_id < party <= last_id for party in ids):
        raise ValueError("read_sharing returned a customer outside its page")
    for row in page.rows:
        if (
            row.device_co_users + row.ip_co_users == 0
            or row.new_device_co_users_30d > row.device_co_users
            or row.new_ip_co_users_30d > row.ip_co_users
        ):
            raise ValueError(f"read_sharing returned invalid co-users for {row.party_id}")
    return page


def sharing_rows(
    executor: QueryExecutor,
    params: dict[str, Any],
    *,
    page_size: int = 10_000,
    timeout_s: float | None = None,
) -> list[SharingRow]:
    """Every customer's co-user row of a date, reading pages until one holds no customer.

    A page of customers without a co-user has no rows but goes on.
    """
    after, found = "", list[SharingRow]()
    while True:
        rows = executor.run(
            READ_SHARING,
            {**params, "after_id": after, "batch_size": page_size},
            timeout_s=timeout_s,
        )
        page = parse_sharing(rows, after)
        if not page.last_id:
            return found
        found.extend(page.rows)
        after = page.last_id


# fetch_subgraphs.


@dataclass(frozen=True)
class SubgraphRequest:
    """Seed accounts at one as-of time."""

    seed_ids: tuple[str, ...]
    as_of_ms: int

    def __post_init__(self) -> None:
        if not self.seed_ids or len(set(self.seed_ids)) != len(self.seed_ids):
            raise ValueError("A subgraph request needs distinct seed accounts")
        if self.as_of_ms <= 0:
            raise ValueError("A subgraph request needs a positive as-of time")


def subgraph_params(
    request: SubgraphRequest, config: SubgraphConfig, leaves: Leaves
) -> dict[str, Any]:
    """The fetch_subgraphs parameters: the request, every setting but the leaf thresholds,
    and the date's leaves."""
    settings = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.name not in LEAF_SETTINGS
    }
    sets = {field.name: _vertices(getattr(leaves, field.name)) for field in fields(Leaves)}
    return {"seed_ids": list(request.seed_ids), "as_of_ms": request.as_of_ms, **settings, **sets}


@dataclass(frozen=True)
class AccountNode:
    """An account; owner_id ("" with none) is for the join, not a feature."""

    id: str
    owner_id: str
    account_type: str
    is_external: bool
    hub: bool
    in_7d: int
    in_30d: int
    in_365d: int
    out_7d: int
    out_30d: int
    out_365d: int
    amount_in_7d: int
    amount_in_30d: int
    amount_out_7d: int
    amount_out_30d: int
    counterparties_in_30d: int
    counterparties_in_365d: int
    counterparties_out_30d: int
    counterparties_out_365d: int
    new_payers_7d: int
    new_payers_30d: int
    new_payees_7d: int
    new_payees_30d: int
    burst_in_1h: int
    burst_in_24h: int
    burst_out_1h: int
    burst_out_24h: int
    burst_new_payers_24h: int
    passed_on_1h: int
    passed_on_24h: int
    shortest_in_gap_present: bool
    shortest_in_gap_ms: int
    last_in_present: bool
    since_last_in_ms: int
    last_out_present: bool
    since_last_out_ms: int


@dataclass(frozen=True)
class PartyNode:
    id: str
    accounts: int
    devices_now: int
    ips_now: int
    new_devices_7d: int
    new_devices_30d: int
    new_ips_7d: int
    new_ips_30d: int


@dataclass(frozen=True)
class IdentityNode:
    """A Device or IP."""

    id: str
    leaf: bool
    observed_only: bool
    parties_now: int
    parties_lookback: int
    new_parties_7d: int
    new_parties_30d: int
    accounts: int
    payments_7d: int
    payments_30d: int
    senders_30d: int


@dataclass(frozen=True)
class CommunityRow:
    """The community block of a seed; density and isolation are shares."""

    parties: int
    capped: bool
    nodes: int
    links: int
    new_links_30d: int
    accounts: int
    internal_payments_30d: int
    internal_amount_30d: int
    external_payments_30d: int
    density: float
    isolation: float


@dataclass(frozen=True)
class PaymentEdge:
    """One payment, from its sender to its receiver; amount in cents."""

    id: str
    from_id: str
    to_id: str
    amount: int
    rail: str
    age_ms: int
    prior_count: int
    first_in_pair: bool
    gap_present: bool
    gap_ms: int
    pair_age_ms: int
    next_outflow_present: bool
    next_outflow_ms: int
    sender_gap_present: bool
    sender_gap_ms: int
    receiver_gap_present: bool
    receiver_gap_ms: int


@dataclass(frozen=True)
class OwnedByEdge:
    """An account's ownership, from the account to its owner."""

    from_id: str
    to_id: str


@dataclass(frozen=True)
class LinkEdge:
    """An account's owner's link to a device or IP, from the account."""

    from_id: str
    to_id: str
    start_known: bool
    since_start_ms: int
    held_ms: int
    active: bool
    since_end_ms: int


@dataclass(frozen=True)
class SentEdge:
    """A device or IP observed on an account's own payments, from the account."""

    from_id: str
    to_id: str
    since_first_ms: int
    since_last_ms: int
    uses: int


@dataclass(frozen=True)
class Work:
    """What a request read, for the bench; the reference graph does not mirror it."""

    payments_read: int
    identity_nodes_read: int
    community_candidates: int


@dataclass(frozen=True)
class Subgraph:
    """One seed's subgraph: its nodes by type, its community block, its edges by relation."""

    seed: str
    accounts: tuple[AccountNode, ...]
    parties: tuple[PartyNode, ...]
    devices: tuple[IdentityNode, ...]
    ips: tuple[IdentityNode, ...]
    community: CommunityRow
    pays: tuple[PaymentEdge, ...]
    owned_by: tuple[OwnedByEdge, ...]
    enrolled_on_device: tuple[LinkEdge, ...]
    enrolled_on_ip: tuple[LinkEdge, ...]
    sent_from_device: tuple[SentEdge, ...]
    sent_from_ip: tuple[SentEdge, ...]


# Each node list's row type, and each relation's row type and its end node lists.
NODES: dict[str, type[AccountNode | PartyNode | IdentityNode]] = {
    "accounts": AccountNode,
    "parties": PartyNode,
    "devices": IdentityNode,
    "ips": IdentityNode,
}
RELATIONS: dict[str, tuple[type[PaymentEdge | OwnedByEdge | LinkEdge | SentEdge], str, str]] = {
    "pays": (PaymentEdge, "accounts", "accounts"),
    "owned_by": (OwnedByEdge, "accounts", "parties"),
    "enrolled_on_device": (LinkEdge, "accounts", "devices"),
    "enrolled_on_ip": (LinkEdge, "accounts", "ips"),
    "sent_from_device": (SentEdge, "accounts", "devices"),
    "sent_from_ip": (SentEdge, "accounts", "ips"),
}


@dataclass(frozen=True)
class SubgraphBatch:
    """A fetch_subgraphs response: the query's version, what it read, one subgraph per seed."""

    version: str
    as_of_ms: int
    work: Work
    subgraphs: tuple[Subgraph, ...]


def _list(row: Mapping[str, Any], key: str) -> list[Any]:
    items = row[key]
    if not isinstance(items, list):
        raise ValueError(f"The subgraph of {row.get('seed')!r} has {key} {items!r}, not a list")
    return items


def _community(value: object) -> CommunityRow:
    community = parse_row(CommunityRow, value)
    if not (0 <= community.density <= 1 and 0 <= community.isolation <= 1):
        raise ValueError(f"A community block has a share outside [0, 1]: {community}")
    return community


def _subgraph(row: Mapping[str, Any]) -> Subgraph:
    seed = row.get("seed")
    if set(row) != {"seed", *NODES, "community", *RELATIONS}:
        raise ValueError(f"The subgraph of {seed!r} has the fields {sorted(row)}")
    parsed: dict[str, Any] = {"seed": seed, "community": _community(row["community"])}
    ids: dict[str, set[str]] = {}
    for key, kind in NODES.items():
        nodes = tuple(parse_row(kind, item) for item in _list(row, key))
        ids[key] = {node.id for node in nodes}
        if len(ids[key]) != len(nodes):
            raise ValueError(f"The subgraph of {seed!r} repeats a node of {key}")
        parsed[key] = nodes
    if seed not in ids["accounts"]:
        raise ValueError(f"The subgraph of {seed!r} does not hold its seed")
    for account in parsed["accounts"]:
        if account.account_type not in ACCOUNT_TYPES:
            raise ValueError(f"{account.id} has the account type {account.account_type!r}")
    for key, (kind, start, end) in RELATIONS.items():
        edges = tuple(parse_row(kind, item) for item in _list(row, key))
        for edge in edges:
            if edge.from_id not in ids[start] or edge.to_id not in ids[end]:
                raise ValueError(f"The subgraph of {seed!r} has a {key} edge without its nodes")
        parsed[key] = edges
    for payment in parsed["pays"]:
        if payment.rail not in RAILS:
            raise ValueError(f"Payment {payment.id} has the rail {payment.rail!r}")
    return Subgraph(**parsed)


def parse_subgraphs(rows: list[dict[str, Any]], request: SubgraphRequest) -> SubgraphBatch:
    """The subgraphs of a fetch_subgraphs response to request."""
    head = response_head(rows, FETCH_SUBGRAPHS)
    if head.get("as_of_ms") != request.as_of_ms:
        raise ValueError("fetch_subgraphs answered another as-of time")
    if [row.get("seed") for row in rows[1:]] != list(request.seed_ids):
        raise ValueError("fetch_subgraphs did not return one subgraph per seed, in order")
    work = parse_row(Work, head.get("work"))
    subgraphs = tuple(_subgraph(row) for row in rows[1:])
    return SubgraphBatch(SUBGRAPH_VERSION, request.as_of_ms, work, subgraphs)


# list_population and read_supervision.


def population_params(as_of_ms: int) -> dict[str, Any]:
    """The list_population parameters but the page (executor.account_pages adds it)."""
    return {"as_of_ms": _as_of(as_of_ms)}


@dataclass(frozen=True)
class PopulationAccount:
    """An account of a date's population; opened_ms is for evaluation's groups only."""

    account_id: str
    owner_id: str
    account_type: str
    opened_ms: int


def parse_population(page: Sequence[Mapping[str, Any]]) -> list[PopulationAccount]:
    """A page of list_population (as executor.account_pages yields it)."""
    accounts = [parse_row(PopulationAccount, row) for row in page]
    for account in accounts:
        if not account.owner_id or account.account_type not in ACCOUNT_TYPES:
            raise ValueError(f"list_population returned {account}, not an owned account")
    return accounts


def supervision_params(known_by_ms: int) -> dict[str, Any]:
    if known_by_ms <= 0:
        raise ValueError("Supervision needs a positive date it is known by")
    return {"known_by_ms": known_by_ms}


@dataclass(frozen=True)
class Supervision:
    """What a bank knows of a mule it discovered: when and how, and its activity's dates."""

    account_id: str
    discovered_ms: int
    discovery_channel: str
    mule_onset_ms: int
    mule_last_ms: int


def parse_supervision(rows: list[dict[str, Any]], known_by_ms: int) -> list[Supervision]:
    """The read_supervision response to known_by_ms; a mule discovered after it is refused,
    as one the query must not have returned."""
    response_head(rows, READ_SUPERVISION)
    found = [
        parse_row(Supervision, item.get("attributes", item) if isinstance(item, Mapping) else item)
        for item in printed(rows, "accounts")
    ]
    for row in found:
        if not 0 < row.discovered_ms <= known_by_ms:
            raise ValueError(f"read_supervision returned {row.account_id}, not discovered in time")
        if row.discovery_channel not in DISCOVERY_CHANNELS:
            raise ValueError(f"{row.account_id} was discovered by {row.discovery_channel!r}")
    return found


# read_world and profile_hubs.


@dataclass(frozen=True)
class World:
    """What identifies a loaded world (docs/design.md, "Extraction and caching"): counts,
    sums in whole seconds (keyed "Type.attribute"), sums of amounts in cents, squared
    degrees per edge type, and the times that bound the data. All are integers."""

    vertices: dict[str, int]
    edges: dict[str, int]
    account_types: dict[str, int]
    payment_rails: dict[str, int]
    external_accounts: int
    seconds: dict[str, int]
    cents: dict[str, int]
    squared_degrees: dict[str, int]
    first_joined_ms: int
    first_event_ms: int
    last_event_ms: int


def _counts(value: object, what: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(count, int) and not isinstance(count, bool)
        for key, count in value.items()
    ):
        raise ValueError(f"read_world printed {what} {value!r}, not integers by name")
    counts: Mapping[str, int] = value
    return dict(sorted(counts.items()))


def parse_world(rows: list[dict[str, Any]]) -> World:
    head = response_head(rows, READ_WORLD)
    names = [field.name for field in fields(World)]
    if set(head) != {"status", "version", *names}:
        raise ValueError(f"read_world printed the fields {sorted(head)}")
    values: dict[str, Any] = {}
    for field in fields(World):
        value = head[field.name]
        if field.type == "int":
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"read_world printed {field.name} {value!r}")
            values[field.name] = value
        else:
            values[field.name] = _counts(value, field.name)
    return World(**values)


@dataclass(frozen=True)
class HubProfile:
    """profile_hubs's histograms: bucket j counts the measures from buckets[j] up to the
    next bound, the last bucket unbounded."""

    buckets: tuple[int, ...]
    accounts: tuple[int, ...]
    enrolled_devices: tuple[int, ...]
    observed_devices: tuple[int, ...]
    enrolled_ips: tuple[int, ...]
    observed_ips: tuple[int, ...]


def profile_params(as_of_ms: int, config: SubgraphConfig, buckets: Sequence[int]) -> dict[str, Any]:
    if not buckets or buckets[0] < 0 or any(b <= a for a, b in zip(buckets, buckets[1:])):
        raise ValueError(f"Buckets must be ascending bounds from 0 or more, got {buckets!r}")
    return {
        "as_of_ms": _as_of(as_of_ms),
        "lookback_days": config.lookback_days,
        "buckets": list(buckets),
    }


def parse_hub_profile(rows: list[dict[str, Any]], buckets: Sequence[int]) -> HubProfile:
    head = response_head(rows, PROFILE_HUBS)
    names = [field.name for field in fields(HubProfile)]
    if set(head) != {"status", "version", *names}:
        raise ValueError(f"profile_hubs printed the fields {sorted(head)}")
    values: dict[str, tuple[int, ...]] = {}
    for name in names:
        counts = head[name]
        if (
            not isinstance(counts, list)
            or len(counts) != len(buckets)
            or not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in counts)
        ):
            raise ValueError(f"profile_hubs printed {name} {counts!r}")
        values[name] = tuple(counts)
    if list(values["buckets"]) != list(buckets):
        raise ValueError("profile_hubs answered other buckets")
    return HubProfile(**values)
