"""The installed queries: their names and files, the subgraph query's version, the
parameters of a request, and their responses parsed into plain dataclasses.

docs/design.md, "Response format" under "The subgraph query", is the format of
fetch_subgraphs. read_answer_key is parsed in evaluation.answer_key, the only code that
reads the answer key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..config import SubgraphConfig
from .executor import checked_rows, printed

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

GRAPH_NAME = "Mule_Pattern_Learner"

FETCH_SUBGRAPHS = "fetch_subgraphs"
LIST_POPULATION = "list_population"
READ_SUPERVISION = "read_supervision"
WRITE_SPLIT = "write_split"
READ_ANSWER_KEY = "read_answer_key"

# Every query the code installs, and its file under gsql/.
QUERY_FILES = {
    FETCH_SUBGRAPHS: "queries/fetch_subgraphs.gsql",
    LIST_POPULATION: "queries/list_population.gsql",
    READ_SUPERVISION: "queries/read_supervision.gsql",
    WRITE_SPLIT: "queries/write_split.gsql",
    READ_ANSWER_KEY: "evaluation/read_answer_key.gsql",
}

# The version fetch_subgraphs prints: "subgraphs-" and 12 hex digits of the hash of its
# text (installer.text_version). A test keeps the two equal, so a change to the query needs
# a new value here; a model stores it, and scoring refuses a model built for another.
SUBGRAPH_VERSION = "subgraphs-40d794a3cd10"

# How a discovered mule was found.
DISCOVERY_CHANNELS = frozenset({"victim_report", "monitoring", "trace"})


class Phase(StrEnum):
    """Which customers a query sees."""

    # Training and validation: the held-out customers (split 1) are invisible.
    TRAINING = "training"
    # Test and scoring: every customer.
    TEST = "test"

    @property
    def held_out_visible(self) -> bool:
        return self is Phase.TEST


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


# fetch_subgraphs.


@dataclass(frozen=True)
class SubgraphRequest:
    """Seed accounts at one as-of time, seen in one phase."""

    seed_ids: tuple[str, ...]
    as_of_ms: int
    phase: Phase

    def __post_init__(self) -> None:
        if not self.seed_ids or len(set(self.seed_ids)) != len(self.seed_ids):
            raise ValueError("A subgraph request needs distinct seed accounts")
        if self.as_of_ms <= 0:
            raise ValueError("A subgraph request needs a positive as-of time")


def subgraph_params(request: SubgraphRequest, config: SubgraphConfig) -> dict[str, Any]:
    """The fetch_subgraphs parameters: the request, and every setting of config."""
    caps = {field.name: getattr(config, field.name) for field in fields(config)}
    return {
        "seed_ids": list(request.seed_ids),
        "as_of_ms": request.as_of_ms,
        "held_out_visible": request.phase.held_out_visible,
        **caps,
    }


@dataclass(frozen=True)
class AccountNode:
    id: str
    account_type: str
    is_external: bool
    age_ms: int
    token_present: bool
    token_age_ms: int
    in_7d: int
    in_30d: int
    in_365d: int
    out_7d: int
    out_30d: int
    out_365d: int
    amount_in_30d: float
    amount_out_30d: float
    counterparties_in_30d: int
    counterparties_out_30d: int
    first_inflows_7d: int
    first_inflows_30d: int
    passed_on_30d: float
    last_in_present: bool
    since_last_in_ms: int
    last_out_present: bool
    since_last_out_ms: int
    closed: bool


@dataclass(frozen=True)
class PartyNode:
    id: str
    age_ms: int
    open_accounts: int
    closed_accounts: int


@dataclass(frozen=True)
class IdentityNode:
    """A Device, IP or Address."""

    id: str
    age_ms: int
    parties_now: int
    parties_ever: int
    accounts: int
    new_parties_30d: int
    observed_only: bool
    hub: bool


@dataclass(frozen=True)
class PaymentEdge:
    """One payment, from its sender to its receiver."""

    id: str
    from_id: str
    to_id: str
    amount: float
    rail: str
    age_ms: int
    gap_present: bool
    gap_ms: int
    pair_age_ms: int
    prior_count: int
    first_in_pair: bool
    next_outflow_present: bool
    next_outflow_ms: int


@dataclass(frozen=True)
class LinkEdge:
    """An ownership or identity link, from an account."""

    from_id: str
    to_id: str
    since_start_ms: int
    active: bool
    since_end_ms: int
    held_ms: int


@dataclass(frozen=True)
class SentEdge:
    """A device or IP observed on an account's own payments, from the account."""

    from_id: str
    to_id: str
    since_first_ms: int
    since_last_ms: int
    uses: int


@dataclass(frozen=True)
class Subgraph:
    """One seed's subgraph: its nodes by type and its edges by relation."""

    seed: str
    accounts: tuple[AccountNode, ...]
    parties: tuple[PartyNode, ...]
    devices: tuple[IdentityNode, ...]
    ips: tuple[IdentityNode, ...]
    addresses: tuple[IdentityNode, ...]
    pays: tuple[PaymentEdge, ...]
    owned_by: tuple[LinkEdge, ...]
    enrolled_on_device: tuple[LinkEdge, ...]
    enrolled_on_ip: tuple[LinkEdge, ...]
    at_address: tuple[LinkEdge, ...]
    sent_from_device: tuple[SentEdge, ...]
    sent_from_ip: tuple[SentEdge, ...]


# Each node list's row type, and each relation's row type and its end node lists.
NODES: dict[str, type[AccountNode | PartyNode | IdentityNode]] = {
    "accounts": AccountNode,
    "parties": PartyNode,
    "devices": IdentityNode,
    "ips": IdentityNode,
    "addresses": IdentityNode,
}
RELATIONS: dict[str, tuple[type[PaymentEdge | LinkEdge | SentEdge], str, str]] = {
    "pays": (PaymentEdge, "accounts", "accounts"),
    "owned_by": (LinkEdge, "accounts", "parties"),
    "enrolled_on_device": (LinkEdge, "accounts", "devices"),
    "enrolled_on_ip": (LinkEdge, "accounts", "ips"),
    "at_address": (LinkEdge, "accounts", "addresses"),
    "sent_from_device": (SentEdge, "accounts", "devices"),
    "sent_from_ip": (SentEdge, "accounts", "ips"),
}


@dataclass(frozen=True)
class SubgraphBatch:
    """A fetch_subgraphs response: the query's version and one subgraph per seed."""

    version: str
    as_of_ms: int
    phase: Phase
    subgraphs: tuple[Subgraph, ...]


def _subgraph(row: Mapping[str, Any]) -> Subgraph:
    seed = row.get("seed")
    if set(row) != {"seed", *NODES, *RELATIONS}:
        raise ValueError(f"The subgraph of {seed!r} has the fields {sorted(row)}")
    parsed: dict[str, Any] = {"seed": seed}
    ids: dict[str, set[str]] = {}
    for key, kind in NODES.items():
        items = row[key]
        if not isinstance(items, list):
            raise ValueError(f"The subgraph of {seed!r} has {key} {items!r}, not a list")
        nodes = tuple(parse_row(kind, item) for item in items)
        ids[key] = {node.id for node in nodes}
        if len(ids[key]) != len(nodes):
            raise ValueError(f"The subgraph of {seed!r} repeats a node of {key}")
        parsed[key] = nodes
    if seed not in ids["accounts"]:
        raise ValueError(f"The subgraph of {seed!r} does not hold its seed")
    for key, (kind, start, end) in RELATIONS.items():
        items = row[key]
        if not isinstance(items, list):
            raise ValueError(f"The subgraph of {seed!r} has {key} {items!r}, not a list")
        edges = tuple(parse_row(kind, item) for item in items)
        for edge in edges:
            if edge.from_id not in ids[start] or edge.to_id not in ids[end]:
                raise ValueError(f"The subgraph of {seed!r} has a {key} edge without its nodes")
        parsed[key] = edges
    return Subgraph(**parsed)


def parse_subgraphs(rows: list[dict[str, Any]], request: SubgraphRequest) -> SubgraphBatch:
    """The subgraphs of a fetch_subgraphs response to request.

    A response of another version is refused, as the contract lock asks: the installed
    query is not the one this code reads, so install the queries again.
    """
    head = checked_rows(rows)[0]
    if head.get("version") != SUBGRAPH_VERSION:
        raise ValueError(
            f"fetch_subgraphs is version {head.get('version')!r}, not {SUBGRAPH_VERSION}: "
            "run `mule install`"
        )
    if (head.get("as_of_ms"), head.get("held_out_visible")) != (
        request.as_of_ms,
        request.phase.held_out_visible,
    ):
        raise ValueError("fetch_subgraphs answered another as-of time or phase")
    if [row.get("seed") for row in rows[1:]] != list(request.seed_ids):
        raise ValueError("fetch_subgraphs did not return one subgraph per seed, in order")
    subgraphs = tuple(_subgraph(row) for row in rows[1:])
    return SubgraphBatch(SUBGRAPH_VERSION, request.as_of_ms, request.phase, subgraphs)


# list_population, read_supervision and write_split.


def population_params(as_of_ms: int, phase: Phase) -> dict[str, Any]:
    """The list_population parameters but the page (executor.account_pages adds it)."""
    if as_of_ms <= 0:
        raise ValueError("The population needs a positive as-of time")
    return {"as_of_ms": as_of_ms, "held_out_visible": phase.held_out_visible}


@dataclass(frozen=True)
class PopulationAccount:
    account_id: str
    account_type: str
    opened_ms: int
    split: int


def parse_population(page: Sequence[Mapping[str, Any]]) -> list[PopulationAccount]:
    """A page of list_population (as executor.account_pages yields it)."""
    accounts = [parse_row(PopulationAccount, row) for row in page]
    if any(account.split not in (0, 1) for account in accounts):
        raise ValueError("list_population returned a split other than 0 or 1")
    return accounts


def supervision_params(cutoff_ms: int, phase: Phase) -> dict[str, Any]:
    if cutoff_ms <= 0:
        raise ValueError("Supervision needs a positive cutoff")
    return {"cutoff_ms": cutoff_ms, "held_out_visible": phase.held_out_visible}


@dataclass(frozen=True)
class Supervision:
    """What a bank knows of a mule it discovered."""

    account_id: str
    discovered_ms: int
    discovery_channel: str
    mule_onset_ms: int


def parse_supervision(rows: list[dict[str, Any]], cutoff_ms: int) -> list[Supervision]:
    """The read_supervision response to cutoff_ms; a mule discovered at or after it is
    refused, as one the query must not have returned."""
    found = [
        parse_row(Supervision, item.get("attributes", item) if isinstance(item, Mapping) else item)
        for item in printed(checked_rows(rows), "accounts")
    ]
    for row in found:
        if not 0 < row.discovered_ms < cutoff_ms:
            raise ValueError(f"read_supervision returned {row.account_id}, not discovered in time")
        if row.discovery_channel not in DISCOVERY_CHANNELS:
            raise ValueError(f"{row.account_id} was discovered by {row.discovery_channel!r}")
    return found


def split_params(splits: Mapping[str, int]) -> dict[str, Any]:
    """The write_split parameters: each party's split, 0 (training) or 1 (held out)."""
    if not splits or any(split not in (0, 1) for split in splits.values()):
        raise ValueError("write_split needs parties, each with split 0 or 1")
    return {"party_ids": list(splits), "splits": list(splits.values())}


@dataclass(frozen=True)
class SplitWritten:
    """How many parties, and accounts they own, write_split wrote."""

    parties: int
    accounts: int


def parse_split(rows: list[dict[str, Any]]) -> SplitWritten:
    head = checked_rows(rows)[0]
    return parse_row(SplitWritten, {key: head.get(key) for key in ("parties", "accounts")})
