"""An in-memory graph that answers the installed queries as TigerGraph does: the test oracle.

It holds PhantomLedger's 27 tables (use case mule-temporal) and the split write_split
writes, and runs each query of gsql/ from its specification: fetch_subgraphs from
docs/design.md ("The subgraph query" and its "Response format"), the others from their
files. ReferenceGraph.run has the signature of tigergraph.executor.QueryExecutor.run and
returns the rows TigerGraph prints, so code that reads a live graph reads this one in tests,
and tests/testing/test_parity.py checks a live graph against it.

Lists in a response are unordered; this graph sorts them so its answers are stable. It
reads a whole export into memory, so it suits small worlds.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import astuple, dataclass, field, fields, replace
from functools import cache
import heapq
import math
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from ..tigergraph.installer import parameter_names, repository_queries
from ..tigergraph.queries import (
    FETCH_SUBGRAPHS,
    GRAPH_NAME,
    LIST_POPULATION,
    NODES,
    READ_ANSWER_KEY,
    READ_SUPERVISION,
    RELATIONS,
    SUBGRAPH_VERSION,
    WRITE_SPLIT,
    LinkEdge,
    PaymentEdge,
    SubgraphBatch,
)

DAY_MS = 86_400_000

# PhantomLedger's tables in its export order: the vertex tables with their columns, then the
# association tables (links) and the event tables, each with the types of its two ends.
VERTICES: dict[str, tuple[str, ...]] = {
    "Party": ("id", "joined_ms"),
    "Account": (
        "id",
        "account_type",
        "is_external",
        "opened_ms",
        "closed_ms",
        "is_mule",
        "mule_ring_id",
        "mule_layer",
        "mule_flow",
        "mule_onset_ms",
        "mule_last_ms",
        "discovered_ms",
        "discovery_channel",
    ),
    "Token": ("id",),
    "Device": ("id",),
    "IP": ("id",),
    "Address": ("id",),
    "Payment_Transaction": ("id", "amount", "payment_rail", "event_ts_ms"),
    "Zelle_Transfer": ("id", "amount", "event_ts_ms"),
}
LINKS: dict[str, tuple[str, str]] = {
    "Party_Owns_Account": ("Party", "Account"),
    "Party_Uses_Token": ("Party", "Token"),
    "Token_Bound_To_Account": ("Token", "Account"),
    "Party_Uses_Device": ("Party", "Device"),
    "Account_Uses_Device": ("Account", "Device"),
    "Party_Uses_IP": ("Party", "IP"),
    "Party_Has_Address": ("Party", "Address"),
}
EVENTS: dict[str, tuple[str, str]] = {
    f"{prefix}_{role}": (payment, end)
    for prefix, payment in (("Transfer", "Zelle_Transfer"), ("Transaction", "Payment_Transaction"))
    for role, end in (
        ("From_Account", "Account"),
        ("To_Account", "Account"),
        ("From_Token", "Token"),
        ("To_Token", "Token"),
        ("Used_Device", "Device"),
        ("Used_IP", "IP"),
    )
}
TABLES: dict[str, tuple[str, ...]] = {
    **VERTICES,
    **{name: ("from_id", "to_id", "valid_from_ms", "valid_to_ms") for name in LINKS},
    **{name: ("from_id", "to_id", "event_ts_ms") for name in EVENTS},
}
_INTEGERS = frozenset(
    {
        "joined_ms",
        "opened_ms",
        "closed_ms",
        "is_mule",
        "mule_ring_id",
        "mule_onset_ms",
        "mule_last_ms",
        "discovered_ms",
        "event_ts_ms",
        "valid_from_ms",
        "valid_to_ms",
    }
)


@dataclass(frozen=True, slots=True)
class Account:
    """An Account row, in the table's column order."""

    id: str
    account_type: str
    is_external: bool
    opened_ms: int
    closed_ms: int
    is_mule: int
    mule_ring_id: int
    mule_layer: str
    mule_flow: str
    mule_onset_ms: int
    mule_last_ms: int
    discovered_ms: int
    discovery_channel: str


@dataclass(frozen=True, slots=True)
class Link:
    """A row of an association table: source to target over [start, end), end 0 while open."""

    source: str
    target: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Payment:
    """A Payment_Transaction or Zelle_Transfer with its sender, its receiver and the devices
    and IPs observed on it. rail is the payment_rail, or "zelle"."""

    id: str
    amount: float
    rail: str
    ts: int
    sender: str
    receiver: str
    devices: tuple[str, ...]
    ips: tuple[str, ...]


@dataclass(frozen=True)
class _Kind:
    """An identity node type: its node list, its party link table, the relation of an
    account's link through its owner and the setting that caps it, and for devices and IPs
    the relation of a node observed on an account's payments and its cap."""

    nodes: str
    links: str
    enrolled: str
    enrolled_cap: str
    sent: str = ""
    sent_cap: str = ""


DEVICES = _Kind(
    "devices",
    "Party_Uses_Device",
    "enrolled_on_device",
    "enrolled_devices",
    "sent_from_device",
    "sent_devices",
)
IPS = _Kind("ips", "Party_Uses_IP", "enrolled_on_ip", "enrolled_ips", "sent_from_ip", "sent_ips")
ADDRESSES = _Kind("addresses", "Party_Has_Address", "at_address", "addresses")
KINDS = (DEVICES, IPS, ADDRESSES)


def _observed_on(payment: Payment, kind: _Kind) -> tuple[str, ...]:
    if kind is DEVICES:
        return payment.devices
    return payment.ips if kind is IPS else ()


def _cell(column: str, value: Any) -> str | int | float | bool:
    """A table's value as its attribute's type. A boolean may be written True, true or 1."""
    if column in _INTEGERS:
        return int(value)
    if column == "amount":
        return float(value)
    if column == "is_external":
        text = str(value).lower()
        if text not in ("true", "false", "1", "0"):
            raise ValueError(f"is_external is {value!r}, not a boolean")
        return text in ("true", "1")
    return str(value)


def _rows(name: str, frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    columns = TABLES[name]
    if tuple(frame.columns) != columns:
        raise ValueError(f"{name} has the columns {list(frame.columns)}, not {list(columns)}")
    return [
        tuple(_cell(column, value) for column, value in zip(columns, row, strict=True))
        for row in frame.itertuples(index=False, name=None)
    ]


def _grouped[V](items: Iterable[V], key: Callable[[V], str]) -> dict[str, list[V]]:
    grouped: dict[str, list[V]] = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return dict(grouped)


def _newest[V](items: Iterable[V], cap: int, key: Callable[[V], Any]) -> list[V]:
    """The first cap items in key order; every caller orders newest first, ties by id."""
    return heapq.nsmallest(cap, items, key=key)


def _usage(uses: Iterable[tuple[str, int]]) -> dict[str, tuple[int, int, int]]:
    """Per key, its first and last time and its number of times."""
    found: dict[str, tuple[int, int, int]] = {}
    for key, ts in uses:
        first, last, count = found.get(key, (ts, ts, 0))
        found[key] = (min(first, ts), max(last, ts), count + 1)
    return found


def _vertex(account_id: str, attributes: dict[str, Any]) -> dict[str, Any]:
    """An account of a printed vertex set projection, as TigerGraph prints it."""
    return {"v_id": account_id, "v_type": "Account", "attributes": attributes}


@cache
def _parameters() -> dict[str, list[str]]:
    """Each installed query's parameters, from its file under gsql/."""
    return {name: parameter_names(text) for name, text in repository_queries().items()}


class ReferenceGraph:
    """The graph of one export, with the splits written to it and the installed queries.

    Every party and account starts with split -1, as on a freshly loaded graph. It refuses
    tables TigerGraph would load into another graph than the export describes: a missing
    or repeated vertex, an edge to an unknown vertex, a repeated link of one pair and start,
    and a payment without exactly one sender and receiver or whose edges give another time.
    """

    graph_name = GRAPH_NAME

    def __init__(self, tables: Mapping[str, pd.DataFrame]) -> None:
        if set(tables) != set(TABLES):
            raise ValueError(f"The graph needs the tables {sorted(TABLES)}, not {sorted(tables)}")
        rows = {name: _rows(name, tables[name]) for name in TABLES}
        ids: dict[str, set[str]] = {}
        for name in VERTICES:
            ids[name] = {row[0] for row in rows[name]}
            if len(ids[name]) != len(rows[name]):
                raise ValueError(f"{name} repeats an id")
        self.joined_ms: dict[str, int] = {row[0]: row[1] for row in rows["Party"]}
        self.accounts: dict[str, Account] = {row[0]: Account(*row) for row in rows["Account"]}
        self.party_split = dict.fromkeys(self.joined_ms, -1)
        self.account_split = dict.fromkeys(self.accounts, -1)
        self._account_ids = sorted(self.accounts)
        links = {name: self._links(name, rows[name], ids) for name in LINKS}
        ownership = links["Party_Owns_Account"]
        self.owners_of = _grouped(ownership, lambda link: link.target)
        self.owned_by = _grouped(ownership, lambda link: link.source)
        self.tokens_of = _grouped(links["Token_Bound_To_Account"], lambda link: link.target)
        self.party_links = {
            kind.nodes: _grouped(links[kind.links], lambda link: link.source) for kind in KINDS
        }
        self.node_links = {
            kind.nodes: _grouped(links[kind.links], lambda link: link.target) for kind in KINDS
        }
        self.payments = self._payments(rows, ids)
        ordered = sorted(self.payments.values(), key=lambda payment: (payment.ts, payment.id))
        self.sent_by = _grouped(ordered, lambda payment: payment.sender)
        self.received_by = _grouped(ordered, lambda payment: payment.receiver)
        self.observed: dict[str, dict[str, list[Payment]]] = {}
        for kind in KINDS:
            uses = [(node, payment) for payment in ordered for node in _observed_on(payment, kind)]
            self.observed[kind.nodes] = {
                node: [payment for _, payment in group]
                for node, group in _grouped(uses, lambda use: use[0]).items()
            }

    @classmethod
    def from_csv(cls, directory: Path) -> ReferenceGraph:
        """The graph of an export directory: one <table>.csv per table, with its header line."""
        return cls(
            {
                name: pd.read_csv(directory / f"{name}.csv", dtype=str, keep_default_na=False)
                for name in TABLES
            }
        )

    @staticmethod
    def _links(name: str, rows: list[tuple[Any, ...]], ids: dict[str, set[str]]) -> list[Link]:
        source_type, target_type = LINKS[name]
        links = [Link(*row) for row in rows]
        for link in links:
            if link.source not in ids[source_type] or link.target not in ids[target_type]:
                raise ValueError(f"{name} links {link.source} to {link.target}, an unknown vertex")
        if len({(link.source, link.target, link.start) for link in links}) != len(links):
            raise ValueError(f"{name} repeats a link of one pair and start")
        return links

    @staticmethod
    def _payments(
        rows: dict[str, list[tuple[Any, ...]]], ids: dict[str, set[str]]
    ) -> dict[str, Payment]:
        found = {row[0]: (row[1], row[2], row[3]) for row in rows["Payment_Transaction"]}
        transfers = {row[0]: (row[1], "zelle", row[2]) for row in rows["Zelle_Transfer"]}
        if found.keys() & transfers.keys():
            raise ValueError("A Payment_Transaction and a Zelle_Transfer share an id")
        found.update(transfers)
        ends: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for name, (payment_type, end_type) in EVENTS.items():
            role = name.split("_", 1)[1]
            for payment_id, end, ts in rows[name]:
                if payment_id not in ids[payment_type] or end not in ids[end_type]:
                    raise ValueError(f"{name} joins {payment_id} to {end}, an unknown vertex")
                if ts != found[payment_id][2]:
                    raise ValueError(f"{name} gives {payment_id} another time than its own")
                ends[payment_id][role].append(end)
        payments: dict[str, Payment] = {}
        for payment_id, (amount, rail, ts) in found.items():
            role = ends[payment_id]
            if len(role["From_Account"]) != 1 or len(role["To_Account"]) != 1:
                raise ValueError(f"{payment_id} needs one sender and one receiver")
            payments[payment_id] = Payment(
                payment_id,
                amount,
                rail,
                ts,
                role["From_Account"][0],
                role["To_Account"][0],
                tuple(sorted(set(role["Used_Device"]))),
                tuple(sorted(set(role["Used_IP"]))),
            )
        return payments

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        """The rows the installed query name prints for params. Parameters other than the
        query's are refused, as TigerGraph refuses them; the keywords are ignored."""
        del timeout_s, attempts, timeout_retries
        queries: dict[str, Callable[..., list[dict[str, Any]]]] = {
            FETCH_SUBGRAPHS: self.fetch_subgraphs,
            LIST_POPULATION: self.list_population,
            READ_SUPERVISION: self.read_supervision,
            WRITE_SPLIT: self.write_split,
            READ_ANSWER_KEY: self.read_answer_key,
        }
        if name not in queries:
            raise ValueError(f"No query {name} is installed")
        expected = _parameters()[name]
        if sorted(params) != sorted(expected):
            raise ValueError(f"{name} takes the parameters {expected}, not {sorted(params)}")
        return queries[name](**params)

    def fetch_subgraphs(
        self, seed_ids: list[str], as_of_ms: int, held_out_visible: bool, **settings: int
    ) -> list[dict[str, Any]]:
        """Each seed's subgraph as of as_of_ms, as docs/design.md describes it; settings are
        the caps, hub_threshold and lookback_days. A seed's subgraph does not depend on the
        other seeds of the request."""
        lookback = settings["lookback_days"]
        caps = [value for name, value in settings.items() if name != "lookback_days"]
        if (
            not seed_ids
            or min(caps) < 1
            or not 30 <= lookback <= 3650
            or as_of_ms < max(365, lookback) * DAY_MS
        ):
            return [{"status": "invalid_parameters"}]
        if len(set(seed_ids)) != len(seed_ids):
            return [{"status": "duplicate_seed"}]
        unknown = sorted(set(seed_ids) - self.accounts.keys())
        if unknown:
            return [{"status": "unknown_seed", "seeds": unknown}]
        view = _AsOf(self, as_of_ms, held_out_visible, settings)
        invisible = sorted(
            seed
            for seed in seed_ids
            if self.accounts[seed].opened_ms > as_of_ms or not view.account_visible(seed)
        )
        if invisible:
            return [{"status": "invisible_seed", "seeds": invisible}]
        head = {
            "status": "ok",
            "version": SUBGRAPH_VERSION,
            "as_of_ms": as_of_ms,
            "held_out_visible": held_out_visible,
        }
        return [head, *(view.subgraph(seed) for seed in seed_ids)]

    def _customer(self, account_id: str, held_out_visible: bool) -> bool:
        """Whether the account belongs to a customer the phase sees (split 0, or 1 too)."""
        return self.account_split[account_id] in ((0, 1) if held_out_visible else (0,))

    def _page(
        self, after_id: str, batch_size: int, keep: Callable[[Account], bool]
    ) -> list[Account]:
        """Up to batch_size accounts kept by keep, by id after after_id."""
        page: list[Account] = []
        for account_id in self._account_ids[bisect_right(self._account_ids, after_id) :]:
            if len(page) == batch_size:
                break
            if keep(self.accounts[account_id]):
                page.append(self.accounts[account_id])
        return page

    def list_population(
        self, as_of_ms: int, held_out_visible: bool, after_id: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """A page of the accounts open at as_of_ms of the customers the phase sees."""
        if as_of_ms <= 0 or not 1 <= batch_size <= 10_000:
            return [{"status": "invalid_parameters"}]

        def open_customer(account: Account) -> bool:
            return (
                self._customer(account.id, held_out_visible)
                and account.opened_ms <= as_of_ms
                and (account.closed_ms == 0 or account.closed_ms > as_of_ms)
            )

        page = self._page(after_id, batch_size, open_customer)
        printed = [
            _vertex(
                account.id,
                {
                    "account_id": account.id,
                    "account_type": account.account_type,
                    "opened_ms": account.opened_ms,
                    "split": self.account_split[account.id],
                },
            )
            for account in page
        ]
        return [{"status": "ok"}, {"accounts": printed}]

    def read_supervision(self, cutoff_ms: int, held_out_visible: bool) -> list[dict[str, Any]]:
        """The phase's customers' accounts discovered before cutoff_ms, with their supervision."""
        if cutoff_ms <= 0:
            return [{"status": "invalid_parameters"}]
        printed = [
            _vertex(
                account.id,
                {
                    "account_id": account.id,
                    "discovered_ms": account.discovered_ms,
                    "discovery_channel": account.discovery_channel,
                    "mule_onset_ms": account.mule_onset_ms,
                },
            )
            for account in (self.accounts[account_id] for account_id in self._account_ids)
            if 0 < account.discovered_ms < cutoff_ms
            and self._customer(account.id, held_out_visible)
        ]
        return [{"status": "ok"}, {"accounts": printed}]

    def write_split(self, party_ids: list[str], splits: list[int]) -> list[dict[str, Any]]:
        """Writes each party's split on it and on every account it owns: an account owned by
        several of the parties takes the largest of their splits."""
        if (
            not party_ids
            or len(splits) != len(party_ids)
            or any(split not in (0, 1) for split in splits)
            or len(set(party_ids)) != len(party_ids)
        ):
            return [{"status": "invalid_parameters"}]
        if any(party_id not in self.joined_ms for party_id in party_ids):
            return [{"status": "unknown_party"}]
        written: dict[str, int] = {}
        for party_id, split in zip(party_ids, splits, strict=True):
            self.party_split[party_id] = split
            for link in self.owned_by.get(party_id, []):
                written[link.target] = max(written.get(link.target, split), split)
        self.account_split.update(written)
        return [{"status": "ok", "parties": len(party_ids), "accounts": len(written)}]

    def read_answer_key(self, after_id: str, batch_size: int) -> list[dict[str, Any]]:
        """A page of every account's truth, by id after after_id."""
        if not 1 <= batch_size <= 10_000:
            return [{"status": "invalid_parameters"}]
        printed = [
            _vertex(
                account.id,
                {
                    "account_id": account.id,
                    "is_mule": account.is_mule,
                    "mule_ring_id": account.mule_ring_id,
                    "mule_layer": account.mule_layer,
                    "mule_flow": account.mule_flow,
                    "mule_onset_ms": account.mule_onset_ms,
                    "mule_last_ms": account.mule_last_ms,
                    "discovered_ms": account.discovered_ms,
                },
            )
            for account in self._page(after_id, batch_size, lambda _: True)
        ]
        return [{"status": "ok"}, {"accounts": printed}]


class _LinkEdge(NamedTuple):
    relation: str
    from_id: str
    to_id: str
    start: int
    end: int


class _SentEdge(NamedTuple):
    relation: str
    from_id: str
    to_id: str
    first: int
    last: int
    uses: int


class _Candidate(NamedTuple):
    """An account an identity node of a seed may add, in the order candidates are taken:
    newest first (minus the link start or the last use), then by account, node and relation.
    party is the linked party, "" for an account observed sending from the node; first and
    last are the link's start and end, or the first and last use."""

    newest: int
    account: str
    node: str
    relation: str
    party: str
    first: int
    last: int
    uses: int


class _Reach(NamedTuple):
    """What expanding an account reached: the other accounts of its payments, the owners it
    picked and the identity nodes it picked, by node list."""

    counterparties: set[str]
    owners: set[str]
    nodes: dict[str, set[str]]


@dataclass
class _Subgraph:
    """One seed's nodes and edges as they are gathered."""

    accounts: set[str]
    parties: set[str] = field(default_factory=set[str])
    nodes: dict[str, set[str]] = field(
        default_factory=lambda: {kind.nodes: set[str]() for kind in KINDS}
    )
    pays: set[str] = field(default_factory=set[str])
    links: set[_LinkEdge] = field(default_factory=set[_LinkEdge])
    sent: set[_SentEdge] = field(default_factory=set[_SentEdge])


def _active(end: int, t: int) -> bool:
    return end == 0 or end > t


class _AsOf:
    """One fetch_subgraphs request's view: its time t, its payment window, the customers its
    phase sees and its settings. It caches what the seeds of the request share.

    A payment is visible when window start <= event_ts_ms < t and both its accounts are; a
    link when valid_from_ms <= t and its party is, however long ago it ended.
    """

    def __init__(
        self, graph: ReferenceGraph, t: int, everyone: bool, settings: Mapping[str, int]
    ) -> None:
        self.graph, self.t, self.everyone, self.settings = graph, t, everyone, settings
        self.start = t - settings["lookback_days"] * DAY_MS
        self._sent: dict[str, list[Payment]] = {}
        self._received: dict[str, list[Payment]] = {}
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}

    # Visibility.

    def account_visible(self, account_id: str) -> bool:
        return self.everyone or self.graph.account_split[account_id] != 1

    def party_visible(self, party_id: str) -> bool:
        return self.everyone or self.graph.party_split[party_id] != 1

    def shows(self, payment: Payment) -> bool:
        return (
            self.start <= payment.ts < self.t
            and self.account_visible(payment.sender)
            and self.account_visible(payment.receiver)
        )

    def sent(self, account_id: str) -> list[Payment]:
        """The account's visible sent payments, oldest first."""
        if account_id not in self._sent:
            sent = self.graph.sent_by.get(account_id, [])
            self._sent[account_id] = [payment for payment in sent if self.shows(payment)]
        return self._sent[account_id]

    def received(self, account_id: str) -> list[Payment]:
        """The account's visible received payments, oldest first."""
        if account_id not in self._received:
            received = self.graph.received_by.get(account_id, [])
            self._received[account_id] = [payment for payment in received if self.shows(payment)]
        return self._received[account_id]

    def owners(self, account_id: str) -> list[Link]:
        """The account's visible ownership links (each from its party)."""
        return [
            link
            for link in self.graph.owners_of.get(account_id, [])
            if link.start <= self.t and self.party_visible(link.source)
        ]

    def owned(self, party_id: str) -> list[Link]:
        """The party's visible ownership links to visible accounts."""
        return [
            link
            for link in self.graph.owned_by.get(party_id, [])
            if link.start <= self.t and self.account_visible(link.target)
        ]

    def links(self, kind: _Kind, party_id: str) -> list[Link]:
        """The visible party's links to nodes of kind."""
        return [
            link
            for link in self.graph.party_links[kind.nodes].get(party_id, [])
            if link.start <= self.t
        ]

    def users(self, kind: _Kind, node_id: str) -> list[Link]:
        """The node's visible links, each from its party."""
        return [
            link
            for link in self.graph.node_links[kind.nodes].get(node_id, [])
            if link.start <= self.t and self.party_visible(link.source)
        ]

    def observed(self, kind: _Kind, node_id: str) -> list[Payment]:
        """The visible payments the node was observed on."""
        found = self.graph.observed[kind.nodes].get(node_id, [])
        return [payment for payment in found if self.shows(payment)]

    # Expansion.

    def subgraph(self, seed: str) -> dict[str, Any]:
        """The printed row of the seed's subgraph."""
        sub = _Subgraph({seed})
        reach = self._expand(sub, seed, self.settings["seed_payments"])
        for account_id in sorted(reach.counterparties):
            if not self.counterparty_hub(account_id):
                self._expand(sub, account_id, self.settings["other_payments"])
        for owner in sorted(reach.owners):
            siblings = _newest(
                self.owned(owner),
                self.settings["siblings"],
                lambda link: (-link.start, link.target),
            )
            for link in siblings:
                if link.target != seed:
                    sub.accounts.add(link.target)
                    sub.links.add(_LinkEdge("owned_by", link.target, owner, link.start, link.end))
        self._add_identity_accounts(sub, reach)
        return self._printed(seed, sub)

    def _expand(self, sub: _Subgraph, account_id: str, payments: int) -> _Reach:
        """Adds an expanded account's picks to sub: its newest payments each way, its owners,
        its owners' links to identity nodes and the devices and IPs observed on its sent
        payments."""
        counterparties: set[str] = set()
        for visible in (self.sent(account_id), self.received(account_id)):
            for payment in _newest(visible, payments, lambda payment: (-payment.ts, payment.id)):
                other = payment.receiver if payment.sender == account_id else payment.sender
                sub.pays.add(payment.id)
                sub.accounts.add(other)
                counterparties.add(other)
        counterparties.discard(account_id)
        owners = self.owners(account_id)
        picked = _newest(owners, self.settings["owners"], lambda link: (-link.start, link.source))
        for link in picked:
            sub.parties.add(link.source)
            sub.links.add(_LinkEdge("owned_by", account_id, link.source, link.start, link.end))
        reached = {kind.nodes: set[str]() for kind in KINDS}
        for kind in KINDS:
            enrolled = {link for owner in owners for link in self.links(kind, owner.source)}
            newest = _newest(
                enrolled,
                self.settings[kind.enrolled_cap],
                lambda link: (-link.start, link.target, link.source, link.end),
            )
            for link in newest:
                reached[kind.nodes].add(link.target)
                sub.links.add(
                    _LinkEdge(kind.enrolled, account_id, link.target, link.start, link.end)
                )
            if kind.sent:
                sent = self.sent(account_id)
                uses = _usage((node, p.ts) for p in sent for node in _observed_on(p, kind))
                last_used = _newest(
                    uses.items(), self.settings[kind.sent_cap], lambda use: (-use[1][1], use[0])
                )
                for node, (first, last, count) in last_used:
                    reached[kind.nodes].add(node)
                    sub.sent.add(_SentEdge(kind.sent, account_id, node, first, last, count))
            sub.nodes[kind.nodes] |= reached[kind.nodes]
        return _Reach(counterparties, {link.source for link in picked}, reached)

    def _add_identity_accounts(self, sub: _Subgraph, reach: _Reach) -> None:
        """Adds the other customers' visible deposit accounts that the seed's identity nodes
        that are not hubs reach: one candidate per link of a party that is not an owner of
        the seed, and one per account observed sending from the node with such an owner.
        Newest first, at most reverse_accounts per node and relation and identity_accounts
        distinct accounts."""
        candidates: set[_Candidate] = set()
        for kind in KINDS:
            for node in reach.nodes[kind.nodes]:
                if self.node_row(kind, node)["hub"]:
                    continue
                for link in self.users(kind, node):
                    if link.source in reach.owners:
                        continue
                    candidates.update(
                        _Candidate(
                            -link.start,
                            owned.target,
                            node,
                            kind.enrolled,
                            link.source,
                            link.start,
                            link.end,
                            0,
                        )
                        for owned in self.owned(link.source)
                        if self._deposit(owned.target)
                    )
                if not kind.sent:
                    continue
                senders = _usage((p.sender, p.ts) for p in self.observed(kind, node))
                for account_id, (first, last, count) in senders.items():
                    owners = {link.source for link in self.owners(account_id)}
                    if self._deposit(account_id) and owners - reach.owners:
                        candidates.add(
                            _Candidate(-last, account_id, node, kind.sent, "", first, last, count)
                        )
        taken = Counter[tuple[str, str]]()
        kept: set[str] = set()
        for candidate in sorted(candidates):
            full = candidate.account not in kept and len(kept) >= self.settings["identity_accounts"]
            if (
                full
                or taken[candidate.node, candidate.relation] >= self.settings["reverse_accounts"]
            ):
                continue
            taken[candidate.node, candidate.relation] += 1
            kept.add(candidate.account)
            sub.accounts.add(candidate.account)
            node_id, first, last = candidate.node, candidate.first, candidate.last
            if candidate.party:
                sub.links.add(
                    _LinkEdge(candidate.relation, candidate.account, node_id, first, last)
                )
            else:
                edge = _SentEdge(
                    candidate.relation, candidate.account, node_id, first, last, candidate.uses
                )
                sub.sent.add(edge)

    def _deposit(self, account_id: str) -> bool:
        return self.graph.accounts[account_id].account_type == "deposit"

    def counterparty_hub(self, account_id: str) -> bool:
        """Whether a hop-1 counterparty is a hub: more than hub_threshold visible parties own
        the other accounts of its visible payments. One with no more payments in all time
        than that is not counted."""
        threshold = self.settings["hub_threshold"]
        graph = self.graph
        total = len(graph.sent_by.get(account_id, [])) + len(graph.received_by.get(account_id, []))
        if total <= threshold:
            return False
        others = {payment.receiver for payment in self.sent(account_id)}
        others |= {payment.sender for payment in self.received(account_id)}
        others.discard(account_id)
        owners = {link.source for other in others for link in self.owners(other)}
        return len(owners) > threshold

    # Rows.

    def _printed(self, seed: str, sub: _Subgraph) -> dict[str, Any]:
        nodes: dict[str, list[dict[str, Any]]] = {
            "accounts": [self.account_row(account_id) for account_id in sorted(sub.accounts)],
            "parties": [self.party_row(party_id) for party_id in sorted(sub.parties)],
        }
        for kind in KINDS:
            nodes[kind.nodes] = [
                self.node_row(kind, node) for node in sorted(sub.nodes[kind.nodes])
            ]
        printed: dict[str, Any] = {"seed": seed}
        for name in NODES:
            printed[name] = [dict(row) for row in nodes[name]]
        for name, (edge, _, _) in RELATIONS.items():
            if edge is PaymentEdge:
                rows = [self.payment_row(self.graph.payments[pay]) for pay in sorted(sub.pays)]
            elif edge is LinkEdge:
                rows = [self.link_row(link) for link in sorted(sub.links) if link.relation == name]
            else:
                rows = [self.sent_row(sent) for sent in sorted(sub.sent) if sent.relation == name]
            printed[name] = rows
        return printed

    def account_row(self, account_id: str) -> dict[str, Any]:
        key = ("accounts", account_id)
        if key not in self._rows:
            self._rows[key] = self._account_row(account_id)
        return self._rows[key]

    def _account_row(self, account_id: str) -> dict[str, Any]:
        account, t = self.graph.accounts[account_id], self.t
        sent, received = self.sent(account_id), self.received(account_id)

        def since(days: int) -> int:
            return t - days * DAY_MS

        def count(payments: list[Payment], days: int) -> int:
            return sum(payment.ts >= since(days) for payment in payments)

        recent_in = [payment for payment in received if payment.ts >= since(30)]
        recent_out = [payment for payment in sent if payment.ts >= since(30)]
        first_from: dict[str, int] = {}
        for payment in received:
            first_from[payment.sender] = min(first_from.get(payment.sender, payment.ts), payment.ts)
        first = [payment for payment in recent_in if payment.ts == first_from[payment.sender]]
        out_times = [payment.ts for payment in sent]

        def passed_on(inflow: Payment) -> bool:
            """Whether a visible outflow followed it within 24 hours, at or after it."""
            index = bisect_left(out_times, inflow.ts)
            return index < len(out_times) and out_times[index] <= inflow.ts + DAY_MS

        tokens = [
            link.start for link in self.graph.tokens_of.get(account_id, []) if link.start <= t
        ]
        last_in = max((payment.ts for payment in received), default=None)
        last_out = max((payment.ts for payment in sent), default=None)
        return {
            "id": account_id,
            "account_type": account.account_type,
            "is_external": account.is_external,
            "age_ms": max(t - account.opened_ms, 0),
            "token_present": bool(tokens),
            "token_age_ms": t - min(tokens) if tokens else 0,
            "in_7d": count(received, 7),
            "in_30d": len(recent_in),
            "in_365d": count(received, 365),
            "out_7d": count(sent, 7),
            "out_30d": len(recent_out),
            "out_365d": count(sent, 365),
            "amount_in_30d": math.fsum(payment.amount for payment in recent_in),
            "amount_out_30d": math.fsum(payment.amount for payment in recent_out),
            "counterparties_in_30d": len({payment.sender for payment in recent_in}),
            "counterparties_out_30d": len({payment.receiver for payment in recent_out}),
            "first_inflows_7d": count(first, 7),
            "first_inflows_30d": len(first),
            "passed_on_30d": math.fsum(p.amount for p in recent_in if passed_on(p)),
            "last_in_present": last_in is not None,
            "since_last_in_ms": 0 if last_in is None else t - last_in,
            "last_out_present": last_out is not None,
            "since_last_out_ms": 0 if last_out is None else t - last_out,
            "closed": 0 < account.closed_ms <= t,
        }

    def party_row(self, party_id: str) -> dict[str, Any]:
        """The party's age and the visible accounts it owns by t, open or closed at t."""
        owned = {link.target for link in self.owned(party_id)}
        closed = {
            account_id
            for account_id in owned
            if 0 < self.graph.accounts[account_id].closed_ms <= self.t
        }
        return {
            "id": party_id,
            "age_ms": max(self.t - self.graph.joined_ms[party_id], 0),
            "open_accounts": len(owned - closed),
            "closed_accounts": len(closed),
        }

    def node_row(self, kind: _Kind, node_id: str) -> dict[str, Any]:
        """A device's, IP's or address's features; hub tells whether it is one."""
        key = (kind.nodes, node_id)
        if key not in self._rows:
            self._rows[key] = self._node_row(kind, node_id)
        return self._rows[key]

    def _node_row(self, kind: _Kind, node_id: str) -> dict[str, Any]:
        t = self.t
        links, observed = self.users(kind, node_id), self.observed(kind, node_id)
        first_link: dict[str, int] = {}
        for link in links:
            first_link[link.source] = min(first_link.get(link.source, link.start), link.start)
        accounts = {owned.target for party in first_link for owned in self.owned(party)}
        accounts |= {payment.sender for payment in observed}
        owners = set(first_link)
        owners |= {link.source for payment in observed for link in self.owners(payment.sender)}
        first = min(
            [link.start for link in links] + [payment.ts for payment in observed], default=t
        )
        return {
            "id": node_id,
            "age_ms": t - first,
            "parties_now": len({link.source for link in links if _active(link.end, t)}),
            "parties_ever": len(first_link),
            "accounts": len(accounts),
            "new_parties_30d": sum(start >= t - 30 * DAY_MS for start in first_link.values()),
            "observed_only": bool(observed) and not links,
            "hub": len(owners) > self.settings["hub_threshold"],
        }

    def payment_row(self, payment: Payment) -> dict[str, Any]:
        """A payment with its pair's history in the window and its receiver's next outflow."""
        prior = [
            earlier.ts
            for earlier in self.sent(payment.sender)
            if earlier.receiver == payment.receiver and earlier.ts < payment.ts
        ]
        later = [out.ts for out in self.sent(payment.receiver) if out.ts >= payment.ts]
        return {
            "id": payment.id,
            "from_id": payment.sender,
            "to_id": payment.receiver,
            "amount": payment.amount,
            "rail": payment.rail,
            "age_ms": self.t - payment.ts,
            "gap_present": bool(prior),
            "gap_ms": payment.ts - max(prior) if prior else 0,
            "pair_age_ms": payment.ts - min(prior) if prior else 0,
            "prior_count": len(prior),
            "first_in_pair": not prior,
            "next_outflow_present": bool(later),
            "next_outflow_ms": min(later) - payment.ts if later else 0,
        }

    def link_row(self, link: _LinkEdge) -> dict[str, Any]:
        """A link that ended by t shows how long ago, and how long it held."""
        active = _active(link.end, self.t)
        return {
            "from_id": link.from_id,
            "to_id": link.to_id,
            "since_start_ms": self.t - link.start,
            "active": active,
            "since_end_ms": 0 if active else self.t - link.end,
            "held_ms": (self.t if active else link.end) - link.start,
        }

    def sent_row(self, sent: _SentEdge) -> dict[str, Any]:
        return {
            "from_id": sent.from_id,
            "to_id": sent.to_id,
            "since_first_ms": self.t - sent.first,
            "since_last_ms": self.t - sent.last,
            "uses": sent.uses,
        }


def canonical(batch: SubgraphBatch) -> SubgraphBatch:
    """batch with each list sorted and each amount rounded to a millionth, so two answers
    holding the same nodes and edges compare equal: lists are unordered, and TigerGraph adds
    amounts in any order."""

    def rounded(item: Any) -> Any:
        changes = {
            column.name: round(getattr(item, column.name), 6)
            for column in fields(item)
            if column.type == "float"
        }
        return replace(item, **changes)

    subgraphs = tuple(
        replace(
            subgraph,
            **{
                name: tuple(
                    sorted((rounded(item) for item in getattr(subgraph, name)), key=astuple)
                )
                for name in (*NODES, *RELATIONS)
            },
        )
        for subgraph in batch.subgraphs
    )
    return replace(batch, subgraphs=subgraphs)
