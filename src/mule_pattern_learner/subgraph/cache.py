"""The cache of fetched rows (docs/design.md, "Cache").

Under a world's identity and the key of its query versions and settings
(tigergraph.world.cache_key), one folder per scoring time holds the population (data.population),
the hubs, the crowds, and one table per query, B1 to B6:

    data/cache/<world identity>/<cache key>/<time>/<table>/part-<n>/<kind>.parquet
    data/cache/<world identity>/<cache key>/<time>/<table>/part-<n>/index.parquet

- A table holds one Parquet file per kind of row its query prints, one row per node or edge,
  every field named as its row dataclass names it: B4's accounts, payments and ownership
  edges; B5's customers, devices, IPs, co-users' accounts and its three kinds of edge; B1's
  accounts and tracked inflows; one kind for B2, B3 and B6. A row of B4 or B5 carries the
  scored account it belongs to, scored_id, since one example holds its own copy of its
  subgraph; every other row names its account.
- A part holds the rows of a set of accounts, and is written once: its files and its index of
  those accounts go into a temporary folder, which a rename puts in place whole. So a part is
  never without its index, and a crash between the two leaves nothing: no part, and no
  account that reads as cached without its rows. A table's index is its parts' indexes
  together, read once and kept for the time being read.
- The hubs and the crowds of a scoring time are one part each, without an index. The complete
  table is parts that hold an index alone: the scored accounts whose subgraph and every account
  node's B1 to B3 rows are all cached.
- Rows read back are reparsed by their query's own parser, from the response they were parsed
  from, so cached rows pass the live checks.
- The cache holds the indexes of one scoring time at a time, the one last asked about, and
  forget drops them early, so a fetch or a reader that goes through 52 weeks never holds more
  than one time's. A 300,000-person world has millions of entries in a time's indexes.

Only fetch asks which accounts a table lacks (missing). Every other reader names its keys:
rows(table, t, accounts) gives exactly those accounts' rows, and nothing lists the index
to its caller, so training, selection and scoring read only the keys they computed themselves
(the Keys chosen by their reader guard).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import SubgraphConfig
from ..paths import cache_dir
from ..tigergraph.queries import (
    MARK_CROWDS,
    READ_ACTIVITY_CHANGE,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_MONEY_FLOW,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    mark_crowds,
    read_activity_change,
    read_community,
    read_counterparties,
    read_hubs,
    read_money_flow,
    read_payment_events,
    read_shared_identity,
)
from ..tigergraph.queries.mark_crowds import Crowd, Crowds
from ..tigergraph.queries.read_activity_change import ActivityChange
from ..tigergraph.queries.read_community import Community
from ..tigergraph.queries.read_counterparties import Counterparties
from ..tigergraph.queries.read_hubs import HubAccount, Hubs, Leaf
from ..tigergraph.queries.read_money_flow import MoneyFlow, MoneyFlows, TrackedInflow
from ..tigergraph.queries.read_payment_events import (
    EventAccount,
    OwnedBy,
    PaymentEvent,
    PaymentGraph,
)
from ..tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityAccount,
    IdentityGraph,
    NodeRow,
    OwnerEnrolled,
    Seen,
)
from ..tigergraph.world import cache_key, time_folder

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# The column that names the scored account a row of B4 or B5 belongs to.
SCORED = "scored_id"
ACCOUNT = "account_id"
# The file of a part that names its accounts.
INDEX_FILE = "index.parquet"
INDEX_SCHEMA = pa.schema([(ACCOUNT, pa.string())])
# The one part of the hubs and of the crowds of a scoring time.
WHOLE_PART = "part-0"
# The Arrow type of each type a row's field may have (tigergraph.queries.rows.FIELD_TYPES).
ARROW_TYPES = {"str": pa.string(), "int": pa.int64(), "bool": pa.bool_(), "float": pa.float64()}
# The rows of each kind of a table, as plain dicts.
Rows = dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Kind:
    """A kind of row of a table, kept in its own file of each part: its name (the file's, and
    the key its query prints it under), its row dataclass, and the column naming the account
    each row belongs to."""

    name: str
    row: type[DataclassInstance]
    key: str = ACCOUNT

    @property
    def schema(self) -> pa.Schema:
        columns = [(SCORED, pa.string())] if self.key == SCORED else []
        for field in fields(self.row):
            kind = field.type if isinstance(field.type, str) else getattr(field.type, "__name__")
            columns.append((field.name, ARROW_TYPES[kind]))
        return pa.schema(columns)


@dataclass(frozen=True)
class Table[R]:
    """A cached table: its folder's name, its query, its kinds of row, and how a parsed answer
    R turns into rows of each kind (encode) and back into the response it was parsed from
    (printed, every row after the response's head), which the query's parser reads (parse)."""

    name: str
    query: str
    kinds: tuple[Kind, ...]
    encode: Callable[[R], Rows]
    printed: Callable[[Rows], list[dict[str, Any]]]
    parse: Callable[[Sequence[object], Collection[str]], R]


def _plain(rows: Iterable[DataclassInstance]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def _scored(scored: str, rows: Iterable[DataclassInstance]) -> list[dict[str, Any]]:
    return [{SCORED: scored, **asdict(row)} for row in rows]


def _vertices(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A printed vertex set's projection, as the parsers read it."""
    return [{"attributes": dict(row)} for row in rows]


def _unscored(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name != SCORED}


def _unique(rows: Iterable[Mapping[str, Any]], id_field: str) -> list[dict[str, Any]]:
    """One row per id of rows the subgraphs share (a vertex's row depends on it, the scoring time
    and the hubs alone, never on the scored account), refusing two that differ."""
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        plain = _unscored(row)
        held = found.setdefault(plain[id_field], plain)
        if held != plain:
            raise ValueError(f"The cache holds two rows of {plain[id_field]} that differ")
    return list(found.values())


def _grouped(rows: Iterable[Mapping[str, Any]], id_field: str | None) -> dict[str, list[Any]]:
    """Each scored account's rows, without scored_id, or their ids when id_field names one."""
    found: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        found[row[SCORED]].append(_unscored(row) if id_field is None else row[id_field])
    return dict(found)


def _one_kind(rows: Iterable[DataclassInstance]) -> Rows:
    return {"accounts": _plain(rows)}


def _accounts_printed(rows: Rows) -> list[dict[str, Any]]:
    return [{"accounts": _vertices(rows["accounts"])}]


def _money_flow_printed(rows: Rows) -> list[dict[str, Any]]:
    return [
        {"accounts": _vertices(rows["accounts"])},
        {"tracked_inflows": [dict(row) for row in rows["tracked_inflows"]]},
    ]


def _payment_events_rows(graphs: tuple[PaymentGraph, ...]) -> Rows:
    found: Rows = {"accounts": [], "payments": [], "owned_by": []}
    for graph in graphs:
        found["accounts"] += _scored(graph.scored_id, graph.accounts)
        found["payments"] += _scored(graph.scored_id, graph.payments)
        found["owned_by"] += _scored(graph.scored_id, graph.owned_by)
    return found


def _payment_events_printed(rows: Rows) -> list[dict[str, Any]]:
    return [
        {"accounts": _vertices(_unique(rows["accounts"], "account_id"))},
        {"payments": _vertices(_unique(rows["payments"], "payment_id"))},
        {"subgraph_accounts": _grouped(rows["accounts"], "account_id")},
        {"subgraph_payments": _grouped(rows["payments"], "payment_id")},
        {"owned_by": _grouped(rows["owned_by"], None)},
    ]


# B5's vertex kinds with their id fields, and its edge kinds.
IDENTITY_VERTICES = {
    "customers": "customer_id",
    "devices": "node_id",
    "ips": "node_id",
    "accounts": "account_id",
}
IDENTITY_EDGES = ("enrolled", "owner_enrolled", "seen")


def _shared_identity_rows(graphs: tuple[IdentityGraph, ...]) -> Rows:
    found: Rows = {name: [] for name in (*IDENTITY_VERTICES, *IDENTITY_EDGES)}
    for graph in graphs:
        for name in found:
            found[name] += _scored(graph.scored_id, getattr(graph, name))
    return found


def _shared_identity_printed(rows: Rows) -> list[dict[str, Any]]:
    printed: list[dict[str, Any]] = [
        {name: _vertices(_unique(rows[name], id_field))}
        for name, id_field in IDENTITY_VERTICES.items()
    ]
    nodes = _grouped(rows["devices"], "node_id")
    for scored, ips in _grouped(rows["ips"], "node_id").items():
        nodes[scored] = nodes.get(scored, []) + ips
    printed += [
        {"subgraph_customers": _grouped(rows["customers"], "customer_id")},
        {"subgraph_nodes": nodes},
        {"co_user_accounts": _grouped(rows["accounts"], "account_id")},
    ]
    printed += [{name: _grouped(rows[name], None)} for name in IDENTITY_EDGES]
    return printed


def _hubs_rows(hubs: Hubs) -> Rows:
    return {
        "accounts": _plain(hubs.accounts),
        "devices": _plain(hubs.devices),
        "ips": _plain(hubs.ips),
    }


def _hubs_printed(rows: Rows) -> list[dict[str, Any]]:
    return [{name: _vertices(rows[name])} for name in ("accounts", "devices", "ips")]


def _hubs_parse(rows: Sequence[object], accounts: Collection[str]) -> Hubs:
    del accounts
    return read_hubs.parse(rows)


HUBS = Table[Hubs](
    "hubs",
    READ_HUBS,
    (Kind("accounts", HubAccount), Kind("devices", Leaf, "node_id"), Kind("ips", Leaf, "node_id")),
    _hubs_rows,
    _hubs_printed,
    _hubs_parse,
)


def _crowds_rows(crowds: Crowds) -> Rows:
    return {"devices": _plain(crowds.devices), "ips": _plain(crowds.ips)}


def _crowds_printed(rows: Rows) -> list[dict[str, Any]]:
    return [{name: _vertices(rows[name])} for name in ("devices", "ips")]


def _crowds_parse(rows: Sequence[object], accounts: Collection[str]) -> Crowds:
    del accounts
    return mark_crowds.parse(rows)


CROWDS = Table[Crowds](
    "crowds",
    MARK_CROWDS,
    (Kind("devices", Crowd, "node_id"), Kind("ips", Crowd, "node_id")),
    _crowds_rows,
    _crowds_printed,
    _crowds_parse,
)
MONEY_FLOW = Table[MoneyFlows](
    "money_flow",
    READ_MONEY_FLOW,
    (Kind("accounts", MoneyFlow), Kind("tracked_inflows", TrackedInflow)),
    lambda flows: {
        "accounts": _plain(flows.accounts),
        "tracked_inflows": _plain(flows.tracked_inflows),
    },
    _money_flow_printed,
    read_money_flow.parse,
)
COUNTERPARTIES = Table[tuple[Counterparties, ...]](
    "counterparties",
    READ_COUNTERPARTIES,
    (Kind("accounts", Counterparties),),
    _one_kind,
    _accounts_printed,
    read_counterparties.parse,
)
ACTIVITY = Table[tuple[ActivityChange, ...]](
    "activity",
    READ_ACTIVITY_CHANGE,
    (Kind("accounts", ActivityChange),),
    _one_kind,
    _accounts_printed,
    read_activity_change.parse,
)
PAYMENT_EVENTS = Table[tuple[PaymentGraph, ...]](
    "payment_events",
    READ_PAYMENT_EVENTS,
    (
        Kind("accounts", EventAccount, SCORED),
        Kind("payments", PaymentEvent, SCORED),
        Kind("owned_by", OwnedBy, SCORED),
    ),
    _payment_events_rows,
    _payment_events_printed,
    read_payment_events.parse,
)
SHARED_IDENTITY = Table[tuple[IdentityGraph, ...]](
    "shared_identity",
    READ_SHARED_IDENTITY,
    (
        Kind("customers", CustomerRow, SCORED),
        Kind("devices", NodeRow, SCORED),
        Kind("ips", NodeRow, SCORED),
        Kind("accounts", IdentityAccount, SCORED),
        Kind("enrolled", Enrolled, SCORED),
        Kind("owner_enrolled", OwnerEnrolled, SCORED),
        Kind("seen", Seen, SCORED),
    ),
    _shared_identity_rows,
    _shared_identity_printed,
    read_shared_identity.parse,
)
COMMUNITY = Table[tuple[Community, ...]](
    "community",
    READ_COMMUNITY,
    (Kind("accounts", Community),),
    _one_kind,
    _accounts_printed,
    read_community.parse,
)
# The tables of B1 to B6, by name.
TABLES: dict[str, Table[Any]] = {
    table.name: table
    for table in (MONEY_FLOW, COUNTERPARTIES, ACTIVITY, PAYMENT_EVENTS, SHARED_IDENTITY, COMMUNITY)
}
# The index of the scored accounts whose subgraph and account nodes' rows are all cached.
COMPLETE = "complete"


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _parts(folder: Path) -> list[int]:
    """The numbers of a table folder's parts, in increasing order."""
    if not folder.is_dir():
        return []
    numbers = (path.name.removeprefix("part-") for path in folder.glob("part-*"))
    return sorted(int(number) for number in numbers if number.isdigit())


def _write_part(
    folder: Path,
    name: str,
    kinds: Sequence[Kind],
    rows: Rows,
    accounts: Collection[str] | None = None,
) -> None:
    """Write a part's files, and its index of accounts unless that is None, into a temporary
    folder and rename it into place whole."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(folder / name)
    try:
        temporary.mkdir()
        for kind in kinds:
            table = pa.Table.from_pylist(rows[kind.name], schema=kind.schema)
            pq.write_table(table, temporary / f"{kind.name}.parquet")
        if accounts is not None:
            index = pa.table({ACCOUNT: sorted(set(accounts))}, schema=INDEX_SCHEMA)
            pq.write_table(index, temporary / INDEX_FILE)
        temporary.rename(folder / name)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class Cache:
    """The cached rows of one world under one key of query versions and settings, at root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        # The indexes held, by table name: those of one scoring time, the one last asked about.
        self._time: str | None = None
        self._indexes: dict[str, dict[str, int]] = {}

    @classmethod
    def of(cls, identity: str, subgraph: SubgraphConfig) -> Cache:
        """The cache of the world of that identity, for the current query versions and these
        settings."""
        return cls(cache_dir(identity) / cache_key(subgraph))

    def folder(self, name: str, t_ms: int) -> Path:
        """The folder of a table at the time t; only a whole second after the lookback has one."""
        return self.root / time_folder(t_ms) / name

    def _index(self, name: str, t_ms: int) -> dict[str, int]:
        """The part of each account the table name holds at the time t: its parts' indexes,
        read once, a later part naming an account that two hold. The indexes of another
        time are dropped first."""
        stamp = time_folder(t_ms)
        if stamp != self._time:
            self._indexes.clear()
            self._time = stamp
        if name not in self._indexes:
            folder = self.folder(name, t_ms)
            found: dict[str, int] = {}
            for part in _parts(folder):
                index = pq.read_table(folder / f"part-{part}" / INDEX_FILE, schema=INDEX_SCHEMA)
                found.update(dict.fromkeys(index[ACCOUNT].to_pylist(), part))
            self._indexes[name] = found
        return self._indexes[name]

    def forget(self, t_ms: int) -> None:
        """Drop the indexes held for the time t, which a reader that has finished with the
        time no longer needs; they are read again from the parts if it is asked about
        again."""
        if self._time == time_folder(t_ms):
            self._indexes.clear()
            self._time = None

    def _add(
        self,
        name: str,
        t_ms: int,
        kinds: Sequence[Kind],
        rows: Rows,
        accounts: Collection[str],
    ) -> None:
        """Write rows as a new part of the table name for accounts, with its index."""
        if not accounts:
            raise ValueError(f"A part of {name} holds at least one account")
        folder = self.folder(name, t_ms)
        part = max(_parts(folder), default=-1) + 1
        _write_part(folder, f"part-{part}", kinds, rows, accounts)
        self._index(name, t_ms).update(dict.fromkeys(accounts, part))

    def missing(self, name: str, t_ms: int, accounts: Iterable[str]) -> list[str]:
        """Of accounts, those the table name lacks at the time t, each once, sorted. Only
        fetch asks."""
        index = self._index(name, t_ms)
        return sorted({account for account in accounts if account not in index})

    def write[R](
        self, table: Table[R], t_ms: int, accounts: Collection[str], answers: Iterable[R]
    ) -> None:
        """Cache the answers of table's query for accounts at the time t, as one new part."""
        rows: Rows = {kind.name: [] for kind in table.kinds}
        for answer in answers:
            for name, found in table.encode(answer).items():
                rows[name] += found
        keyed = {row[kind.key] for kind in table.kinds for row in rows[kind.name]}
        if not keyed <= set(accounts):
            raise ValueError(f"An answer of {table.name} holds rows of accounts it was not asked")
        self._add(table.name, t_ms, table.kinds, rows, accounts)

    def complete(self, t_ms: int, accounts: Collection[str]) -> None:
        """Record that every row of the subgraphs of these scored accounts is cached."""
        self._add(COMPLETE, t_ms, (), {}, accounts)

    def rows[R](self, table: Table[R], t_ms: int, accounts: Iterable[str]) -> R:
        """The cached answer of table's query for exactly these accounts at the time t,
        reparsed by the query's parser. An account the table lacks is refused."""
        asked = sorted(set(accounts))
        folder = self.folder(table.name, t_ms)
        index = self._index(table.name, t_ms)
        lacking = [account for account in asked if account not in index]
        if lacking:
            raise KeyError(f"The cache of {table.name} on {time_folder(t_ms)} lacks {lacking}")
        rows: Rows = {kind.name: [] for kind in table.kinds}
        for part in sorted({index[account] for account in asked}):
            for kind in table.kinds:
                found = pq.read_table(
                    folder / f"part-{part}" / f"{kind.name}.parquet",
                    schema=kind.schema,
                    filters=[(kind.key, "in", asked)],
                )
                rows[kind.name] += found.to_pylist()
        head = {"status": "ok"}
        return table.parse([head, *table.printed(rows)], asked)

    def _whole[R](self, table: Table[R], t_ms: int) -> R | None:
        """The cached answer of a table kept whole at the time t, the hubs or the crowds,
        reparsed, or None when it is not cached."""
        folder = self.folder(table.name, t_ms) / WHOLE_PART
        if not folder.is_dir():
            return None
        rows: Rows = {
            kind.name: pq.read_table(
                folder / f"{kind.name}.parquet", schema=kind.schema
            ).to_pylist()
            for kind in table.kinds
        }
        head = {"status": "ok"}
        return table.parse([head, *table.printed(rows)], ())

    def _write_whole[R](self, table: Table[R], t_ms: int, answer: R) -> None:
        """Cache a table kept whole at the time t, once."""
        folder = self.folder(table.name, t_ms)
        if (folder / WHOLE_PART).is_dir():
            raise ValueError(f"The {table.name} of {time_folder(t_ms)} are cached already")
        _write_part(folder, WHOLE_PART, table.kinds, table.encode(answer))

    def hubs(self, t_ms: int) -> Hubs | None:
        """The cached hubs of the time t, reparsed, or None when they are not cached."""
        return self._whole(HUBS, t_ms)

    def write_hubs(self, t_ms: int, hubs: Hubs) -> None:
        """Cache the hubs of the time t, once."""
        self._write_whole(HUBS, t_ms, hubs)

    def crowds(self, t_ms: int) -> Crowds | None:
        """The cached crowds of the time t, reparsed, or None when they are not cached."""
        return self._whole(CROWDS, t_ms)

    def write_crowds(self, t_ms: int, crowds: Crowds) -> None:
        """Cache the crowds of the time t, once."""
        self._write_whole(CROWDS, t_ms, crowds)
