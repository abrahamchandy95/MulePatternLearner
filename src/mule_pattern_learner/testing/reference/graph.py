"""The reference graph's core: one export's 22 contract tables in memory, indexed for the
queries, with the rules of time every query shares (docs/design.md, "Time"), and run, which
answers an installed query as TigerGraph prints it.

It is the test oracle. Written from the design, apart from the GSQL, it answers each query on
small worlds that tests work out by hand, and mule parity compares it with the live graph on
the same export, read from the loader's shards by testing.shards. Each query's answer is
written in a module of this package beside this one, and the package's ANSWERS names them.

It refuses tables that TigerGraph would hold otherwise than as they read, or that break what
every query takes as given: a column set unlike the contract's or a value of another type; a
repeated id; an edge to a vertex that does not exist; a repeated link of one pair and start,
which DISCRIMINATOR(valid_from_ms) merges into one edge; a payment without exactly one sender
and one receiver, or with two rows of one role; an event edge whose time is not its
payment's; and an account with more than one ownership row, since every query takes one owner
and one ownership tenure per account, as the loader does. Lists in an answer are unordered;
this graph sorts them, so its answers are stable. It reads a whole export into memory, so it
suits small worlds.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
import math
from typing import Any

import numpy as np
import pandas as pd

from ...config import DEFAULT_CONFIG
from ...contract import TABLES, Kind, Table
from ...paths import GSQL_DIR
from ...tigergraph.installer import definitions, parameter_names
from ...tigergraph.queries import GRAPH_NAME, QUERY_FILES
from ...tigergraph.queries.rows import DAY_MS, LOOKBACK_MS

BY_NAME: dict[str, Table] = {table.name: table for table in TABLES}
VERTEX_TABLES = tuple(table for table in TABLES if table.kind is Kind.VERTEX)
EDGE_TABLES = tuple(table for table in TABLES if table.kind is not Kind.VERTEX)
# The two kinds of node a customer shares, by vertex type, and the link table of each that
# enrols a customer on one.
DEVICE = "Device"
IP = "IP"
NODE_KINDS = (DEVICE, IP)
ENROLMENTS = {DEVICE: "Party_Uses_Device", IP: "Party_Uses_IP"}
# The rail of a Zelle_Transfer, which has no payment_rail of its own.
ZELLE = "zelle"
# The window of the current test: a link held at some time in the last 30 days.
CURRENT_MS = DEFAULT_CONFIG.subgraph.sharing_days * DAY_MS
# The payments a time-ordered list keeps (docs/design.md, "Time", under Caps), and the second
# whose whole cut keeps them.
RECENT = DEFAULT_CONFIG.subgraph.recent_payments
SECOND_MS = 1_000
# The pandas dtype of each graph type in a contract frame, whose values read back as str, int,
# bool and float. "str" is pandas 3's string dtype, which holds nothing but strings; on pandas 2
# it would be object, which holds anything, so pyproject.toml requires pandas 3.
DTYPES = {"STRING": "str", "UINT": "int64", "INT": "int64", "BOOL": "bool", "DOUBLE": "float64"}
CONTRACT_DTYPES = {kind: pd.Series([], dtype=dtype).dtype for kind, dtype in DTYPES.items()}


def _type_name(graph_type: str) -> str:
    """A graph type as a refusal names it: a DOUBLE must be finite, since an amount of NaN or
    infinity has no cents ("A NaN amount fails the export", docs/design.md, "What PhantomLedger
    must provide")."""
    return "finite DOUBLE" if graph_type == "DOUBLE" else graph_type


def _value(table: str, column: str, graph_type: str, value: object) -> object:
    """A table's value, once it has its column's type: a str, an int (not negative for a
    UINT), a bool, or a finite float (an int too) for a DOUBLE."""
    if graph_type == "STRING" and isinstance(value, str):
        return value
    if graph_type == "BOOL" and isinstance(value, bool):
        return value
    number = not isinstance(value, bool)
    if number and graph_type == "INT" and isinstance(value, int):
        return value
    if number and graph_type == "UINT" and isinstance(value, int) and value >= 0:
        return value
    if number and graph_type == "DOUBLE" and isinstance(value, int | float):
        if math.isfinite(value):
            return float(value)
    raise ValueError(f"{table}.{column} holds {value!r}, not a {_type_name(graph_type)}")


def frame(table: str, rows: Iterable[Sequence[object]]) -> pd.DataFrame:
    """A contract frame: the rows of table with its columns in the contract's order, each of
    the dtype of its graph type. Every value must have its column's type."""
    columns = BY_NAME[table].columns
    records = [tuple(row) for row in rows]
    for record in records:
        if len(record) != len(columns):
            raise ValueError(f"A row of {table} has {len(record)} values, not {len(columns)}")
        for column, value in zip(columns, record, strict=True):
            _value(table, column.name, column.graph_type, value)
    return pd.DataFrame(
        {
            column.name: pd.Series(
                [record[index] for record in records], dtype=DTYPES[column.graph_type]
            )
            for index, column in enumerate(columns)
        }
    )


def _rows(table: Table, data: pd.DataFrame) -> list[tuple[Any, ...]]:
    """A frame's rows, once its columns are the table's and each value has its type. A column
    of its contract dtype is checked whole: no missing string, no negative UINT and no NaN or
    infinite DOUBLE; any other column value by value."""
    if tuple(str(column) for column in data.columns) != table.column_names:
        raise ValueError(
            f"{table.name} has the columns {list(data.columns)}, not {list(table.column_names)}"
        )
    columns: list[list[Any]] = []
    for column in table.columns:
        series = data[column.name]
        kind = column.graph_type
        if series.dtype == CONTRACT_DTYPES[kind]:
            bad = (
                (kind == "UINT" and bool((series < 0).any()))
                or (kind == "STRING" and bool(series.isna().any()))
                or (kind == "DOUBLE" and not bool(np.isfinite(series.to_numpy()).all()))
            )
            if bad:
                raise ValueError(
                    f"{table.name}.{column.name} holds a value that is not a {_type_name(kind)}"
                )
            columns.append(series.tolist())
        else:
            columns.append([_value(table.name, column.name, kind, v) for v in series.tolist()])
    return list(zip(*columns, strict=True))


@dataclass(frozen=True, slots=True)
class Account:
    """An Account row, in the contract's column order."""

    id: str
    account_type: str
    is_external: bool
    opened_ms: int
    closed_ms: int
    is_mule: int
    pu_label: int
    mule_ring_id: int
    mule_onset_ms: int
    mule_last_ms: int


@dataclass(frozen=True, slots=True)
class Link:
    """A row of a link table: source to target from start until end, 0 while open."""

    source: str
    target: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Payment:
    """A Payment_Transaction or a Zelle_Transfer (its kind) with its sender, its receiver and
    the device and IP seen on it, if any. rail is its payment_rail, or "zelle" for a
    Zelle_Transfer; cents is floor(amount x 100 + 0.5), the integer every query reads."""

    id: str
    kind: str
    amount: float
    cents: int
    rail: str
    ts: int
    sender: str
    receiver: str
    device: str | None
    ip: str | None

    @property
    def self_payment(self) -> bool:
        """The same account at both ends: a payment no query reads but the crowd count."""
        return self.sender == self.receiver


def cents(amount: float) -> int:
    """An amount in integer cents, floor(amount x 100 + 0.5) in double arithmetic, as GSQL
    computes it (docs/design.md, "Time", under Money)."""
    return math.floor(amount * 100 + 0.5)


def share(part: int, whole: int) -> float:
    """A share of cents, part / whole, in one division of two integers as GSQL makes it; 0
    with an empty denominator, which another column shows (docs/design.md, "Encoding")."""
    return part / whole if whole > 0 else 0.0


def second(ts: int) -> int:
    """The whole second a time in milliseconds falls in."""
    return ts // SECOND_MS


def cut_second(times: Collection[int], cap: int = RECENT) -> int | None:
    """The cut of a time-ordered list of cap payments (docs/design.md, "Time", under Ties):
    None when it keeps every payment, else b, the second of the newest payment it leaves out.
    The list then keeps only the payments after the second b, so the payments of one second
    are kept or dropped together, whatever their order."""
    if len(times) <= cap:
        return None
    return second(sorted(times, reverse=True)[cap])


def kept_times(times: Collection[int], cap: int = RECENT) -> list[int]:
    """The times a time-ordered list of cap payments keeps, on its own cut, oldest first."""
    cut = cut_second(times, cap)
    return sorted(ts for ts in times if cut is None or second(ts) > cut)


def most_within(times: Collection[int], window_ms: int) -> int:
    """The most of times in [x, x + window_ms) with x one of them, a window anchored at a
    payment and never at a clock bin (docs/design.md, "Time", under Ties); 0 with none. Equal
    times all count."""
    return max((sum(1 for y in times if x <= y < x + window_ms) for x in times), default=0)


class _Timeline:
    """Payments in (ts, id) order, with their times, so a window is two bisections."""

    def __init__(self, payments: Iterable[Payment]) -> None:
        self.payments = sorted(payments, key=lambda payment: (payment.ts, payment.id))
        self.times = [payment.ts for payment in self.payments]

    def between(self, start: int, end: int) -> list[Payment]:
        """The payments with start <= ts < end, oldest first."""
        return self.payments[bisect_left(self.times, start) : bisect_left(self.times, end)]


def _grouped[V](items: Iterable[V], key: Callable[[V], str]) -> dict[str, list[V]]:
    grouped: dict[str, list[V]] = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return dict(grouped)


@cache
def query_parameters(name: str) -> tuple[str, ...]:
    """The parameters of the installed query name, from its file under gsql/."""
    return tuple(parameter_names(definitions((GSQL_DIR / QUERY_FILES[name]).read_text())[name]))


class ReferenceGraph:
    """The graph of one export, from its 22 contract frames (frame, testing.world and
    testing.shards make them), and the installed queries."""

    graph_name = GRAPH_NAME

    def __init__(self, tables: Mapping[str, pd.DataFrame]) -> None:
        if set(tables) != set(BY_NAME):
            raise ValueError(f"The graph needs the tables {sorted(BY_NAME)}, not {sorted(tables)}")
        # Every table's rows, typed, for read_world, which reads every one of them.
        self.rows: dict[str, list[tuple[Any, ...]]] = {
            table.name: _rows(table, tables[table.name]) for table in TABLES
        }
        self.ids: dict[str, list[str]] = {}
        for table in VERTEX_TABLES:
            found = [row[0] for row in self.rows[table.name]]
            if len(set(found)) != len(found):
                raise ValueError(f"{table.name} repeats an id")
            self.ids[table.name] = sorted(found)
        self._check_edges()
        self.joined_ms: dict[str, int] = dict(self.rows["Party"])
        self.accounts = {row[0]: Account(*row) for row in self.rows["Account"]}
        ownership = [Link(*row) for row in self.rows["Party_Owns_Account"]]
        owners = _grouped(ownership, lambda link: link.target)
        for account_id, links in owners.items():
            if len(links) > 1:
                raise ValueError(f"{account_id} has {len(links)} ownership rows, not one")
        # Each owned account's one ownership link; and by customer, its ownership links in
        # account id order.
        self.ownership = {account_id: links[0] for account_id, links in owners.items()}
        self.owned = {
            party: sorted(links, key=lambda link: link.target)
            for party, links in _grouped(ownership, lambda link: link.source).items()
        }
        # Each enrolment link by kind of node, from its customer and to its node.
        enrolments = {
            kind: [Link(*row) for row in self.rows[ENROLMENTS[kind]]] for kind in NODE_KINDS
        }
        self.enrolled = {
            kind: _grouped(links, lambda link: link.source) for kind, links in enrolments.items()
        }
        self.users = {
            kind: _grouped(links, lambda link: link.target) for kind, links in enrolments.items()
        }
        self.payments = self._payments()
        self.sent_by = {
            account: _Timeline(payments)
            for account, payments in _grouped(
                self.payments.values(), lambda payment: payment.sender
            ).items()
        }
        self.received_by = {
            account: _Timeline(payments)
            for account, payments in _grouped(
                self.payments.values(), lambda payment: payment.receiver
            ).items()
        }
        self.seen_on: dict[str, dict[str, _Timeline]] = {}
        for kind in NODE_KINDS:
            seen = [
                (node, payment)
                for payment in self.payments.values()
                if (node := payment.device if kind == DEVICE else payment.ip) is not None
            ]
            self.seen_on[kind] = {
                node: _Timeline(payment for _, payment in uses)
                for node, uses in _grouped(seen, lambda use: use[0]).items()
            }

    def _check_edges(self) -> None:
        """Every edge's ends exist; no link repeats a pair and start; no role table holds two
        rows of one payment."""
        known = {name: set(ids) for name, ids in self.ids.items()}
        for table in EDGE_TABLES:
            assert table.source is not None and table.target is not None
            rows = self.rows[table.name]
            for row in rows:
                if row[0] not in known[table.source] or row[1] not in known[table.target]:
                    raise ValueError(f"{table.name} joins {row[0]} to {row[1]}, an unknown vertex")
            if table.kind is Kind.LINK:
                keys = [(row[0], row[1], row[2]) for row in rows]
                what = "a link of one pair and start"
            else:
                keys = [row[0] for row in rows]
                what = "a payment"
            if len(set(keys)) != len(keys):
                raise ValueError(f"{table.name} repeats {what}")

    def _payments(self) -> dict[str, Payment]:
        """Every payment with its roles, each role table's time its payment's."""
        found: dict[str, tuple[str, float, str, int]] = {
            row[0]: ("Payment_Transaction", row[1], row[2], row[3])
            for row in self.rows["Payment_Transaction"]
        }
        for payment_id, amount, ts in self.rows["Zelle_Transfer"]:
            if payment_id in found:
                raise ValueError(
                    f"A Payment_Transaction and a Zelle_Transfer share the id {payment_id}"
                )
            found[payment_id] = ("Zelle_Transfer", amount, ZELLE, ts)
        roles: dict[str, dict[str, str]] = defaultdict(dict)
        for table in EDGE_TABLES:
            if table.kind is not Kind.ROLE:
                continue
            role = table.name.split("_", 1)[1]
            for payment_id, end, ts in self.rows[table.name]:
                if ts != found[payment_id][3]:
                    raise ValueError(f"{table.name} gives {payment_id} another time than its own")
                roles[payment_id][role] = end
        payments: dict[str, Payment] = {}
        for payment_id, (kind, amount, rail, ts) in found.items():
            role = roles[payment_id]
            if "From_Account" not in role or "To_Account" not in role:
                raise ValueError(f"{payment_id} needs one sender and one receiver")
            payments[payment_id] = Payment(
                payment_id,
                kind,
                amount,
                cents(amount),
                rail,
                ts,
                role["From_Account"],
                role["To_Account"],
                role.get("Used_Device"),
                role.get("Used_IP"),
            )
        return payments

    def as_of(self, t_ms: int) -> AsOf:
        """The graph as of the time t_ms."""
        return AsOf(self, t_ms)

    def page(self, after_id: str, page: int, keep: Callable[[Account], bool]) -> list[Account]:
        """The first page accounts that keep keeps, by strictly increasing id after after_id."""
        found: list[Account] = []
        ids = self.ids["Account"]
        for account_id in ids[bisect_right(ids, after_id) :]:
            if len(found) == page:
                break
            if keep(self.accounts[account_id]):
                found.append(self.accounts[account_id])
        return found

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        """The rows the installed query name prints for params, as TigerGraph prints them,
        with QueryExecutor's signature, so code that reads a live graph reads this one in
        tests. Parameters other than the query's are refused, as TigerGraph refuses them; the
        keywords are ignored."""
        del timeout_s, attempts, timeout_retries
        from . import ANSWERS  # the package imports this module

        if name not in ANSWERS:
            raise ValueError(f"No query {name} is installed")
        expected = query_parameters(name)
        if sorted(params) != sorted(expected):
            raise ValueError(f"{name} takes the parameters {list(expected)}, not {sorted(params)}")
        return ANSWERS[name](self, **params)


class AsOf:
    """The graph as of a time t, by the rules of docs/design.md, "Time": every boundary is
    strictly before t.

    - A payment is in the lookback when t - 90 days <= ts < t.
    - A link is visible when it began before t and had not ended by t - 90 days, and current
      when it began before t and had not ended by t - 30 days. No other test reads a link's
      end.
    - An account exists when it opened before t, and is open when it exists and has not
      closed by t.
    """

    def __init__(self, graph: ReferenceGraph, t_ms: int) -> None:
        self.graph = graph
        self.t = t_ms
        self.start = t_ms - LOOKBACK_MS

    def visible(self, link: Link) -> bool:
        return link.start < self.t and (link.end == 0 or link.end > self.start)

    def current(self, link: Link) -> bool:
        return link.start < self.t and (link.end == 0 or link.end > self.t - CURRENT_MS)

    def exists(self, account_id: str) -> bool:
        return self.graph.accounts[account_id].opened_ms < self.t

    def open(self, account_id: str) -> bool:
        closed = self.graph.accounts[account_id].closed_ms
        return self.exists(account_id) and (closed == 0 or closed >= self.t)

    def owner(self, account_id: str) -> str | None:
        """The customer that owns the account through a visible ownership link, or None."""
        link = self.graph.ownership.get(account_id)
        return link.source if link is not None and self.visible(link) else None

    def _window(self, timelines: Mapping[str, _Timeline], key: str) -> list[Payment]:
        timeline = timelines.get(key)
        return [] if timeline is None else timeline.between(self.start, self.t)

    def payments(self, account_id: str) -> list[Payment]:
        """Every payment of the account in the lookback, either way, self-payments included,
        oldest first, ties by id."""
        found = {
            payment.id: payment
            for timelines in (self.graph.sent_by, self.graph.received_by)
            for payment in self._window(timelines, account_id)
        }
        return sorted(found.values(), key=lambda payment: (payment.ts, payment.id))

    def sent(self, account_id: str) -> list[Payment]:
        """The account's payments sent in the lookback, but self-payments, oldest first."""
        return [p for p in self._window(self.graph.sent_by, account_id) if not p.self_payment]

    def received(self, account_id: str) -> list[Payment]:
        """The account's payments received in the lookback, but self-payments, oldest first."""
        return [p for p in self._window(self.graph.received_by, account_id) if not p.self_payment]

    def enrolled(self, kind: str, node: str) -> list[Link]:
        """The visible enrolment links to a device or IP node, each from its customer."""
        return [link for link in self.graph.users[kind].get(node, []) if self.visible(link)]

    def seen(self, kind: str, node: str) -> list[Payment]:
        """The payments of the lookback seen on a device or IP node, self-payments included,
        oldest first."""
        return self._window(self.graph.seen_on[kind], node)
