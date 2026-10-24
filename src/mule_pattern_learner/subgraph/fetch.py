"""Fetch: fill the cache for the keys a role needs, (Monday, scored account) pairs, and only for
the rows it lacks (docs/design.md, "Cache" and "Query plumbing").

For each Monday, fetch:

1. reads read_hubs once, from the cache or TigerGraph;
2. takes the scored accounts whose subgraph is not complete, a part at a time, and runs B4, B5
   and B6 for those whose rows each lacks, with the Monday's account hubs as a parameter;
3. runs B1 to B3 for every account node of those subgraphs whose rows each lacks: the
   accounts of B4's subgraph and B5's co-users' accounts, but for the account hubs, which get
   flag-only rows; the scored account always gets them, even when it is a hub;
4. records the part's scored accounts as complete, so a fetch interrupted at any point resumes
   where it stopped, and a second fetch of the same keys makes no request.

Every table's rows are written a part of accounts_per_part accounts at a time, as soon as the
answers for them are in, so a fetch that stops loses only the part it was filling. Nothing
keeps the answers of a whole call in memory: the nodes of a part are read back from the cache
the part's B4 and B5 were written to, on a first fetch and on a resumed one alike.

Every query runs through the fetch executor, and fetch refuses to start unless the parity
record holds the version of every query it runs. A request holds at most accounts_per_request
accounts, fewer once a query's answers have been measured: as many as fit the response budget
at the 90th percentile of the bytes per account its answers took, which each request reads
anew from the answers before it. A request TigerGraph refuses for its size or that times out
is halved, down to one account (tigergraph.executor.run_in_halves). The number of accounts in
a request never changes a result.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_CONFIG, TransportConfig
from ..tigergraph.executor import FetchExecutor, QueryExecutor, run_in_halves
from ..tigergraph.installer import require_parity
from ..tigergraph.queries import (
    READ_ACTIVITY_CHANGE,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_MONEY_FLOW,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    VERSIONS,
    read_activity_change,
    read_community,
    read_counterparties,
    read_hubs,
    read_money_flow,
    read_payment_events,
    read_shared_identity,
)
from ..tigergraph.queries.read_hubs import Hubs
from .cache import (
    ACTIVITY,
    COMMUNITY,
    COMPLETE,
    COUNTERPARTIES,
    MONEY_FLOW,
    PAYMENT_EVENTS,
    SHARED_IDENTITY,
    Cache,
    Table,
)

# Every query fetch runs; the parity record must hold each one's version.
FETCHED = (
    READ_HUBS,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    READ_COMMUNITY,
    READ_MONEY_FLOW,
    READ_COUNTERPARTIES,
    READ_ACTIVITY_CHANGE,
)
# The share of a query's measured accounts whose bytes size its requests, and how many of the
# latest accounts are kept: a Monday of a 300,000-person world has millions.
PERCENTILE = 0.9
MEASURED_ACCOUNTS = 10_000
# Each query's parameters, from the Monday, the request's accounts and the account hubs.
PARAMS: dict[str, Callable[[int, tuple[str, ...], Sequence[str]], dict[str, Any]]] = {
    READ_PAYMENT_EVENTS: read_payment_events.params,
    READ_SHARED_IDENTITY: read_shared_identity.params,
    READ_COMMUNITY: read_community.params,
    READ_COUNTERPARTIES: read_counterparties.params,
    READ_MONEY_FLOW: lambda t_ms, accounts, hubs: read_money_flow.params(t_ms, accounts),
    READ_ACTIVITY_CHANGE: lambda t_ms, accounts, hubs: read_activity_change.params(t_ms, accounts),
}
# The tables B4 to B6 fill for scored accounts, and B1 to B3 for every account node.
SUBGRAPH_TABLES: tuple[Table[Any], ...] = (PAYMENT_EVENTS, SHARED_IDENTITY, COMMUNITY)
ACCOUNT_TABLES: tuple[Table[Any], ...] = (MONEY_FLOW, COUNTERPARTIES, ACTIVITY)
# Each per-account query with the table its answer is parsed and kept by, B1 to B6 in order.
PER_ACCOUNT: dict[str, Table[Any]] = {
    table.query: table for table in (*ACCOUNT_TABLES, *SUBGRAPH_TABLES)
}


@dataclass
class Fetched:
    """What a fetch did, per query: the requests it answered (each half of a halved one counted
    on its own), the accounts it fetched, and the accounts it found cached; under "complete",
    the scored accounts whose whole subgraph it found cached."""

    requests: Counter[str] = field(default_factory=lambda: Counter[str]())
    accounts: Counter[str] = field(default_factory=lambda: Counter[str]())
    cached: Counter[str] = field(default_factory=lambda: Counter[str]())


class Sizer:
    """The accounts of each query's requests: accounts_per_request until an answer of the query
    has been measured, then as many as fit the response budget at the 90th percentile of the
    bytes per account of its latest measured accounts, at least one and at most
    accounts_per_request."""

    def __init__(self, transport: TransportConfig) -> None:
        self.most = transport.accounts_per_request
        self.budget = transport.response_budget_bytes
        self.measured: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=MEASURED_ACCOUNTS))

    def size(self, query: str) -> int:
        found = self.measured.get(query)
        if not found:
            return self.most
        per_account = float(np.quantile(found, PERCENTILE))
        if per_account <= 0:
            return self.most
        return max(1, min(self.most, int(self.budget // per_account)))

    def measure(self, query: str, per_account: Iterable[int]) -> None:
        """Record the bytes each account of an answer took."""
        self.measured[query].extend(per_account)


def account_bytes[R](table: Table[R], answer: R, asked: Sequence[str]) -> list[int]:
    """The bytes each of the asked accounts' rows of an answer take, as compact JSON, in the
    order asked: the rows of a scored account's own copy of its subgraph for B4 and B5, so a
    subgraph's rows that another shares count for each, and an account with no row takes 0.
    This is the bytes the account would take if asked alone, which keeps the 90th percentile
    the size of a large subgraph and never the mean of a request."""
    sizes: Counter[str] = Counter()
    for kind in table.kinds:
        for row in table.encode(answer)[kind.name]:
            sizes[row[kind.key]] += len(json.dumps(row, separators=(",", ":")))
    return [sizes[account] for account in asked]


def chunks[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


class Fetcher:
    """One fetch: the executor it runs queries through, the cache it fills, how it sizes
    requests and what it did."""

    def __init__(self, executor: QueryExecutor, cache: Cache, transport: TransportConfig) -> None:
        self.executor = FetchExecutor(executor)
        self.cache = cache
        self.part_accounts = transport.accounts_per_part
        self.sizer = Sizer(transport)
        self.done = Fetched()

    def hubs(self, t_ms: int) -> Hubs:
        """The Monday's hubs, cached once."""
        found = self.cache.hubs(t_ms)
        if found is not None:
            self.done.cached[READ_HUBS] += 1
            return found
        rows = self.executor.run(READ_HUBS, read_hubs.params(t_ms))
        self.done.requests[READ_HUBS] += 1
        found = read_hubs.parse(rows)
        self.cache.write_hubs(t_ms, found)
        return found

    def run[R](
        self, table: Table[R], t_ms: int, accounts: Sequence[str], hubs: Sequence[str]
    ) -> list[R]:
        """The answers of table's query for accounts, each request sized from the answers
        before it, and halved when refused for its size or timing out."""
        query = table.query
        answers: list[R] = []
        start = 0
        while start < len(accounts):
            chunk = accounts[start : start + self.sizer.size(query)]
            start += len(chunk)
            replies = run_in_halves(
                self.executor, query, chunk, lambda part: PARAMS[query](t_ms, part, hubs)
            )
            for part, rows in replies:
                self.done.requests[query] += 1
                answer = table.parse(rows, part)
                self.sizer.measure(query, account_bytes(table, answer, part))
                answers.append(answer)
        self.done.accounts[query] += len(accounts)
        return answers

    def fill[R](
        self, table: Table[R], t_ms: int, accounts: Collection[str], hubs: Sequence[str]
    ) -> None:
        """Fetch and cache the rows of accounts that table lacks, a part of accounts_per_part
        accounts at a time, each written as soon as its answers are in."""
        missing = self.cache.missing(table.name, t_ms, accounts)
        self.done.cached[table.query] += len(set(accounts)) - len(missing)
        for part in chunks(missing, self.part_accounts):
            self.cache.write(table, t_ms, part, self.run(table, t_ms, part, hubs))

    def nodes(self, t_ms: int, scored: Sequence[str], hubs: Collection[str]) -> set[str]:
        """The account nodes of the scored accounts' subgraphs that get B1 to B3 rows, read
        from the cached B4 and B5: every account of B4's subgraphs and B5's co-users'
        accounts but the account hubs, and the scored accounts themselves. An owner-enrolment
        or seen edge of B5 joins an account of its own subgraph, and one that joins another
        is refused, since its account would get no B1 to B3 rows."""
        found: set[str] = set()
        graphs = {graph.scored_id: graph for graph in self.cache.rows(PAYMENT_EVENTS, t_ms, scored)}
        for identity in self.cache.rows(SHARED_IDENTITY, t_ms, scored):
            listed = {account.account_id for account in graphs[identity.scored_id].accounts}
            listed |= {account.account_id for account in identity.accounts}
            for edge in (*identity.owner_enrolled, *identity.seen):
                if edge.account_id not in listed:
                    raise ValueError(
                        f"{READ_SHARED_IDENTITY} gave {identity.scored_id} an edge of the "
                        f"account {edge.account_id}, which no account of its subgraph lists"
                    )
            found |= listed
        return (found - set(hubs)) | set(scored)

    def monday(self, t_ms: int, scored: Collection[str]) -> None:
        """Fill the cache for the scored accounts of the Monday t, and drop the cache's
        indexes of the Monday, which no later Monday needs."""
        try:
            self._monday(t_ms, scored)
        finally:
            self.cache.forget(t_ms)

    def _monday(self, t_ms: int, scored: Collection[str]) -> None:
        todo = self.cache.missing(COMPLETE, t_ms, scored)
        self.done.cached[COMPLETE] += len(set(scored)) - len(todo)
        if not todo:
            return
        hubs = {account.account_id for account in self.hubs(t_ms).accounts}
        listed = sorted(hubs)
        for part in chunks(todo, self.part_accounts):
            for table in SUBGRAPH_TABLES:
                self.fill(table, t_ms, part, listed)
            nodes = self.nodes(t_ms, part, hubs)
            for table in ACCOUNT_TABLES:
                self.fill(table, t_ms, nodes, listed)
            self.cache.complete(t_ms, part)


def fetch(
    executor: QueryExecutor,
    cache: Cache,
    keys: Mapping[int, Collection[str]],
    transport: TransportConfig = DEFAULT_CONFIG.transport,
    parity: Path | None = None,
) -> Fetched:
    """Fill the cache for keys, the scored accounts of each Monday, through the fetch
    executor. The parity record (gsql/parity.toml unless parity names another) must hold the
    version of every query fetch runs."""
    require_parity({name: VERSIONS[name] for name in FETCHED}, parity)
    fetcher = Fetcher(executor, cache, transport)
    for t_ms in sorted(keys):
        fetcher.monday(t_ms, keys[t_ms])
    return fetcher.done
