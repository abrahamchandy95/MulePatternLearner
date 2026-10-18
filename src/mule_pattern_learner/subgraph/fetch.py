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

Every query runs through the fetch executor, and fetch refuses to start unless the parity
record holds the version of every query it runs. A request holds at most accounts_per_request
accounts, fewer once a query's answers have been measured: as many as fit the response budget
at the 90th percentile of the bytes per account its answers took so far. A request TigerGraph
refuses for its size or that times out is halved, down to one account
(tigergraph.executor.run_in_halves). The number of accounts in a request never changes a
result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
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
from ..tigergraph.queries.read_payment_events import PaymentGraph
from ..tigergraph.queries.read_shared_identity import IdentityGraph
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
# The scored accounts of one part of a Monday's subgraph tables: a storage setting, which
# never changes a result.
PART_ACCOUNTS = 1_024
# The share of a query's measured answers whose bytes per account size its requests.
PERCENTILE = 0.9
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
    has been measured, then as many as fit the response budget at the 90th percentile of its
    bytes per account, at least one and at most accounts_per_request."""

    def __init__(self, transport: TransportConfig) -> None:
        self.most = transport.accounts_per_request
        self.budget = transport.response_budget_bytes
        self.measured: dict[str, list[float]] = defaultdict(list)

    def size(self, query: str) -> int:
        found = self.measured.get(query)
        if not found:
            return self.most
        per_account = float(np.quantile(found, PERCENTILE))
        if per_account <= 0:
            return self.most
        return max(1, min(self.most, int(self.budget // per_account)))

    def measure(self, query: str, accounts: int, rows: Sequence[object]) -> None:
        """Record an answer's bytes per account, as compact JSON."""
        size = len(json.dumps(rows, separators=(",", ":")))
        self.measured[query].append(size / accounts)


def _chunks[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


class Fetcher:
    """One fetch: the executor it runs queries through, the cache it fills, how it sizes
    requests and what it did."""

    def __init__(self, executor: QueryExecutor, cache: Cache, transport: TransportConfig) -> None:
        self.executor = FetchExecutor(executor)
        self.cache = cache
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
        """The answers of table's query for accounts, in requests of the sized number of
        accounts, each halved when refused for its size or timing out."""
        query = table.query
        answers: list[R] = []
        for chunk in _chunks(accounts, self.sizer.size(query)):
            replies = run_in_halves(
                self.executor, query, chunk, lambda part: PARAMS[query](t_ms, part, hubs)
            )
            for part, rows in replies:
                self.done.requests[query] += 1
                self.sizer.measure(query, len(part), rows)
                answers.append(table.parse(rows, part))
        self.done.accounts[query] += len(accounts)
        return answers

    def fill[R](
        self, table: Table[R], t_ms: int, accounts: Collection[str], hubs: Sequence[str]
    ) -> list[R]:
        """Fetch and cache the rows of accounts that table lacks, and give their answers."""
        missing = self.cache.missing(table.name, t_ms, accounts)
        self.done.cached[table.query] += len(set(accounts)) - len(missing)
        if not missing:
            return []
        answers = self.run(table, t_ms, missing, hubs)
        self.cache.write(table, t_ms, missing, answers)
        return answers

    def nodes(self, t_ms: int, scored: Sequence[str], hubs: Collection[str]) -> set[str]:
        """The account nodes of the scored accounts' subgraphs that get B1 to B3 rows: every
        account of B4's subgraphs and B5's co-users' accounts but the account hubs, and the
        scored accounts themselves."""
        listed = sorted(hubs)
        graphs: list[PaymentGraph] = [
            graph for answer in self.fill(PAYMENT_EVENTS, t_ms, scored, listed) for graph in answer
        ]
        identities: list[IdentityGraph] = [
            graph for answer in self.fill(SHARED_IDENTITY, t_ms, scored, listed) for graph in answer
        ]
        # The subgraphs cached by an earlier fetch that stopped before they were complete.
        earlier = set(scored) - {graph.scored_id for graph in graphs}
        if earlier:
            graphs += self.cache.rows(PAYMENT_EVENTS, t_ms, earlier)
        earlier = set(scored) - {graph.scored_id for graph in identities}
        if earlier:
            identities += self.cache.rows(SHARED_IDENTITY, t_ms, earlier)
        found = {account.account_id for graph in graphs for account in graph.accounts}
        for graph in identities:
            found |= {account.account_id for account in graph.accounts}
            found |= {edge.account_id for edge in (*graph.owner_enrolled, *graph.seen)}
        return (found - set(hubs)) | set(scored)

    def monday(self, t_ms: int, scored: Collection[str]) -> None:
        """Fill the cache for the scored accounts of the Monday t."""
        todo = self.cache.missing(COMPLETE, t_ms, scored)
        self.done.cached[COMPLETE] += len(set(scored)) - len(todo)
        if not todo:
            return
        hubs = {account.account_id for account in self.hubs(t_ms).accounts}
        for part in _chunks(todo, PART_ACCOUNTS):
            nodes = self.nodes(t_ms, part, hubs)
            self.fill(COMMUNITY, t_ms, part, sorted(hubs))
            for table in ACCOUNT_TABLES:
                self.fill(table, t_ms, nodes, sorted(hubs))
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
