"""Fetch against a spy executor backed by the reference graph: it fills the cache for the keys
and nothing else, writes a part at a time and resumes where it stopped, makes no request the
second time, forgets each week's indexes, and sizes and halves its requests without changing
a result (docs/design.md, "Cache" and "Query plumbing")."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import threading
from typing import Any

import pytest
import requests

from mule_pattern_learner.config import FULL_DESIGN_CONFIG, TransportConfig
from mule_pattern_learner.subgraph.cache import (
    ACTIVITY,
    COMMUNITY,
    COMPLETE,
    COUNTERPARTIES,
    MONEY_FLOW,
    PAYMENT_EVENTS,
    SHARED_IDENTITY,
    TABLES,
    Cache,
    Table,
)
from mule_pattern_learner.subgraph.fetch import (
    FETCHED,
    MEASURED_ACCOUNTS,
    Sizer,
    account_bytes,
    fetch,
)
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor
from mule_pattern_learner.tigergraph.queries import (
    MARK_CROWDS,
    READ_ACTIVITY_CHANGE,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_MONEY_FLOW,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    read_activity_change,
    read_community,
    read_counterparties,
    read_money_flow,
    read_payment_events,
    read_shared_identity,
)
from mule_pattern_learner.tigergraph.queries.mark_crowds import Crowds

W = 7 * DAY_MS
# The requests of the full design, one query's at a time: the tests of how a fetch sizes, orders
# and resumes its requests read them in the order they were sent, which concurrent requests
# (the proof of concept's setting, TWO below) leave to the timing of the answers.
SEQUENTIAL = FULL_DESIGN_CONFIG.transport
TWO = replace(SEQUENTIAL, concurrent_requests=2)
SCORED = ["a_mule", "a_collector", "a_landlord", "a_h1", "a_r1", "a_c000"]
SHOPPERS = [f"a_c{index:03d}" for index in range(33)]
FORTY = [f"a_c{index:03d}" for index in range(40)]
SUBGRAPH_QUERIES = (READ_PAYMENT_EVENTS, READ_SHARED_IDENTITY, READ_COMMUNITY)
SUBGRAPH_TABLES: tuple[Table[Any], ...] = (PAYMENT_EVENTS, SHARED_IDENTITY, COMMUNITY)
ACCOUNT_TABLES: tuple[Table[Any], ...] = (MONEY_FLOW, COUNTERPARTIES, ACTIVITY)
# The hub set at T: a_shop, paid by 201 customers.
HUBS = ["a_shop"]
# The request the reference graph answers for each table, with the hubs of T.
REFERENCE: dict[str, Callable[[Sequence[str]], dict[str, Any]]] = {
    MONEY_FLOW.name: lambda accounts: read_money_flow.params(T, accounts),
    COUNTERPARTIES.name: lambda accounts: read_counterparties.params(T, accounts, HUBS),
    ACTIVITY.name: lambda accounts: read_activity_change.params(T, accounts),
    PAYMENT_EVENTS.name: lambda accounts: read_payment_events.params(T, accounts, HUBS),
    SHARED_IDENTITY.name: lambda accounts: read_shared_identity.params(T, accounts, HUBS),
    COMMUNITY.name: lambda accounts: read_community.params(T, accounts, HUBS),
}


def _reference(graph: ReferenceGraph, table: Table[Any], accounts: Sequence[str]) -> Any:
    """The parsed answer of table's query for accounts at T, from the reference graph."""
    keys = sorted(accounts)
    return table.parse(graph.run(table.query, REFERENCE[table.name](keys)), keys)


def _nodes(cache: Cache, scored: Sequence[str]) -> set[str]:
    """The account nodes of the scored accounts' cached subgraphs: B4's accounts and B5's
    co-users' accounts and edge ends."""
    found: set[str] = set()
    for graph in cache.rows(PAYMENT_EVENTS, T, scored):
        found |= {account.account_id for account in graph.accounts}
    for graph in cache.rows(SHARED_IDENTITY, T, scored):
        found |= {account.account_id for account in graph.accounts}
        found |= {edge.account_id for edge in (*graph.owner_enrolled, *graph.seen)}
    return found


def test_fetch_fills_the_cache_for_the_keys(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    cache = Cache(tmp_path)
    done = fetch(spy, cache, {T: SCORED}, parity=parity)
    # The crowds of T are marked before any read of T, then the hubs are read.
    assert set(spy.names()) <= set(FETCHED) and spy.names()[:2] == [MARK_CROWDS, READ_HUBS]
    assert cache.crowds(T) == Crowds((), ())
    # B4 to B6 for the scored accounts, equal to the reference's answers.
    assert cache.rows(PAYMENT_EVENTS, T, SCORED) == _reference(graph, PAYMENT_EVENTS, SCORED)
    for table in SUBGRAPH_TABLES[1:]:
        assert not cache.missing(table.name, T, SCORED)
    # B1 to B3 for every account node of their subgraphs, but the hub a_shop, which a_c000
    # paid: a flag-only row, never expanded and never given B1 to B3.
    nodes = _nodes(cache, SCORED)
    assert "a_shop" in nodes and {"a_vic1", "x_vic2", "a_h2", "a_r2", "x_offshore"} <= nodes
    for table in ACCOUNT_TABLES:
        assert cache.missing(table.name, T, nodes) == ["a_shop"]
    asked = {
        account
        for name, params in spy.calls
        if name in (READ_MONEY_FLOW,)
        for account in params["account_ids"]
    }
    assert asked == nodes - {"a_shop"}
    assert not cache.missing(COMPLETE, T, SCORED)
    assert done.requests[READ_HUBS] == 1 and done.accounts[READ_MONEY_FLOW] == len(asked)
    assert done.requests[MARK_CROWDS] == 1
    # Every table of two accounts equals the reference's answer given a_shop as the hub set:
    # B2, B4, B5 and B6 ran with it, and it changes a_c000's rows. a_c000 pays only the hub,
    # so its payees are all hubs (hub_payee_share_30d 1.0, 0.0 without the set) and its
    # community has 1 member, itself (2 without it).
    keys = ["a_c000", "a_h1"]
    for table in TABLES.values():
        assert cache.rows(table, T, keys) == _reference(graph, table, keys), table.name
    (counterparties,) = cache.rows(COUNTERPARTIES, T, ["a_c000"])
    assert counterparties.hub_payee_share_30d == 1.0
    (community,) = cache.rows(COMMUNITY, T, ["a_c000"])
    assert community.members == 1
    # The reader gives exactly the keys asked, after the larger fetch of the same time.
    assert [row.account_id for row in cache.rows(COMMUNITY, T, ["a_h1", "a_r1"])] == [
        "a_h1",
        "a_r1",
    ]
    flows = cache.rows(MONEY_FLOW, T, ["a_mule"])
    assert [row.account_id for row in flows.accounts] == ["a_mule"]
    assert {row.account_id for row in flows.tracked_inflows} == {"a_mule"}


def test_a_scored_hub_keeps_its_full_row(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    cache = Cache(tmp_path)
    fetch(Spy(graph), cache, {T: ["a_shop"]}, parity=parity)
    for table in ACCOUNT_TABLES:
        assert cache.rows(table, T, ["a_shop"])
    # Its subgraph holds the payers of its newest 32 payments, each given B1 to B3 rows.
    nodes = _nodes(cache, ["a_shop"])
    assert len(nodes) == 33 and all(node.startswith("a_c") for node in nodes - {"a_shop"})
    for table in ACCOUNT_TABLES:
        assert not cache.missing(table.name, T, nodes)


def test_a_second_fetch_makes_no_request(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    cache = Cache(tmp_path)
    fetch(Spy(graph), cache, {T: SCORED}, parity=parity)
    spy = Spy(graph)
    done = fetch(spy, Cache(tmp_path), {T: list(reversed(SCORED))}, parity=parity)
    assert spy.calls == [] and done.cached[COMPLETE] == len(SCORED)
    # A larger fetch of the same time asks only for what is new.
    fetch(spy, cache, {T: [*SCORED, "a_h2"]}, parity=parity)
    assert all(
        params["account_ids"] == ["a_h2"] for name, params in spy.calls if name in SUBGRAPH_QUERIES
    )
    # The graph may hold another time's crowds by then, so T is marked again; its rows are
    # cached already.
    assert spy.names()[0] == MARK_CROWDS and READ_HUBS not in spy.names()


@dataclass
class Stopping:
    """An executor that stops the process at a request of one query: the first, or the one
    after that many of its requests were answered."""

    spy: Spy
    query: str
    after: int = 0
    asked: int = 0

    def run(self, name: str, params: dict[str, Any], **keywords: Any) -> list[dict[str, Any]]:
        if name == self.query:
            if self.asked == self.after:
                raise KeyboardInterrupt
            self.asked += 1
        return self.spy.run(name, params, **keywords)


def test_an_interrupted_fetch_resumes_where_it_stopped(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    cache = Cache(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        fetch(Stopping(Spy(graph), READ_MONEY_FLOW), cache, {T: SCORED}, parity=parity)
    assert cache.missing(COMPLETE, T, SCORED) == sorted(SCORED)
    spy = Spy(graph)
    fetch(spy, Cache(tmp_path), {T: SCORED}, parity=parity)
    # The hubs and B4 to B6 were cached before it stopped; B1 to B3 come now, after T is
    # marked again.
    assert not set(spy.names()) & {READ_HUBS, *SUBGRAPH_QUERIES}
    assert spy.names()[0] == MARK_CROWDS and READ_MONEY_FLOW in spy.names()
    nodes = _nodes(cache, SCORED)
    for table in ACCOUNT_TABLES:
        assert Cache(tmp_path).missing(table.name, T, nodes) == ["a_shop"]


def test_a_fetch_stopped_between_parts_keeps_the_scored_accounts_it_completed(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    # Forty scored accounts in parts of 8. Each part's nodes are its own 8 accounts (the hub
    # a_shop is not given B1), so B1 makes one request a part; the third is stopped.
    transport = replace(SEQUENTIAL, accounts_per_part=8)
    cache = Cache(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        fetch(Stopping(Spy(graph), READ_MONEY_FLOW, after=2), cache, {T: FORTY}, transport, parity)
    assert Cache(tmp_path).missing(COMPLETE, T, FORTY) == FORTY[16:]
    spy = Spy(graph)
    done = fetch(spy, Cache(tmp_path), {T: FORTY}, transport, parity)
    # The two complete parts are not fetched again. The third part's B4 to B6 were cached
    # before the stop, so they are asked for the last two parts alone, and B1 for the last
    # three.
    assert done.cached[COMPLETE] == 16
    assert spy.sizes(READ_PAYMENT_EVENTS) == [8, 8]
    assert spy.sizes(READ_MONEY_FLOW) == [8, 8, 8]
    assert READ_HUBS not in spy.names()
    assert not Cache(tmp_path).missing(COMPLETE, T, FORTY)


def test_a_fetch_stopped_inside_a_table_resumes_at_the_part_it_was_filling(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    # a_shop's subgraph holds 33 accounts, itself and the payers of its newest 32 payments, and
    # all 33 get B1 to B3 rows, in parts of 8: 8, 8, 8, 8 and 1.
    transport = replace(SEQUENTIAL, accounts_per_part=8)
    cache = Cache(tmp_path)
    stopped = Stopping(Spy(graph), READ_MONEY_FLOW, after=2)
    with pytest.raises(KeyboardInterrupt):
        fetch(stopped, cache, {T: ["a_shop"]}, transport, parity)
    nodes = sorted(_nodes(cache, ["a_shop"]))
    assert len(nodes) == 33 and nodes[-1] == "a_shop"
    # Two parts of B1 were written before the third request stopped it, so 16 accounts are
    # cached, and a fresh reader of the folder finds just those.
    assert [path.name for path in cache.folder(MONEY_FLOW.name, T).iterdir()] == [
        "part-0",
        "part-1",
    ]
    assert Cache(tmp_path).missing(MONEY_FLOW.name, T, nodes) == nodes[16:]
    assert Cache(tmp_path).missing(COUNTERPARTIES.name, T, nodes) == nodes
    assert cache.missing(COMPLETE, T, ["a_shop"]) == ["a_shop"]
    spy = Spy(graph)
    fetch(spy, Cache(tmp_path), {T: ["a_shop"]}, transport, parity)
    # It asks for the 17 B1 accounts it lacks, 8, 8 and 1, and for no account it has.
    assert spy.sizes(READ_MONEY_FLOW) == [8, 8, 1]
    assert [
        account
        for name, params in spy.calls
        if name == READ_MONEY_FLOW
        for account in params["account_ids"]
    ] == nodes[16:]
    assert not set(spy.names()) & {READ_HUBS, *SUBGRAPH_QUERIES}
    reader = Cache(tmp_path)
    for table in ACCOUNT_TABLES:
        assert not reader.missing(table.name, T, nodes)
    assert not reader.missing(COMPLETE, T, ["a_shop"])
    # Whatever the part size, the rows are the same.
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, {T: ["a_shop"]}, parity=parity)
    for table in ACCOUNT_TABLES:
        assert reader.rows(table, T, nodes) == whole.rows(table, T, nodes)


@pytest.mark.parametrize("interrupted", [False, True])
def test_an_account_only_b5_reaches_gets_b1_to_b3_rows(
    graph: ReferenceGraph, tmp_path: Path, parity: Path, interrupted: bool
) -> None:
    """The owner's Account Z: a_z has no payment, and shares the ring's device with a_r1. B5
    gives it as a co-user's account, on a first fetch and on one that resumes after B5 was
    cached, so no account is marked complete without its B1 to B3 rows."""
    cache = Cache(tmp_path)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            fetch(Stopping(Spy(graph), READ_MONEY_FLOW), cache, {T: ["a_r1"]}, parity=parity)
        assert cache.missing(MONEY_FLOW.name, T, ["a_z"]) == ["a_z"]
    spy = Spy(graph)
    fetch(spy, Cache(tmp_path), {T: ["a_r1"]}, parity=parity)
    reader = Cache(tmp_path)
    (payments,) = reader.rows(PAYMENT_EVENTS, T, ["a_r1"])
    (identity,) = reader.rows(SHARED_IDENTITY, T, ["a_r1"])
    assert "a_z" not in {account.account_id for account in payments.accounts}
    assert {account.account_id for account in identity.accounts} == {"a_r2", "a_z"}
    for table in ACCOUNT_TABLES:
        assert not reader.missing(table.name, T, ["a_r1", "a_r2", "a_z"])
        assert reader.rows(table, T, ["a_z"]) == _reference(graph, table, ["a_z"])
    assert "a_z" in {
        account
        for name, params in spy.calls
        if name == READ_MONEY_FLOW
        for account in params["account_ids"]
    }


@dataclass
class Planting:
    """An executor that adds to B5's answers an owner-enrolment edge of an account that no
    list of the subgraph holds."""

    spy: Spy

    def run(self, name: str, params: dict[str, Any], **keywords: Any) -> list[dict[str, Any]]:
        rows = self.spy.run(name, params, **keywords)
        if name != READ_SHARED_IDENTITY:
            return rows
        planted = copy.deepcopy(rows)
        for row in planted:
            for edges in row.get("owner_enrolled", {}).values():
                edges.append({**edges[-1], "account_id": "a_zz"})
        return planted


def test_an_edge_of_an_account_the_subgraph_does_not_list_is_refused(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    with pytest.raises(ValueError, match="gave a_r1 an edge of the account a_zz"):
        fetch(Planting(Spy(graph)), Cache(tmp_path), {T: ["a_r1"]}, parity=parity)
    assert Cache(tmp_path).missing(COMPLETE, T, ["a_r1"]) == ["a_r1"]


def test_fetch_forgets_each_times_indexes(
    graph: ReferenceGraph, tmp_path: Path, parity: Path, index_reads: list[Path]
) -> None:
    cache = Cache(tmp_path)
    fetch(Spy(graph), cache, {T: SCORED, T + W: SCORED}, parity=parity)
    # Neither time's indexes are held once its fetch is done, the last included: asking
    # about either reads them again.
    for t_ms in (T + W, T):
        before = len(index_reads)
        assert not cache.missing(COMPLETE, t_ms, SCORED)
        assert len(index_reads) > before
    # A fetch that stops forgets the time it was filling too, and one that has nothing to
    # fetch.
    stopped = Cache(tmp_path / "stopped")
    with pytest.raises(KeyboardInterrupt):
        fetch(Stopping(Spy(graph), READ_MONEY_FLOW), stopped, {T: SCORED}, parity=parity)
    before = len(index_reads)
    assert not stopped.missing(PAYMENT_EVENTS.name, T, SCORED)
    assert len(index_reads) == before + 1
    fetch(Spy(graph), cache, {T: SCORED}, parity=parity)
    before = len(index_reads)
    assert not cache.missing(COMPLETE, T, SCORED)
    assert len(index_reads) > before


def test_each_time_is_marked_once_before_its_reads_and_read_whole_before_the_next(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    fetch(spy, Cache(tmp_path), {T + W: SCORED, T: SCORED}, parity=parity)
    times = [params["t_ms"] for _, params in spy.calls]
    marks = [index for index, name in enumerate(spy.names()) if name == MARK_CROWDS]
    # The graph holds the crowds of one time alone: each time is marked first, and every read
    # of it runs before the next time is marked.
    assert [times[index] for index in marks] == [T, T + W] and marks[0] == 0
    assert times == sorted(times) and times[marks[1] - 1] == T
    assert spy.names()[marks[1] + 1] == READ_HUBS


def test_a_fetch_of_33_accounts_makes_requests_of_32_and_1(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    fetch(spy, Cache(tmp_path), {T: SHOPPERS}, SEQUENTIAL, parity)
    for query in SUBGRAPH_QUERIES:
        assert spy.sizes(query) == [32, 1]


def test_a_request_refused_for_its_size_is_halved(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, {T: SHOPPERS}, SEQUENTIAL, parity)
    spy = Spy(graph, too_large=(READ_PAYMENT_EVENTS, 8))
    halved = Cache(tmp_path / "halved")
    done = fetch(spy, halved, {T: SHOPPERS}, SEQUENTIAL, parity)
    # 32 is refused, then each 16, and the four 8s answer; then the last account alone.
    assert spy.sizes(READ_PAYMENT_EVENTS) == [32, 16, 8, 8, 16, 8, 8, 1]
    assert done.requests[READ_PAYMENT_EVENTS] == 5
    for name, table in TABLES.items():
        keys = SHOPPERS if table in SUBGRAPH_TABLES else ["a_c000"]
        assert halved.rows(table, T, keys) == whole.rows(table, T, keys), name


def test_one_account_refused_for_its_size_is_an_error(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph, too_large=(READ_SHARED_IDENTITY, 0))
    with pytest.raises(Exception, match="more than the response limit") as refused:
        fetch(spy, Cache(tmp_path), {T: ["a_mule"]}, parity=parity)
    assert "for the account a_mule alone" in " ".join(getattr(refused.value, "__notes__", []))


def test_requests_are_sized_by_the_measured_bytes_per_account() -> None:
    sizer = Sizer(TransportConfig(accounts_per_request=32, response_budget_bytes=10_000))
    assert sizer.size("q") == 32
    # Ten accounts of 502 bytes: 19 fit in 10,000 bytes.
    sizer.measure("q", [502] * 10)
    assert sizer.size("q") == 19
    # One of 10,040 makes eleven: the 90th percentile of the eleven, the tenth smallest, is
    # 502, so a lone large account does not shrink the requests.
    sizer.measure("q", [10_040])
    assert sizer.size("q") == 19
    # A second makes twelve: the percentile lies 0.9 of the way from the tenth (502) to the
    # eleventh (10,040), at 9,086.2, so one account fits.
    sizer.measure("q", [10_040])
    assert sizer.size("q") == 1
    # At least one account, even when one account's bytes pass the whole budget: 10,000 over
    # 25,000 is 0.
    sizer.measure("big", [25_000, 25_000])
    assert sizer.size("big") == 1
    # Never more than accounts_per_request, and a query never measured asks for that many; so
    # does one whose accounts took no bytes.
    sizer.measure("small", [10])
    assert sizer.size("small") == 32 and sizer.size("other") == 32
    sizer.measure("empty", [0, 0])
    assert sizer.size("empty") == 32


def test_only_the_latest_accounts_are_measured() -> None:
    sizer = Sizer(TransportConfig(accounts_per_request=32, response_budget_bytes=10_000))
    sizer.measure("q", [10_040] * 5)
    sizer.measure("q", [502] * MEASURED_ACCOUNTS)
    assert len(sizer.measured["q"]) == MEASURED_ACCOUNTS and sizer.size("q") == 19


def test_an_accounts_bytes_are_those_of_the_rows_it_would_take_alone(
    graph: ReferenceGraph,
) -> None:
    # A scored account's own copy of its subgraph takes the same bytes whoever it is asked
    # with: a_c000 and a_c001 share the hub a_shop, which counts for each.
    asked = ["a_c000", "a_c001"]
    both = account_bytes(PAYMENT_EVENTS, _reference(graph, PAYMENT_EVENTS, asked), asked)
    alone = [
        account_bytes(PAYMENT_EVENTS, _reference(graph, PAYMENT_EVENTS, [account]), [account])[0]
        for account in asked
    ]
    # Each holds two account rows (itself and a_shop, 64 bytes each), its one payment (336)
    # and its ownership edge (67): 531 bytes of compact JSON, and 532 for a_c001, whose
    # payment is 172,740,000 ms old, a digit more than a_c000's 86,400,000.
    assert both == alone == [531, 532]
    # The busiest account of the world takes many times that, and each request's bytes per
    # account is the account's, never the mean of the request.
    busy = ["a_mule", "a_h1"]
    assert account_bytes(PAYMENT_EVENTS, _reference(graph, PAYMENT_EVENTS, busy), busy) == [
        6_745,
        567,
    ]
    # B1 has two kinds of row, an account's and its tracked inflows', both keyed by the
    # account: a_mule's row and its four inflows take 977 bytes, a_h1's row alone 387. An
    # account with no row takes 0 bytes.
    flows = _reference(graph, MONEY_FLOW, busy)
    assert len(flows.tracked_inflows) == 4

    def compact(row: Any) -> int:
        return len(json.dumps(asdict(row), separators=(",", ":")))

    rows = [*flows.accounts, *flows.tracked_inflows]
    by_hand = [sum(compact(row) for row in rows if row.account_id == name) for name in busy]
    assert by_hand == [977, 387]
    assert account_bytes(MONEY_FLOW, flows, [*busy, "a_nobody"]) == [*by_hand, 0]


def test_requests_are_sized_by_the_bytes_measured_so_far(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    transport = replace(SEQUENTIAL, response_budget_bytes=3_000)
    fetch(spy, Cache(tmp_path), {T: FORTY, T + W: FORTY}, transport, parity)
    # Forty accounts a week. Every query's first request holds 32, as nothing has been
    # measured; every later request is sized by the answers before it, within a week's
    # fetch as well as from one week to the next. The first answer sets the size of the
    # next request, so the first week's last 8 accounts go in requests of the size that
    # the 32 imply, and the second week's 40 in requests of the size all 40 imply.
    # B4: a shopper's rows take 531 to 533 bytes, the 90th percentile is 533 and
    # 3,000 // 533 is 5, so 8 accounts go as 5 and 3.
    # B5: 180 bytes, 16 fit. B6: 170 bytes, 17 fit. B1: 383 to 389 bytes, 7 fit. B2: 301
    # bytes, 9 fit. B3: 323 to 325 bytes, 9 fit.
    assert {
        query: spy.sizes(query) for query in FETCHED if query not in (MARK_CROWDS, READ_HUBS)
    } == {
        READ_PAYMENT_EVENTS: [32, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5],
        READ_SHARED_IDENTITY: [32, 8, 16, 16, 8],
        READ_COMMUNITY: [32, 8, 17, 17, 6],
        READ_MONEY_FLOW: [32, 7, 1, 7, 7, 7, 7, 7, 5],
        READ_COUNTERPARTIES: [32, 8, 9, 9, 9, 9, 4],
        READ_ACTIVITY_CHANGE: [32, 8, 9, 9, 9, 9, 4],
    }


def test_a_request_is_sized_by_the_90th_percentile_of_its_accounts_not_their_mean(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    # Eight accounts to a request and a budget of 30,000 bytes. The first request holds four
    # shoppers, whose B4 rows take 531 to 533 bytes, and four busy accounts, a_collector,
    # a_landlord, a_mule and a_split, which take 6,976, 5,167, 6,745 and 11,033. Of the eight,
    # sorted, the 90th percentile lies 0.3 of the way from the seventh (6,976) to the eighth
    # (11,033), at 8,193.1 bytes, so 3 accounts fit in 30,000. The mean of the request is 4,006
    # bytes, which would let 7 fit and refuse the busy accounts that follow.
    busy = ["a_collector", "a_landlord", "a_mule", "a_split"]
    scored = [*FORTY[:4], *busy, "a_tenant1", "a_tenant2", "a_vic1", "a_z"]
    spy = Spy(graph)
    transport = replace(SEQUENTIAL, accounts_per_request=8, response_budget_bytes=30_000)
    fetch(spy, Cache(tmp_path), {T: scored}, transport, parity)
    # The twelve accounts, sorted: the first eight are the request above; then 3, and the last
    # account alone.
    assert spy.sizes(READ_PAYMENT_EVENTS) == [8, 3, 1]


def test_the_request_size_never_changes_a_result(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    keys = {T: FORTY, T + W: FORTY}
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, keys, SEQUENTIAL, parity)
    small = Cache(tmp_path / "small")
    fetch(Spy(graph), small, keys, replace(SEQUENTIAL, response_budget_bytes=3_000), parity)
    for t_ms, scored in keys.items():
        for table in SUBGRAPH_TABLES:
            assert small.rows(table, t_ms, scored) == whole.rows(table, t_ms, scored)
        for table in ACCOUNT_TABLES:
            assert small.rows(table, t_ms, scored) == whole.rows(table, t_ms, scored)
    assert small.rows(SHARED_IDENTITY, T, ["a_c000"]) == _reference(
        graph, SHARED_IDENTITY, ["a_c000"]
    )


def test_fetch_needs_a_parity_record_of_every_query_it_runs(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    record = tmp_path / "partial.toml"
    kept = parity.read_text().split("[[passed]]")
    record.write_text("[[passed]]".join(part for part in kept if READ_COMMUNITY not in part))
    spy = Spy(graph)
    with pytest.raises(ValueError, match=f"No parity record for {READ_COMMUNITY} "):
        fetch(spy, Cache(tmp_path), {T: SCORED}, parity=record)
    assert spy.calls == []


def test_a_key_that_is_not_a_whole_second_is_refused(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    with pytest.raises(ValueError, match="whole second"):
        fetch(spy, Cache(tmp_path), {T + 3_600_001: SCORED}, parity=parity)
    assert spy.calls == []


# Proof of concept (25 October 2026, docs/design.md, "Proof-of-concept settings (25 October
# 2026)"): the requests of one query run concurrent_requests at a time.


class Together:
    """An executor that holds each request of the queries it is given until another request of
    the same query has arrived, which only a fetch that sends two at once can give it: a
    sequential fetch waits out the barrier and fails."""

    def __init__(self, spy: Spy, *queries: str) -> None:
        self.spy = spy
        self.barriers = {query: threading.Barrier(2, timeout=30) for query in queries}

    def run(self, name: str, params: dict[str, Any], **keywords: Any) -> list[dict[str, Any]]:
        if name in self.barriers:
            self.barriers[name].wait()
        return self.spy.run(name, params, **keywords)


def test_two_requests_of_a_query_are_in_flight_at_once(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    fetch(Together(spy, *SUBGRAPH_QUERIES), Cache(tmp_path), {T: FORTY}, TWO, parity)
    # Forty accounts go to each of B4 to B6 in two requests of 20, not one of 32 and one of 8.
    for query in SUBGRAPH_QUERIES:
        assert spy.sizes(query) == [20, 20]
    with pytest.raises(threading.BrokenBarrierError):
        sequential = Together(Spy(graph), READ_PAYMENT_EVENTS)
        sequential.barriers[READ_PAYMENT_EVENTS] = threading.Barrier(2, timeout=0.2)
        fetch(sequential, Cache(tmp_path / "one"), {T: FORTY}, SEQUENTIAL, parity)


def test_a_query_is_cut_into_as_many_requests_as_run_at_once(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    # The 33 accounts of a_c000 to a_c032 go in 32 and 1 one at a time, and in 17 and 16, or 11,
    # 11 and 11, when two or three go at once.
    for concurrent, sizes in ((1, [32, 1]), (2, [17, 16]), (3, [11, 11, 11])):
        spy = Spy(graph)
        transport = replace(SEQUENTIAL, concurrent_requests=concurrent)
        fetch(spy, Cache(tmp_path / str(concurrent)), {T: SHOPPERS}, transport, parity)
        for query in SUBGRAPH_QUERIES:
            assert sorted(spy.sizes(query), reverse=True) == sizes


def test_concurrent_requests_fill_the_same_cache_and_count_the_same_accounts(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    keys = {T: FORTY, T + W: FORTY}
    one = Cache(tmp_path / "one")
    alone = fetch(Spy(graph), one, keys, SEQUENTIAL, parity)
    for concurrent in (2, 4):
        cache = Cache(tmp_path / str(concurrent))
        done = fetch(
            Spy(graph), cache, keys, replace(SEQUENTIAL, concurrent_requests=concurrent), parity
        )
        for t_ms, scored in keys.items():
            for table in (*SUBGRAPH_TABLES, *ACCOUNT_TABLES):
                assert cache.rows(table, t_ms, scored) == one.rows(table, t_ms, scored)
        assert done.accounts == alone.accounts and done.cached == alone.cached


def test_requests_sized_by_the_measured_bytes_stay_the_same_run_after_run(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    # Forty accounts a week and a budget of 3,000 bytes, two requests at a time. The first
    # round of each query is sized by nothing measured and holds two requests of 20; every later
    # round is sized by the answers before it, so B4's 533 bytes an account give requests of 5 on
    # the second week, in rounds of two.
    transport = replace(TWO, response_budget_bytes=3_000)
    found: list[dict[str, list[int]]] = []
    for run in range(3):
        spy = Spy(graph)
        fetch(spy, Cache(tmp_path / str(run)), {T: FORTY, T + W: FORTY}, transport, parity)
        found.append({query: sorted(spy.sizes(query)) for query in SUBGRAPH_QUERIES})
    assert found[0] == found[1] == found[2]
    assert found[0][READ_PAYMENT_EVENTS] == [5] * 8 + [20, 20]


def test_a_request_refused_for_its_size_is_halved_inside_its_round(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, {T: SHOPPERS}, SEQUENTIAL, parity)
    halved = Cache(tmp_path / "halved")
    done = fetch(Spy(graph, too_large=(READ_PAYMENT_EVENTS, 8)), halved, {T: SHOPPERS}, TWO, parity)
    # 17 is refused and halved to 9 and 8, and the 9 to 5 and 4: three answers; 16 is refused and
    # halved to 8 and 8: two. The cache is the whole one's.
    assert done.requests[READ_PAYMENT_EVENTS] == 5
    for table in SUBGRAPH_TABLES:
        assert halved.rows(table, T, SHOPPERS) == whole.rows(table, T, SHOPPERS)


def test_the_failure_raised_is_that_of_the_first_request_that_failed(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    # Every request of B5 is refused for its size down to one account, in both requests of the
    # round, and the one raised is the first request's, whichever ended first.
    for run in range(3):
        spy = Spy(graph, too_large=(READ_SHARED_IDENTITY, 0))
        with pytest.raises(Exception, match="more than the response limit") as refused:
            fetch(spy, Cache(tmp_path / str(run)), {T: SHOPPERS}, TWO, parity)
        assert "for the account a_c000 alone" in " ".join(getattr(refused.value, "__notes__", []))


class Connection:
    """A pyTigerGraph connection that answers through the reference graph, and fails the first
    request of one query with a connection error, which the executor retries."""

    def __init__(self, graph: ReferenceGraph, flaky: str) -> None:
        self.graph = graph
        self.flaky = flaky
        self.lock = threading.Lock()
        self.failed = False
        self.calls = 0

    def runInstalledQuery(self, name: str, params: dict[str, Any], **keywords: Any) -> Any:  # noqa: N802
        with self.lock:
            self.calls += 1
            if name == self.flaky and not self.failed:
                self.failed = True
                raise requests.ConnectionError("reset")
        return self.graph.run(name, params)


class Client:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.graphname = "Mule_Pattern_Learner"


def test_a_request_that_is_retried_changes_no_answer_and_is_retried_once(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, {T: FORTY}, SEQUENTIAL, parity)
    connection = Connection(graph, READ_PAYMENT_EVENTS)
    pauses: list[float] = []
    executor = TigerGraphExecutor(client=Client(connection), sleep=pauses.append)
    cache = Cache(tmp_path / "retried")
    done = fetch(executor, cache, {T: FORTY}, TWO, parity)
    # One request of B4 failed once, was retried after a pause (the other request of the round may
    # wait out the shared backoff too, so a pause or two), and answered; every request counts once.
    assert executor.retries[READ_PAYMENT_EVENTS] == 1 and 1 <= len(pauses) <= 2
    assert done.requests[READ_PAYMENT_EVENTS] == 2
    for table in (*SUBGRAPH_TABLES, *ACCOUNT_TABLES):
        assert cache.rows(table, T, FORTY) == whole.rows(table, T, FORTY)
