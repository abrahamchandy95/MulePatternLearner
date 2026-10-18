"""Fetch against a spy executor backed by the reference graph: it fills the cache for the keys
and nothing else, resumes where it stopped, makes no request the second time, and sizes and
halves its requests without changing a result (docs/design.md, "Cache" and "Query
plumbing")."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.config import TransportConfig
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
from mule_pattern_learner.subgraph.fetch import FETCHED, Sizer, fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries import (
    READ_COMMUNITY,
    READ_HUBS,
    READ_MONEY_FLOW,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    read_payment_events,
    read_shared_identity,
)

W = 7 * DAY_MS
SCORED = ["a_mule", "a_collector", "a_landlord", "a_h1", "a_r1", "a_c000"]
SHOPPERS = [f"a_c{index:03d}" for index in range(33)]
SUBGRAPH_QUERIES = (READ_PAYMENT_EVENTS, READ_SHARED_IDENTITY, READ_COMMUNITY)
SUBGRAPH_TABLES: tuple[Table[Any], ...] = (PAYMENT_EVENTS, SHARED_IDENTITY, COMMUNITY)
ACCOUNT_TABLES: tuple[Table[Any], ...] = (MONEY_FLOW, COUNTERPARTIES, ACTIVITY)


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
    assert set(spy.names()) <= set(FETCHED) and spy.names()[0] == READ_HUBS
    # B4 to B6 for the scored accounts, equal to the reference's answers.
    assert cache.rows(PAYMENT_EVENTS, T, SCORED) == read_payment_events.parse(
        graph.run(READ_PAYMENT_EVENTS, read_payment_events.params(T, sorted(SCORED), ["a_shop"])),
        SCORED,
    )
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
    # The reader gives exactly the keys asked, after the larger fetch of the Monday.
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
    # A larger fetch of the same Monday asks only for what is new.
    fetch(spy, cache, {T: [*SCORED, "a_h2"]}, parity=parity)
    assert all(
        params["account_ids"] == ["a_h2"] for name, params in spy.calls if name in SUBGRAPH_QUERIES
    )
    assert READ_HUBS not in spy.names()


@dataclass
class Stopping:
    """An executor that stops the process at the first request of one query."""

    spy: Spy
    query: str

    def run(self, name: str, params: dict[str, Any], **keywords: Any) -> list[dict[str, Any]]:
        if name == self.query:
            raise KeyboardInterrupt
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
    # The hubs and B4 to B6 were cached before it stopped; B1 to B3 come now.
    assert not set(spy.names()) & {READ_HUBS, *SUBGRAPH_QUERIES}
    assert READ_MONEY_FLOW in spy.names()
    nodes = _nodes(cache, SCORED)
    for table in ACCOUNT_TABLES:
        assert Cache(tmp_path).missing(table.name, T, nodes) == ["a_shop"]


def test_a_fetch_of_33_accounts_makes_requests_of_32_and_1(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    fetch(spy, Cache(tmp_path), {T: SHOPPERS}, parity=parity)
    for query in SUBGRAPH_QUERIES:
        assert spy.sizes(query) == [32, 1]


def test_a_request_refused_for_its_size_is_halved(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, {T: SHOPPERS}, parity=parity)
    spy = Spy(graph, too_large=(READ_PAYMENT_EVENTS, 8))
    halved = Cache(tmp_path / "halved")
    done = fetch(spy, halved, {T: SHOPPERS}, parity=parity)
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
    rows = [{"payments": "x" * 987}]
    assert len(json.dumps(rows, separators=(",", ":"))) == 1_004
    # 1,004 bytes for 2 accounts: 502 each, so 19 fit in 10,000 bytes.
    sizer.measure("q", 2, rows)
    assert sizer.measured["q"] == [502.0] and sizer.size("q") == 19
    # Ten answers of 502 bytes per account and one of 10,040: the 90th percentile of the
    # eleven, the tenth smallest, is 502.
    big = [{"payments": "x" * 10_023}]
    assert len(json.dumps(big, separators=(",", ":"))) == 10_040
    for _ in range(9):
        sizer.measure("q", 2, rows)
    sizer.measure("q", 1, big)
    assert sizer.size("q") == 19
    # A second answer of 10,040: of twelve, the percentile lies 0.9 of the way from the tenth
    # (502) to the eleventh (10,040), at 9,086.2, so one account fits.
    sizer.measure("q", 1, big)
    assert sizer.size("q") == 1
    # Never more than accounts_per_request, and a query never measured asks for that many.
    sizer.measure("small", 10, [{}])
    assert sizer.size("small") == 32 and sizer.size("other") == 32


def test_the_request_size_never_changes_a_result(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    keys = {T: SHOPPERS, T + W: SHOPPERS[:20]}
    whole = Cache(tmp_path / "whole")
    fetch(Spy(graph), whole, keys, parity=parity)
    spy = Spy(graph)
    small = Cache(tmp_path / "small")
    fetch(spy, small, keys, TransportConfig(response_budget_bytes=3_000), parity=parity)
    # The first Monday's requests are sized before any answer is measured; the second's from
    # the bytes per account measured on the first.
    sizes = spy.sizes(READ_PAYMENT_EVENTS)
    assert sizes[:2] == [32, 1] and max(sizes[2:]) < 32
    for t_ms, scored in keys.items():
        for table in SUBGRAPH_TABLES:
            assert small.rows(table, t_ms, scored) == whole.rows(table, t_ms, scored)
    assert small.rows(SHARED_IDENTITY, T, ["a_c000"]) == read_shared_identity.parse(
        graph.run(READ_SHARED_IDENTITY, read_shared_identity.params(T, ["a_c000"], ["a_shop"])),
        ["a_c000"],
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


def test_a_key_that_is_not_a_monday_at_midnight_is_refused(
    graph: ReferenceGraph, tmp_path: Path, parity: Path
) -> None:
    spy = Spy(graph)
    with pytest.raises(ValueError, match="Monday at 00:00 UTC"):
        fetch(spy, Cache(tmp_path), {T + 3_600_000: SCORED}, parity=parity)
    assert spy.calls == []
