"""The streaming context source: statuses, hop pools, LRU, concurrency, bisection, spot checks."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import threading
import time
from typing import Any

import numpy as np
import pytest
from pyTigerGraph.common.exception import TigerGraphException

from mule_pattern_learner.batching.assemble import build_batch, build_root_batch
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY
from mule_pattern_learner.data.contexts import ContextSource, context_hash
from mule_pattern_learner.data.hub_registry import HubRegistry
from mule_pattern_learner.reference.batch_features import node_features
from mule_pattern_learner.testing.builders import (
    CORE_PLAN,
    PLAN,
    SAMPLER,
    SMALL_SAMPLER,
    context,
    context_row,
    event,
    message,
    neighbourhood,
    query_context_batch,
    root,
)
from mule_pattern_learner.testing.fake_connection import FakeConn, executor
from mule_pattern_learner.testing.fake_graph import ContextServer, FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import (
    ContextTimeoutError,
    TigerGraphContextFetcher,
    query_context_split,
    validate_context,
)
from mule_pattern_learner.tigergraph.executor import ServerTimeoutError


def test_per_request_failures_become_none_and_are_counted() -> None:
    keys = [root(i) for i in range(5)]
    server = ContextServer({keys[1]: "history_capacity_exceeded", keys[3]: "missing_entity"})
    store = ContextSource(
        TigerGraphContextFetcher(server), plan=PLAN, sampler=SAMPLER, request_batch_size=16
    )
    rows = store.fetch(keys + [keys[1]])
    assert [row is None for row in rows] == [False, True, False, True, False, True]
    assert store.rejections == Counter({"history_capacity_exceeded": 1, "missing_entity": 1})
    assert rows[0] is not None and rows[0]["node_id"] == keys[0].node_id
    # Cached rejections are served without another call and counted again.
    assert store.fetch([keys[3]]) == [None] and store.database_calls == 1
    assert store.rejections["missing_entity"] == 2
    assert query_context_batch(server, keys[:2], plan=PLAN, sampler=SAMPLER)[1] is None
    # The same counts per hop: roots (hop 1) and children (hop 2) are reported apart.
    assert store.fetch([keys[1]], hop=2) == [None]
    assert store.rejections_by_hop == {
        1: Counter({"history_capacity_exceeded": 1, "missing_entity": 2}),
        2: Counter({"history_capacity_exceeded": 1}),
    }
    assert sum(store.rejections_by_hop.values(), Counter()) == store.rejections
    store.close()


def test_hop_pools_and_flags_are_sent_and_lru_is_keyed_by_hop() -> None:
    plan = FeaturePlan(("entity_meta", "message_core", "time_encoding", "rolling_windows"), "tgat")
    server = ContextServer()
    store = ContextSource(TigerGraphContextFetcher(server), plan=plan, sampler=SAMPLER, capacity=8)
    key = root(0)
    store.fetch([key], hop=1)
    store.fetch([key], hop=2)
    store.fetch([key], hop=1)
    assert store.database_calls == 2 and len(server.calls) == 2
    first, second = server.calls
    assert {k: first[k] for k in SAMPLER.query_params(1)} == SAMPLER.query_params(1)
    assert {k: second[k] for k in SAMPLER.query_params(2)} == SAMPLER.query_params(2)
    assert first["include_rolling_windows"] and not second["include_rolling_windows"]
    assert {k for k in first if k.startswith("include_")} == set(plan.query_flags(1))
    assert (1, key) in store.memory and (2, key) in store.memory
    with pytest.raises(ValueError, match="hop"):
        store.fetch([key], hop=3)
    store.close()


def test_sources_count_requested_distinct_and_cached_contexts() -> None:
    store = ContextSource(
        TigerGraphContextFetcher(ContextServer()), plan=PLAN, sampler=SAMPLER, capacity=8
    )
    keys = [root(i) for i in range(4)]
    # A key repeated within one fetch is asked for once.
    store.fetch([*keys, keys[0]])
    store.fetch(keys[:2])
    store.fetch(keys[:1], hop=2)
    counts = store.counts
    assert (counts.requested, counts.cache_hits, counts.distinct) == (7, 2, 5)
    assert counts.seen == {context_hash(key, 1) for key in keys} | {context_hash(keys[0], 2)}
    store.close()
    # The hash is the same in every process, so a resumed run can restore it.
    key = ContextKey("Account", "A1", 7, 8, "scope", 2)
    assert context_hash(key, 1) == 7462442381914773116
    assert 0 <= context_hash(key, 2) < 2**63 and context_hash(key, 2) != context_hash(key, 1)
    # Each context of a batch is asked for once: the roots it pins are not counted again.
    roots = [ContextKey("Account", f"R{i:02}", 100, 1000) for i in range(8)]
    with ContextSource(
        TigerGraphContextFetcher(FakeTigerGraph(factory=neighbourhood)),
        plan=CORE_PLAN,
        sampler=SMALL_SAMPLER,
    ) as source:
        prepared = build_root_batch(
            source,
            roots,
            fanouts=(8, 4),
            device="cpu",
            plan=CORE_PLAN,
            sampler=SMALL_SAMPLER,
            hubs=HubRegistry.empty(),
            mode="eval",
        )
    assert prepared.stats["contexts"] > len(roots)
    assert source.counts.requested == source.counts.distinct == prepared.stats["contexts"]


def test_lru_is_bounded_and_close_releases_it() -> None:
    store = ContextSource(
        TigerGraphContextFetcher(ContextServer()), plan=PLAN, sampler=SAMPLER, capacity=8
    )
    for start in range(0, 64, 16):
        store.fetch([root(i) for i in range(start, start + 16)])
        assert len(store.memory) <= 8
    # Recency follows key order, not request completion order, and rows carry no
    # request position, so a refetch in another grouping returns identical rows.
    keys = [root(i) for i in range(100, 180)]
    rows = store.fetch(keys)
    assert list(store.memory) == [(1, key) for key in keys[-8:]]
    calls = store.database_calls
    assert store.fetch(keys[-8:]) == rows[-8:] and store.database_calls == calls
    assert store.fetch(keys[:2]) == rows[:2] and "request_index" not in (rows[0] or {})
    store.close()
    assert not store.memory
    with pytest.raises(RuntimeError, match="closed"):
        store.fetch([root(0)])


def test_concurrent_fetches_share_requests_and_respect_concurrency() -> None:
    server = ContextServer(delay=0.01)
    store = ContextSource(
        TigerGraphContextFetcher(server),
        plan=PLAN,
        sampler=SAMPLER,
        capacity=4096,
        request_batch_size=4,
        concurrency=3,
    )
    shared = [root(i) for i in range(40)]
    results: dict[int, list[dict[str, Any] | None]] = {}
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            keys = shared[n : n + 20] + [root(1000 + n)]
            rows = store.fetch(keys, hop=1 + n % 2)
            assert [(row or {}).get("node_id") for row in rows] == [key.node_id for key in keys]
            results[n] = rows
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(results) == 12
    assert server.peak <= 3
    assert max(server.requested.values()) == 1  # each (hop, key) requested once
    assert store.database_calls == len(server.calls)
    store.close()


def test_failed_request_propagates_to_every_waiting_fetch() -> None:
    class Failing(ContextServer):
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            time.sleep(0.05)
            raise ValueError("contract violation")

    store = ContextSource(
        TigerGraphContextFetcher(Failing()), plan=PLAN, sampler=SAMPLER, concurrency=2
    )
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            store.fetch([root(i) for i in range(8)])
        except ValueError as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(errors) == 3 and not store._inflight
    store.close()


def test_close_without_wait_cancels_queued_requests_and_leaves_daemon_workers() -> None:
    release = threading.Event()

    entered: list[int] = []

    class Blocking(ContextServer):
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            entered.append(len(params["node_ids"]))
            release.wait(10)
            return super().run(name, params)

    server = Blocking()
    store = ContextSource(
        TigerGraphContextFetcher(server),
        plan=PLAN,
        sampler=SAMPLER,
        request_batch_size=1,
        concurrency=1,
    )
    started = threading.Event()
    errors: list[BaseException] = []

    def fetch() -> None:
        started.set()
        try:
            store.fetch([root(i) for i in range(4)])
        except BaseException as error:
            errors.append(error)

    fetcher = threading.Thread(target=fetch, daemon=True)
    fetcher.start()
    started.wait(5)
    deadline = time.monotonic() + 5
    while not entered and time.monotonic() < deadline:
        time.sleep(0.01)
    assert entered == [1]  # one request in flight, the others wait in the window
    workers = [t for t in threading.enumerate() if t.name.startswith("context-requests")]
    assert workers and all(t.daemon for t in workers)
    began = time.monotonic()
    store.close(wait=False)
    assert time.monotonic() - began < 1.0  # did not wait for the blocked request
    release.set()
    fetcher.join(5)
    assert not fetcher.is_alive() and errors  # the fetch fails; nothing new is queued
    assert entered == [1]
    with pytest.raises(RuntimeError, match="closed"):
        store.fetch([root(9)])
    store.close()  # a second close waits for the (finished) workers and is harmless


def test_timed_out_blocks_are_bisected_and_a_single_slow_key_is_fatal() -> None:
    class Slow(ContextServer):
        """Times out on any request with more than `limit` keys or with a slow key."""

        def __init__(self, limit: int, slow_ids: set[str]) -> None:
            super().__init__()
            self.limit, self.slow_ids = limit, slow_ids
            self.sizes: list[int] = []

        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            with self.lock:
                self.sizes.append(len(params["node_ids"]))
            if len(params["node_ids"]) > self.limit or self.slow_ids & set(params["node_ids"]):
                raise ServerTimeoutError(f"{CONTEXT_QUERY} timed out")
            return super().run(name, params)

    keys = [root(i) for i in range(8)]
    server = Slow(limit=2, slow_ids=set())
    store = ContextSource(
        TigerGraphContextFetcher(server), plan=PLAN, sampler=SAMPLER, request_batch_size=8
    )
    rows = store.fetch(keys)
    assert [row and row["node_id"] for row in rows] == [key.node_id for key in keys]
    assert server.sizes == [8, 4, 2, 2, 4, 2, 2]
    assert store.database_calls == 4 and store.diagnostics["timeout_splits"] == 3
    store.close()
    server = Slow(limit=8, slow_ids={keys[5].node_id})
    store = ContextSource(
        TigerGraphContextFetcher(server), plan=PLAN, sampler=SAMPLER, request_batch_size=8
    )
    with pytest.raises(ContextTimeoutError, match="A0005") as caught:
        store.fetch(keys)
    assert caught.value.key == keys[5] and caught.value.hop == 1
    assert not store._inflight
    store.close()
    # Through the real executor a multi-key block is split at once (no repeat of the
    # timed-out request) and a single key gets one retry before the fatal error.
    responder = ContextServer()
    timeout = TigerGraphException("Query timeout exceeded", "REST-3002")

    def answer(name: str, params: dict[str, Any], kwargs: dict[str, Any]) -> Any:
        return responder.run(name, params)

    conn = FakeConn([timeout, answer, answer])
    rows, calls = query_context_split(executor(conn), keys[:2], plan=PLAN, sampler=SAMPLER)
    assert calls == 2 and [len(call[1]["node_ids"]) for call in conn.calls] == [2, 1, 1]
    conn = FakeConn([timeout, timeout, answer])
    with pytest.raises(ContextTimeoutError):
        query_context_split(executor(conn), keys[:1], plan=PLAN, sampler=SAMPLER)
    assert len(conn.calls) == 2


def test_encoding_spot_checks_follow_the_cadence_and_are_stripped() -> None:
    server = ContextServer()
    store = ContextSource(
        TigerGraphContextFetcher(server), plan=PLAN, sampler=SAMPLER, request_batch_size=1, concurrency=1,
        encoding_check_every=3,
    )  # fmt: skip
    rows = store.fetch([root(i) for i in range(7)])
    assert [call["emit_encodings"] for call in server.calls] == [
        True, False, False, True, False, False, True,
    ]  # fmt: skip
    assert store.diagnostics["encoding_checks"] == 3
    assert all(row and row["age_encoding"] == {} == row["gap_encoding"] for row in rows)
    store.close()


def test_corrupted_or_missing_spot_check_vectors_fail() -> None:
    for server, expected in (
        (ContextServer(corrupt=True), "shared basis"),
        (ContextServer(omit_encodings=True), "do not cover"),
    ):
        store = ContextSource(TigerGraphContextFetcher(server), plan=PLAN, sampler=SAMPLER)
        with pytest.raises(ValueError, match=expected):
            store.fetch([root(0)])
        store.close()
    key = root(0)
    row = context_row(key, [event(990, key)], encodings=True)
    validate_context(key, row, PLAN, SAMPLER)  # optional vectors are verified when present
    row["gap_encoding"]["payment_out:E990"][0] += 0.5
    with pytest.raises(ValueError, match="encoding"):
        validate_context(key, row, PLAN, SAMPLER)


def test_the_context_source_serves_repeats_from_its_bounded_lru() -> None:
    keys = [ContextKey("Account", str(i), 100, 1000) for i in range(80)]
    memory = ContextSource(
        TigerGraphContextFetcher(FakeTigerGraph({})),
        capacity=3,
        plan=FeaturePlan(),
        sampler=SamplerPlan(),
    )
    rows = memory.fetch(keys)
    assert len(memory.memory) == 3
    calls = memory.database_calls
    assert memory.fetch(keys[-3:]) == rows[-3:]
    assert memory.database_calls == calls


def test_hops_use_their_own_pools_and_only_spot_checks_carry_encodings() -> None:
    root = ContextKey("Account", "root", 100, 1000)
    many = [message(99 - i, 990 - 10 * i, root, node_id=f"p{i}") for i in range(12)]
    executor = FakeTigerGraph({root: context(root, many)})
    with ContextSource(
        TigerGraphContextFetcher(executor),
        plan=CORE_PLAN,
        sampler=SMALL_SAMPLER,
        encoding_check_every=1000,
    ) as source:
        batch = build_batch(
            source,
            [root],
            fanouts=(8, 2),
            plan=CORE_PLAN,
            sampler=SMALL_SAMPLER,
            mode="train",
        )
        assert source.diagnostics["encoding_checks"] == 1
    # The root pool returns 4 + 1 + 1 per payment relation; resampling keeps 3 of them.
    assert batch["first_mask"][0].sum() == 3
    roots_pool = (4, 1, 1, 1, 2048)
    children_pool = (2, 0, 0, 0, 2048)
    assert set(executor.pools) == {roots_pool, children_pool}
    assert executor.encoded_requests == 1
    first = next(params for name, params in executor.calls if name == CONTEXT_QUERY)
    assert first["emit_encodings"] is True and "include_hub_indicator" not in first
    children = [p for n, p in executor.calls if n == CONTEXT_QUERY][1:]
    assert all(not p["emit_encodings"] and not p["include_pair_window_counts"] for p in children)


def test_same_context_in_two_scopes_or_hops_is_never_shared() -> None:
    key = ContextKey("Account", "a", 100, 1000, "strict", 1)
    executor = FakeTigerGraph()
    store = ContextSource(TigerGraphContextFetcher(executor), plan=CORE_PLAN, sampler=SMALL_SAMPLER)
    store.fetch([key], hop=1)
    store.fetch([key], hop=2)
    store.fetch([replace(key, visibility_phase=2)], hop=1)
    store.fetch([key], hop=1)
    assert store.database_calls == 3
    store.close()
    bad = deepcopy(context(key))
    bad["features"]["history_withheld"] = 1
    with pytest.raises(ValueError, match="Unknown node feature"):
        validate_context(key, bad, CORE_PLAN, SMALL_SAMPLER)
    assert np.isfinite(node_features(context(key), CORE_PLAN)).all()
