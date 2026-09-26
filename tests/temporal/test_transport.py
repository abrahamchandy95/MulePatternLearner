"""Transport and preparation: retries, statuses, pools, caching, hubs, labels, config."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import requests
from pyTigerGraph.common.exception import TigerGraphException

from mule_pattern_learner.configuration import REPOSITORY_ROOT, load_config, resolve_path
from mule_pattern_learner.temporal.encoding import BASIS_ID
from mule_pattern_learner.temporal.live import dataset, installation, pipeline, scope, source
from mule_pattern_learner.temporal.live.config_schema import (
    DEFAULT_RUN,
    OPERATIONAL_DEFAULTS,
    LiveConfig,
    run_config,
    validate_config,
)
from mule_pattern_learner.temporal.live.context_query import (
    ContextTimeoutError,
    query_context_batch,
    query_context_split,
    validate_context,
)
from mule_pattern_learner.temporal.live.contract import (
    CONTRACT_VERSION,
    LEGACY_GROUPS,
    ContextKey,
    FeaturePlan,
    PoolPlan,
    SamplerPlan,
)
from mule_pattern_learner.temporal.live.executor import (
    AVAILABILITY,
    DETERMINISTIC,
    SERVER_TIMEOUT,
    ServerTimeoutError,
    TigerGraphExecutor,
    TransientQueryError,
    failure_class,
    is_transient,
)
from mule_pattern_learner.temporal.live.hubs import (
    HUB_COLUMNS,
    HubRegistry,
    hub_manifest,
    hub_threshold,
    load_hub_registry,
    query_hub_registry,
)
from mule_pattern_learner.temporal.live.source import ContextStore, StreamingContextSource
from mule_pattern_learner.temporal.live.supervision import (
    GraphObservedLabels,
    ParquetObservedLabels,
    label_source,
)
from mule_pattern_learner.tigergraph.client import Client, _status_error, _TimeoutConnection
from temporal_fakes import encode, request_keys

PLAN = FeaturePlan(("entity_meta", "message_core", "time_encoding"), "split")
SAMPLER = SamplerPlan(
    "stratified",
    roots=PoolPlan(recent=4, older=2, distinct=1, associations=2, max_history=2048),
    children=PoolPlan(recent=2, associations=0, max_history=1024),
)


# --- fixtures -----------------------------------------------------------------------------


def event(
    seq: int, parent: ContextKey, *, relation: str = "payment_out", gap: int = 5
) -> dict[str, Any]:
    ts = parent.cutoff_ms - 10 * (parent.cutoff_seq - seq)
    return {
        "node_type": "Account",
        "node_id": f"peer{seq}",
        "relation": relation,
        "rail": "card",
        "channel": "digital",
        "stratum": "recent",
        "event_id": f"E{seq}",
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": 12.5,
        "amount_present": True,
        "age_ms": parent.cutoff_ms - ts,
        "gap_ms": gap,
        "gap_present": gap > 0,
        "peer_first_ms": 1,
        "peer_external": False,
        "peer_deposit": True,
    }


def context_row(
    key: ContextKey, messages: list[dict[str, Any]], *, encodings: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **asdict(key),
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
        "basis_id": BASIS_ID,
        "features": {
            "type_Account": 1,
            "is_deposit": 1,
            "1d_out_in_amount_ratio": 0,
            "7d_out_in_amount_ratio": 0,
        },
        "messages": deepcopy(messages),
        "age_encoding": {},
        "gap_encoding": {},
    }
    return encode(row) if encodings else row


def root(i: int, **changes: Any) -> ContextKey:
    return replace(ContextKey("Account", f"A{i:04}", 1000, 100_000, "scope", 1), **changes)


class ContextServer:
    """Fake temporal_training_context endpoint with per-request statuses."""

    def __init__(
        self,
        statuses: dict[ContextKey, str] | None = None,
        *,
        delay: float = 0.0,
        corrupt: bool = False,
        omit_encodings: bool = False,
    ) -> None:
        self.statuses = statuses or {}
        self.delay, self.corrupt, self.omit_encodings = delay, corrupt, omit_encodings
        self.calls: list[dict[str, Any]] = []
        self.requested: Counter[tuple[int, ContextKey]] = Counter()
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        assert name == "temporal_training_context"
        hop = 1 if params["k_assoc"] else 2
        keys = request_keys(params)
        with self.lock:
            self.calls.append(params)
            self.requested.update((hop, key) for key in keys)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            emit = params["emit_encodings"] and params["include_time_encoding"]
            rows = []
            for index, key in enumerate(keys):
                if key in self.statuses:
                    rows.append({"status": self.statuses[key], "request_index": index})
                    continue
                messages = [event(key.cutoff_seq - 1, key), event(key.cutoff_seq - 3, key, gap=0)]
                row = context_row(key, messages, encodings=emit and not self.omit_encodings)
                if emit and self.corrupt:
                    first = next(iter(row["age_encoding"]))
                    row["age_encoding"][first][3] += 0.01
                rows.append({**row, "request_index": index})
            return rows
        finally:
            with self.lock:
                self.active -= 1


class FakeClock:
    """Monotonic clock that advances only when sleeping or when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeConn:
    """Scripted pyTigerGraph connection: each call pops the next outcome.

    An outcome is a result, an exception to raise, or a callable run with the
    call's arguments (it may advance a clock, raise or return).
    """

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[Any, ...]] = []

    def _next(self, *args: Any) -> Any:
        self.calls.append(args)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(*args)
        return outcome

    def runInstalledQuery(self, name: str, params: dict[str, Any], **kwargs: Any) -> Any:
        return self._next(name, params, kwargs)

    def gsql(self, text: str) -> Any:
        return self._next(text)


class RecordingExecutor(TigerGraphExecutor):
    """The real retry policy over a fake connection and a fake clock; sleeps are recorded."""

    def __init__(self, conn: Any, **kwargs: Any) -> None:
        self.clock = FakeClock()
        self.sleeps = self.clock.sleeps
        client = SimpleNamespace(conn=conn, graphname="Mule_Pattern_Learner")
        super().__init__(client=client, sleep=self.clock.sleep, clock=self.clock.time, **kwargs)


def executor(conn: Any, **kwargs: Any) -> RecordingExecutor:
    return RecordingExecutor(conn, **kwargs)


class Runner:
    """A QueryExecutor backed by a function."""

    def __init__(self, respond: Callable[[str, dict[str, Any]], list[dict[str, Any]]]) -> None:
        self.respond = respond

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        return self.respond(name, params)


def http_error(status: int, body: Any = b"") -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return requests.HTTPError(f"{status}", response=response)


def slow(clock: FakeClock, seconds: float, error: BaseException) -> Callable[..., Any]:
    """An outcome that takes `seconds` before failing with `error`."""

    def outcome(*args: Any) -> Any:
        clock.now += seconds
        raise error

    return outcome


# --- failure classes and retry budgets ------------------------------------------------------


def test_availability_failures_are_retried_with_capped_exponential_backoff() -> None:
    html = TigerGraphException("Cannot parse json: <html><h1>Starting workspace</h1></html>")
    conn = FakeConn(
        [
            requests.ConnectionError("reset"),
            html,
            http_error(503, {"error": True, "message": "not ready", "code": "REST-0005"}),
            requests.exceptions.ChunkedEncodingError("cut"),
            http_error(429),
            [{"status": "ok"}],
        ]
    )
    tg = executor(conn, base_delay_s=4, max_delay_s=10)
    assert tg.run("temporal_training_cutoffs", {"cutoff_times": [1]}) == [{"status": "ok"}]
    assert len(conn.calls) == 6
    assert tg.retries["temporal_training_cutoffs"] == 5 and tg.calls == 1
    bases = [4, 8, 10, 10, 10]  # capped at max_delay_s
    assert all(b / 2 <= s <= b for b, s in zip(bases, tg.sleeps, strict=True))
    kwargs = conn.calls[0][2]
    assert kwargs["usePost"] is True and kwargs["timeout"] == 300_000


@pytest.mark.parametrize(
    "error, kind",
    [
        (requests.ConnectionError("refused"), AVAILABILITY),
        (requests.exceptions.ConnectTimeout("connect"), AVAILABILITY),
        (requests.exceptions.ChunkedEncodingError("cut"), AVAILABILITY),
        (http_error(502), AVAILABILITY),
        (http_error(503, {"error": True, "message": "anything"}), AVAILABILITY),
        (http_error(504, b"<html>504 Gateway Time-out</html>"), AVAILABILITY),
        (http_error(408), AVAILABILITY),
        (http_error(429, {"error": True, "message": "Rate limit exceeded"}), AVAILABILITY),
        (http_error(500, b"<!DOCTYPE html><p>oops</p>"), AVAILABILITY),
        (
            TigerGraphException("Cannot parse json: <html>Starting workspace</html>"),
            AVAILABILITY,
        ),
        (TigerGraphException("The graph engine is not ready yet"), AVAILABILITY),
        (TigerGraphException("Server is busy, try again later"), AVAILABILITY),
        (TransientQueryError("HTML page"), AVAILABILITY),
        (json.JSONDecodeError("x", "<html><body>resuming</body></html>", 0), AVAILABILITY),
        (TigerGraphException("Query timeout exceeded", "REST-3002"), SERVER_TIMEOUT),
        (TigerGraphException("The query timed out"), SERVER_TIMEOUT),
        (requests.ReadTimeout("no answer 30 s after the server timeout"), SERVER_TIMEOUT),
        (http_error(500, {"error": True, "code": "REST-3002"}), SERVER_TIMEOUT),
        (http_error(504, {"error": True, "code": "REST-3002"}), SERVER_TIMEOUT),
        (http_error(500), DETERMINISTIC),
        (http_error(500, {"error": True, "message": "Runtime error"}), DETERMINISTIC),
        (TigerGraphException("Cannot parse json: {'a': NaN}"), DETERMINISTIC),
        (TigerGraphException("Query aborted: out of memory"), DETERMINISTIC),
        (json.JSONDecodeError("x", '{"a": NaN}', 6), DETERMINISTIC),
        (http_error(400), None),
        (http_error(401), None),
        (http_error(404), None),
        (TigerGraphException("Query temporal_x is not installed", "REST-1000"), None),
        (ValueError("Returned context differs"), None),
        (KeyError("results"), None),
    ],
)
def test_failure_classes(error: BaseException, kind: str | None) -> None:
    assert failure_class(error) == kind
    assert is_transient(error) is (kind is not None)


def test_suspected_deterministic_failures_are_retried_once() -> None:
    cases: list[tuple[BaseException, type[BaseException]]] = [
        (TigerGraphException("Query timeout exceeded", "REST-3002"), ServerTimeoutError),
        (requests.ReadTimeout("slow"), ServerTimeoutError),
        (http_error(500), TransientQueryError),
        (TigerGraphException("Cannot parse json: NaN"), TransientQueryError),
        (TigerGraphException("out of memory"), TransientQueryError),
    ]
    for error, raised in cases:
        conn = FakeConn([error, error, [{"status": "ok"}]])
        tg = executor(conn, max_attempts=20)
        with pytest.raises(raised, match="after 2 attempt"):
            tg.run("temporal_training_context", {"node_ids": ["a", "b"]})
        assert len(conn.calls) == 2 and len(tg.sleeps) == 1, error
        # One failure followed by success recovers.
        conn = FakeConn([error, [{"status": "ok"}]])
        assert executor(conn).run("q", {}) == [{"status": "ok"}]
    # Callers that split a timed-out request instead ask for no timeout retry.
    conn = FakeConn([TigerGraphException("x", "REST-3002"), [{"status": "ok"}]])
    with pytest.raises(ServerTimeoutError, match="after 1 attempt"):
        executor(conn).run("q", {}, timeout_retries=0)
    assert len(conn.calls) == 1
    # A mixed sequence: availability failures do not use up the deterministic retry.
    timeout = TigerGraphException("x", "REST-3002")
    conn = FakeConn([timeout, requests.ConnectionError("down"), timeout])
    with pytest.raises(ServerTimeoutError, match="after 3 attempt"):
        executor(conn).run("q", {})


def test_fast_outages_are_bounded_by_the_wall_clock_not_by_attempts() -> None:
    # Fast-failing attempts do not count toward max_query_attempts.
    conn = FakeConn([requests.ConnectionError("refused")] * 10 + [[{"status": "ok"}]])
    tg = executor(conn, max_attempts=3, max_outage_s=900)
    assert tg.run("q", {}) == [{"status": "ok"}] and len(conn.calls) == 11
    # The outage budget ends the retries, with a final attempt at its end.
    conn = FakeConn([requests.ConnectionError("refused")] * 100)
    tg = executor(conn, max_attempts=3, max_outage_s=120)
    started = tg.clock.now
    with pytest.raises(TransientQueryError, match=r"unavailable for 120s"):
        tg.run("q", {})
    assert tg.clock.now - started == pytest.approx(120) and len(conn.calls) > 3
    assert sum(tg.sleeps) == pytest.approx(120)
    # max_outage_s = 0 fails on the first availability failure.
    conn = FakeConn([requests.ConnectionError("refused"), [{"status": "ok"}]])
    with pytest.raises(TransientQueryError, match="after 1 attempt"):
        executor(conn, max_outage_s=0).run("q", {})


def test_slow_failing_attempts_count_toward_max_query_attempts() -> None:
    tg = executor(FakeConn([]), max_attempts=3, max_outage_s=3600)
    gateway = http_error(504, b"upstream request timeout")
    tg.client.conn.outcomes = [slow(tg.clock, 45, gateway) for _ in range(10)]
    with pytest.raises(TransientQueryError, match="after 3 attempt.*max_query_attempts = 3"):
        tg.run("q", {})
    assert len(tg.client.conn.calls) == 3


def test_permanent_errors_and_writes_are_not_retried() -> None:
    for error in (
        http_error(404),
        http_error(401),
        http_error(400, {"error": True, "message": "bad parameter"}),
        TigerGraphException("Query temporal_x is not installed", "REST-1000"),
        KeyError("results"),
    ):
        conn = FakeConn([error, [{"status": "ok"}]])
        tg = executor(conn)
        with pytest.raises(type(error)):
            tg.run("q", {})
        assert len(conn.calls) == 1 and not tg.sleeps
    conn = FakeConn([requests.ConnectionError("down")] * 3)
    tg = executor(conn)
    with pytest.raises(TransientQueryError):
        tg.run("q", {}, timeout_s=5, attempts=3)
    assert len(conn.calls) == 3 and len(tg.sleeps) == 2
    assert conn.calls[0][2]["timeout"] == 5000
    # Writes use a single attempt.
    conn = FakeConn([requests.ConnectionError("down"), [{"status": "ok"}]])
    with pytest.raises(TransientQueryError):
        executor(conn).run("temporal_create_training_scope", {}, attempts=1)
    assert len(conn.calls) == 1


def test_worker_threads_share_one_backoff() -> None:
    conn = FakeConn([requests.ConnectionError("down"), [{"status": "ok"}], [{"status": "ok"}]])
    tg = executor(conn, base_delay_s=8, max_delay_s=60)
    waits: list[float] = []
    record = tg.clock.sleep

    def sleep(seconds: float) -> None:
        # While the first operation backs off, another worker starts a request: it
        # waits for the shared backoff before its first attempt.
        if not waits:
            waits.append(seconds)
            tg.run("other", {})
            waits.append(tg.sleeps[-1])
        record(seconds)

    tg._sleep = sleep
    assert tg.run("q", {}) == [{"status": "ok"}]
    assert waits[0] == waits[1] and 4 <= waits[0] <= 8
    assert [call[0] for call in conn.calls] == ["q", "other", "q"]
    # A shorter pause of another thread waits until the latest shared deadline.
    assert tg._backoff_until == 0.0  # cleared by the success
    assert tg._shared_pause(100.0, 30.0) == 30.0 and tg._shared_pause(110.0, 5.0) == 20.0


def test_json_bodied_status_errors_keep_their_status_through_pytigergraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list[tuple[int, Any]] = []

    def fake(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        status, body = script.pop(0)
        response = requests.Response()
        response.status_code, response.url = status, url
        response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.Session, "request", fake)
    conn = _TimeoutConnection(host="http://127.0.0.1", graphname="Mule_Pattern_Learner")
    ok = {"error": False, "message": "", "results": [{"status": "ok"}]}
    script[:] = [
        (503, {"error": True, "message": "The service is not ready", "code": "REST-0005"}),
        (429, {"error": True, "message": "Rate limit exceeded"}),
        (200, ok),
    ]
    tg = executor(conn)
    assert tg.run("q", {"node_ids": ["a"]}) == [{"status": "ok"}]
    assert len(tg.sleeps) == 2 and not script
    script[:] = [(400, {"error": True, "message": "bad parameter", "code": "GSQL-1"}), (200, ok)]
    tg = executor(conn)
    with pytest.raises(TigerGraphException, match="bad parameter"):
        tg.run("q", {})
    assert not tg.sleeps and len(script) == 1
    script[:] = [(500, {"error": True, "message": "timed out", "code": "REST-3002"})] * 2
    with pytest.raises(ServerTimeoutError):
        executor(conn).run("q", {})
    assert not script
    for status in (401, 404, 400, 200):
        response = requests.Response()
        response.status_code = status
        assert _status_error(response) is None
    response = requests.Response()
    response.status_code, response._content = 503, b'{"message": "busy"}'
    error = _status_error(response)
    assert error is not None and "HTTP 503" in str(error) and "busy" in str(error)


def test_resume_page_in_gsql_output_is_transient() -> None:
    conn = FakeConn(["<!DOCTYPE html><title>Starting workspace</title>", "CREATE QUERY q() {}"])
    tg = executor(conn)
    assert tg.gsql("SHOW QUERY q") == "CREATE QUERY q() {}"
    assert len(tg.sleeps) == 1
    assert is_transient(json.JSONDecodeError("x", "<html>", 0))
    assert not is_transient(ValueError("Returned context differs"))


def test_validation_errors_from_a_response_are_never_retried() -> None:
    key = root(1)
    bad = {**context_row(key, [], encodings=False), "request_index": 0, "cutoff_seq": 999}
    conn = FakeConn([[bad], [bad]])
    with pytest.raises(ValueError, match="differs"):
        query_context_batch(executor(conn), [key], plan=PLAN, sampler=SAMPLER)
    assert len(conn.calls) == 1


def test_connection_default_timeout_is_effective_and_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def fake(self: requests.Session, method: str, url: str, **kwargs: Any) -> None:
        seen.append(kwargs.get("timeout"))
        raise requests.ConnectionError("no network in tests")

    monkeypatch.setattr(requests.Session, "request", fake)
    conn = _TimeoutConnection(host="http://127.0.0.1", graphname="Mule_Pattern_Learner")
    client = Client.__new__(Client)
    client.conn = conn
    for timeout in (None, 7):
        with pytest.raises(requests.ConnectionError):
            conn._session.request("GET", "http://127.0.0.1:9/x", timeout=timeout)
    with client.request_timeout(read_s=3600):
        with pytest.raises(requests.ConnectionError):
            conn._session.request("GET", "http://127.0.0.1:9/x", timeout=None)
    assert seen == [(30.0, 600.0), 7, (30.0, 3600)]


# --- per-request statuses, pools, flags, LRU ---------------------------------------------------


def test_per_request_failures_become_none_and_are_counted() -> None:
    keys = [root(i) for i in range(5)]
    server = ContextServer({keys[1]: "history_capacity_exceeded", keys[3]: "missing_entity"})
    store = StreamingContextSource(server, plan=PLAN, sampler=SAMPLER, request_batch_size=16)
    rows = store.fetch(keys + [keys[1]])
    assert [row is None for row in rows] == [False, True, False, True, False, True]
    assert store.rejections == Counter({"history_capacity_exceeded": 1, "missing_entity": 1})
    assert rows[0] is not None and rows[0]["node_id"] == keys[0].node_id
    # Cached rejections are served without another call and counted again.
    assert store.fetch([keys[3]]) == [None] and store.query_calls == 1
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


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"status": "scope_not_ready"}], "rejected the context call"),
        ([{"status": "surprise", "request_index": 0}], "Unknown per-request status"),
        ([], "Incomplete"),
        ([{"status": "missing_entity", "request_index": 1}], "Incomplete"),
    ],
)
def test_call_level_errors_and_malformed_responses_raise(
    rows: list[dict[str, Any]], message: str
) -> None:
    fake = Runner(lambda name, params: rows)
    with pytest.raises(ValueError, match=message):
        query_context_batch(fake, [root(0)], plan=PLAN, sampler=SAMPLER)


def test_hop_pools_and_flags_are_sent_and_lru_is_keyed_by_hop() -> None:
    plan = FeaturePlan(("entity_meta", "message_core", "time_encoding", "rolling_windows"), "split")
    server = ContextServer()
    store = StreamingContextSource(server, plan=plan, sampler=SAMPLER, capacity=8)
    key = root(0)
    store.fetch([key], hop=1)
    store.fetch([key], hop=2)
    store.fetch([key], hop=1)
    assert store.query_calls == 2 and len(server.calls) == 2
    first, second = server.calls
    assert {k: first[k] for k in SAMPLER.query_params(1)} == SAMPLER.query_params(1)
    assert {k: second[k] for k in SAMPLER.query_params(2)} == SAMPLER.query_params(2)
    assert first["include_rolling_windows"] and not second["include_rolling_windows"]
    assert {k for k in first if k.startswith("include_")} == set(plan.query_flags(1))
    assert (1, key) in store.memory and (2, key) in store.memory
    with pytest.raises(ValueError, match="hop"):
        store.fetch([key], hop=3)
    store.close()


def test_response_bound_is_per_hop() -> None:
    key = root(0)
    many = [event(key.cutoff_seq - 1 - i, key) for i in range(SAMPLER.children.response_bound + 1)]
    row = context_row(key, many, encodings=False)
    validate_context(key, row, PLAN, SAMPLER, 1)
    with pytest.raises(ValueError, match="bound"):
        validate_context(key, row, PLAN, SAMPLER, 2)


def test_unknown_channel_is_counted_but_unknown_rail_is_rejected() -> None:
    key = root(0)
    row = context_row(key, [event(990, key)], encodings=False)
    row["messages"][0]["channel"] = "carrier_pigeon"
    assert validate_context(key, row, PLAN, SAMPLER) == 1
    row["messages"][0]["rail"] = "carrier_pigeon"
    with pytest.raises(ValueError, match="rail"):
        validate_context(key, row, PLAN, SAMPLER)


def test_lru_is_bounded_and_close_releases_it() -> None:
    store = StreamingContextSource(ContextServer(), plan=PLAN, sampler=SAMPLER, capacity=8)
    for start in range(0, 64, 16):
        store.fetch([root(i) for i in range(start, start + 16)])
        assert len(store.memory) <= 8
    # Recency follows key order, not request completion order, and rows carry no
    # request position, so a refetch in another grouping returns identical rows.
    keys = [root(i) for i in range(100, 180)]
    rows = store.fetch(keys)
    assert list(store.memory) == [(1, key) for key in keys[-8:]]
    calls = store.query_calls
    assert store.fetch(keys[-8:]) == rows[-8:] and store.query_calls == calls
    assert store.fetch(keys[:2]) == rows[:2] and "request_index" not in (rows[0] or {})
    store.close()
    assert not store.memory
    with pytest.raises(RuntimeError, match="closed"):
        store.fetch([root(0)])


# --- thread safety and in-flight window ------------------------------------------------------


def test_concurrent_fetches_share_requests_and_respect_concurrency() -> None:
    server = ContextServer(delay=0.01)
    store = StreamingContextSource(
        server, plan=PLAN, sampler=SAMPLER, capacity=4096, request_batch_size=4, concurrency=3
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
    assert store.query_calls == len(server.calls)
    store.close()


def test_failed_request_propagates_to_every_waiting_fetch() -> None:
    class Failing(ContextServer):
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            time.sleep(0.05)
            raise ValueError("contract violation")

    store = StreamingContextSource(Failing(), plan=PLAN, sampler=SAMPLER, concurrency=2)
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
    store = StreamingContextSource(
        server, plan=PLAN, sampler=SAMPLER, request_batch_size=1, concurrency=1
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
    workers = [t for t in threading.enumerate() if t.name.startswith("temporal-context")]
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
                raise ServerTimeoutError("temporal_training_context timed out")
            return super().run(name, params)

    keys = [root(i) for i in range(8)]
    server = Slow(limit=2, slow_ids=set())
    store = StreamingContextSource(server, plan=PLAN, sampler=SAMPLER, request_batch_size=8)
    rows = store.fetch(keys)
    assert [row and row["node_id"] for row in rows] == [key.node_id for key in keys]
    assert server.sizes == [8, 4, 2, 2, 4, 2, 2]
    assert store.query_calls == 4 and store.diagnostics["timeout_splits"] == 3
    store.close()
    server = Slow(limit=8, slow_ids={keys[5].node_id})
    store = StreamingContextSource(server, plan=PLAN, sampler=SAMPLER, request_batch_size=8)
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


# --- encoding spot checks ---------------------------------------------------------------------


def test_encoding_spot_checks_follow_the_cadence_and_are_stripped() -> None:
    server = ContextServer()
    store = StreamingContextSource(
        server, plan=PLAN, sampler=SAMPLER, request_batch_size=1, concurrency=1,
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
    for server, message in (
        (ContextServer(corrupt=True), "shared basis"),
        (ContextServer(omit_encodings=True), "do not cover"),
    ):
        store = StreamingContextSource(server, plan=PLAN, sampler=SAMPLER)
        with pytest.raises(ValueError, match=message):
            store.fetch([root(0)])
        store.close()
    key = root(0)
    row = context_row(key, [event(990, key)], encodings=True)
    validate_context(key, row, PLAN, SAMPLER)  # optional vectors are verified when present
    row["gap_encoding"]["payment_out:E990"][0] += 0.5
    with pytest.raises(ValueError, match="encoding"):
        validate_context(key, row, PLAN, SAMPLER)


# --- SQLite store --------------------------------------------------------------------------


def test_context_store_keys_by_hop_caches_rejections_and_reads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = [root(i) for i in range(3)]
    server = ContextServer({keys[2]: "invisible_entity"})
    store = ContextStore(
        tmp_path / "c.sqlite", {"dataset": "x"}, server, plan=PLAN, sampler=SAMPLER
    )
    reads = Counter()
    original = ContextStore._read

    def counting(self: ContextStore, hop: int, key: ContextKey) -> dict[str, Any] | None:
        reads[(hop, key)] += 1
        return original(self, hop, key)

    monkeypatch.setattr(ContextStore, "_read", counting)
    first = store.fetch(keys, hop=1)
    store.fetch(keys[:1], hop=2)
    assert store.query_calls == 2 and first[2] is None
    assert max(reads.values()) == 1
    store.close()
    offline = ContextStore(tmp_path / "c.sqlite", {"dataset": "x"}, plan=PLAN, sampler=SAMPLER)
    again = offline.fetch(keys, hop=1)
    assert again[2] is None and again[0] == first[0] and offline.rejections["invisible_entity"] == 1
    assert offline.fetch(keys[:1], hop=2)[0] is not None
    with pytest.raises(ValueError, match="Offline"):
        offline.fetch(keys[1:2], hop=2)
    offline.close()


# --- hub registry ---------------------------------------------------------------------------


def hub_row(account: str, cutoff: int, phase: int = 3, **changes: Any) -> dict[str, Any]:
    return {
        "account_id": account,
        "cutoff_seq": cutoff,
        "visibility_phase": phase,
        "max_visible": 5000,
        "max_degree": 9000,
        "reason": "visible_history",
        **changes,
    }


def hub_rows(cutoffs: list[int], scope_id: str = "") -> list[dict[str, Any]]:
    if scope_id:
        hubs = [hub_row("H1", cutoffs[0], 2), hub_row("H1", cutoffs[0], 3)]
        hubs.append(hub_row("H2", cutoffs[1], 1))
    else:
        hubs = [hub_row("H1", cutoffs[0]), hub_row("H2", cutoffs[1])]
    return [
        {
            "status": "ok",
            "cutoff_seqs": cutoffs,
            "threshold": 1024,
            "scope_id": scope_id,
            "candidates": 2,
            "hubs": hubs,
        }
    ]


def test_hub_registry_parse_save_load_and_stub_semantics(tmp_path: Path) -> None:
    cutoffs = [1000, 2000]
    calls = []

    def run(name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append((name, params))
        return hub_rows(cutoffs, params["scope_id"])

    fake = Runner(run)
    registry = query_hub_registry(fake, [2000, 1000], threshold=1024)
    assert calls == [
        ("temporal_hub_registry", {"cutoff_seqs": cutoffs, "threshold": 1024, "scope_id": ""})
    ]
    assert registry.is_stub("Account", "H1", 1000) and not registry.is_stub("Account", "H1", 2000)
    assert registry.is_stub("Account", "H2", 2000, 3)
    assert not registry.is_stub("Token", "H1", 1000)
    with pytest.raises(ValueError, match="does not cover"):
        registry.is_stub("Account", "H1", 1500)
    with pytest.raises(ValueError, match="phases 3, not 1"):
        registry.is_stub("Account", "H1", 1000, 1)  # a scoped batch needs a scoped registry
    assert registry.counts() == {"1000": {"3": 1}, "2000": {"3": 1}}
    # A scoped registry is keyed by phase: held-out events can only add later-phase rows.
    scoped = query_hub_registry(fake, cutoffs, threshold=1024, scope_id="scope")
    assert calls[-1][1]["scope_id"] == "scope"
    assert [scoped.is_stub("Account", "H1", 1000, phase) for phase in (1, 2, 3)] == [
        False,
        True,
        True,
    ]
    assert scoped.is_stub("Account", "H2", 2000, 1) and not scoped.is_stub("Account", "H2", 2000)
    path = tmp_path / "hubs.parquet"
    scoped.save(path)
    assert tuple(pd.read_parquet(path).columns) == HUB_COLUMNS
    manifest = {
        "cutoff_seqs": {"2024-07-01": 1000, "2024-10-01": 2000},
        "config": {"evaluation_protocol": "strict_inductive", "scope_id": "scope"},
        **hub_manifest(scoped, path),
    }
    assert manifest["hub_scope_id"] == "scope" and "hub_scan_cap" not in manifest
    assert manifest["hub_counts"] == {
        "1000": {"1": 0, "2": 1, "3": 1},
        "2000": {"1": 1, "2": 0, "3": 0},
    }
    loaded = load_hub_registry(tmp_path, manifest)
    assert loaded.is_stub("Account", "H1", 1000, 2) and len(loaded) == 3
    with pytest.raises(ValueError, match="computed for scope 'scope'"):
        load_hub_registry(tmp_path, {**manifest, "config": {"evaluation_protocol": "x"}})
    with pytest.raises(ValueError, match="no scoped hub registry"):
        load_hub_registry(tmp_path, {k: v for k, v in manifest.items() if k != "hub_scope_id"})
    HubRegistry(loaded.frame.iloc[:1], cutoff_seqs=cutoffs, threshold=1, scope_id="scope").save(
        path
    )
    with pytest.raises(ValueError, match="changed"):
        load_hub_registry(tmp_path, manifest)
    empty = HubRegistry.empty()
    assert not empty.is_stub("Account", "H1", 123, 1) and len(empty) == 0
    assert hub_threshold(SAMPLER) == 1024


@pytest.mark.parametrize(
    "scope_id, change",
    [
        ("", {"reason": "scan_cost"}),
        ("", {"cutoff_seq": 3000}),
        # All-time degree never makes a hub: a row whose visible count is within the
        # threshold is rejected however large the account grows after the cutoff.
        ("", {"max_visible": 10, "max_degree": 300_000}),
        ("", {"max_visible": 1024}),
        ("", {"visibility_phase": 1}),  # unscoped rows are phase 3
        ("scope", {"visibility_phase": 4}),
        ("scope", {"visibility_phase": 0}),
        ("", {"max_degree": -1}),
    ],
)
def test_hub_registry_rejects_contract_violations(scope_id: str, change: dict[str, Any]) -> None:
    rows = hub_rows([1000, 2000], scope_id)
    rows[0]["hubs"][0].update(change)
    with pytest.raises(ValueError, match="contract"):
        query_hub_registry(
            Runner(lambda n, p: rows), [1000, 2000], threshold=1024, scope_id=scope_id
        )


def test_hub_registry_rejects_stale_or_mismatched_responses() -> None:
    def check(rows: list[dict[str, Any]], message: str, scope_id: str = "") -> None:
        with pytest.raises(ValueError, match=message):
            query_hub_registry(
                Runner(lambda n, p: rows), [1000, 2000], threshold=1024, scope_id=scope_id
            )

    rows = hub_rows([1000, 2000])
    rows[0]["threshold"] = 2048
    check(rows, "echoed threshold")
    check(hub_rows([1000, 2000], "other"), "echoed scope_id", "scope")
    rows = hub_rows([1000, 2000])
    rows[0]["scan_cap"] = 262144  # the old query with the all-time degree decision
    check(rows, "mule-temporal install")
    rows = hub_rows([1000, 2000])
    del rows[0]["hubs"][0]["visibility_phase"]
    check(rows, "Malformed")
    check([{"status": "scope_not_ready"}], "rejected", "scope")
    rows = hub_rows([1000, 2000])
    rows[0]["hubs"].append(dict(rows[0]["hubs"][0]))
    check(rows, "Duplicate")


# --- preparation --------------------------------------------------------------------------


def live_config(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    labels = tmp_path / "labels.parquet"
    if not labels.exists():
        pd.DataFrame(
            {"account_id": ["A1"], "known_positive": [True], "known_from_ms": [5]}
        ).to_parquet(labels)
    config = {
        "dataset_id": "unit_snapshot",
        "evaluation_protocol": "strict_inductive",
        "scope_id": "unit_scope",
        "label_policy": "observed",
        "observed_labels": str(labels),
        "dates": {"train": ["2024-07-01"], "validation": ["2024-10-01"], "test": ["2025-01-01"]},
        "sampler": {"policy": "stratified", "recent": 4, "older": 2, "distinct": 1},
    }
    return {**config, **changes}


@pytest.fixture
def fixed_hashes(monkeypatch: pytest.MonkeyPatch):
    hashes = {"gsql/temporal/training_context.gsql": "aaa", "gsql/temporal/hub_registry.gsql": "b"}
    monkeypatch.setattr(dataset, "query_hashes", lambda: dict(hashes))
    return hashes


def write_manifest(
    path: Path, config: dict[str, Any], hashes: dict[str, Any], status: str = "ready"
) -> dict[str, Any]:
    manifest = {
        "status": status,
        "config": config,
        "source": {"query_hashes": dict(hashes), "preparation": dataset.preparation_view(config)},
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_prepare_live_checks_query_hashes_before_reusing_a_ready_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_hashes: dict[str, str]
) -> None:
    config = live_config(tmp_path)
    out = tmp_path / "prepared"
    manifest = write_manifest(out, config, fixed_hashes)

    def no_connection(config: dict[str, Any]) -> None:
        raise AssertionError("prepare_live must not connect for a ready dataset")

    monkeypatch.setattr(pipeline, "live_executor", no_connection)
    assert pipeline.prepare_live(config, out) == manifest
    # Model and transport settings are not preparation settings.
    assert pipeline.prepare_live({**config, "learning_rate": 0.1, "query_concurrency": 2}, out)
    with pytest.raises(ValueError, match=r"split_seed.*new output"):
        pipeline.prepare_live({**config, "split_seed": 7}, out)
    fixed_hashes["gsql/temporal/hub_registry.gsql"] = "changed"
    with pytest.raises(ValueError, match=r"hub_registry\.gsql.*mule-temporal install.*new output"):
        pipeline.prepare_live(config, out)
    with pytest.raises(ValueError, match="different GSQL sources"):
        dataset.load_prepared(out)


def test_prepared_directory_is_inside_the_run_unless_prepared_id_is_set(tmp_path: Path) -> None:
    assert pipeline.dataset_path({}, tmp_path / "m.pt") == tmp_path / "m_run" / "prepared"
    shared = pipeline.dataset_path({"prepared_id": "p2"}, tmp_path / "m.pt")
    assert shared.name == "p2" and shared.parent.name == "temporal"


def test_dataset_identity_comes_from_the_scope_or_the_graph(tmp_path: Path) -> None:
    counts = {"Account": 10, "Party": 4}
    header = {"ready": True, "source_id": "unit_snapshot", "split_seed": 42}
    config = {k: v for k, v in live_config(tmp_path).items() if k != "dataset_id"}
    server = ScopeServer(header, "linked")
    server.client.conn.graphname = "G"
    assert pipeline.resolve_identity(cast(Any, server), config, counts)["dataset_id"] == (
        "unit_snapshot"
    )
    fresh = ScopeServer(None, "linked")
    fresh.client.conn.graphname = "G"
    derived = pipeline.resolve_identity(cast(Any, fresh), config, counts)["dataset_id"]
    assert derived == pipeline.derived_dataset_id("G", counts) and derived.startswith("G_")
    assert derived != pipeline.derived_dataset_id("G", {**counts, "Account": 11})
    explicit = {**config, "dataset_id": "pinned"}
    assert pipeline.resolve_identity(cast(Any, fresh), explicit, counts) == explicit
    # With an existing dataset the identity comes from its manifest, and a pin is kept.
    manifest = {"source": {"dataset_id": "unit_snapshot"}}
    assert pipeline.prepared_config(config, manifest)["dataset_id"] == "unit_snapshot"
    assert pipeline.prepared_config(explicit, manifest)["dataset_id"] == "pinned"


def test_only_strict_runs_reveal_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reveals: list[str] = []
    monkeypatch.setattr(pipeline, "live_executor", lambda config: SimpleNamespace())
    monkeypatch.setattr(pipeline, "install", lambda executor: None)
    monkeypatch.setattr(pipeline, "source_counts", lambda executor: {"Account": 10})
    monkeypatch.setattr(pipeline, "ensure_scope", lambda executor, config: None)
    monkeypatch.setattr(
        pipeline,
        "ensure_revealed_labels",
        lambda executor, config: reveals.append(config["evaluation_protocol"]),
    )
    monkeypatch.setattr(pipeline, "prepare", lambda config, *args, **kwargs: {"status": "ready"})
    config = {
        **live_config(tmp_path, dataset_id="unit_snapshot"),
        "label_policy": "graph_observed",
        "observed_labels": None,
    }
    for protocol in ("shared_history", "strict_inductive"):
        run = {**config, "evaluation_protocol": protocol}
        assert pipeline.prepare_live(run, tmp_path / protocol) == {"status": "ready"}
    # A shared_history run reads whatever labels the graph has.
    assert reveals == ["strict_inductive"]


def test_preparation_keys_fingerprint_only_preparation_settings(tmp_path: Path) -> None:
    base = live_config(tmp_path)
    view = dataset.preparation_view(base)
    assert tuple(view) == dataset.PREPARATION_KEYS and "hub_scan_cap" not in view
    same = [
        {"learning_rate": 0.5, "epochs": 3, "hidden": 8, "request_batch_size": 4},
        {"scope_unowned": "linked", "context_storage": "stream", "max_outage_s": 60},
        {"sampler": {**base["sampler"], "association_slots": 1}},  # selection, not pools
    ]
    for change in same:
        assert dataset.preparation_fingerprint(
            {**base, **change}
        ) == dataset.preparation_fingerprint(base), change
    assert dataset.preparation_fingerprint(
        validate_config(base)
    ) == dataset.preparation_fingerprint(base)
    different = [
        {"split_seed": 1},
        {"seed": 1},
        {"prepared_id": "again"},
        {"dates": {**base["dates"], "test": ["2025-02-01"]}},
        {"scope_unowned": "independent"},
        {"scope_unowned": "shared"},
        {"sampler": {**base["sampler"], "children": {"recent": 2, "associations": 0}}},
        {"extraction_groups": [*LEGACY_GROUPS, "pair_history"]},
        {"context_storage": "sqlite"},
    ]
    for change in different:
        assert dataset.preparation_fingerprint(
            {**base, **change}
        ) != dataset.preparation_fingerprint(base), change
    before = dataset.preparation_fingerprint(base)
    pd.DataFrame({"account_id": ["A2"], "known_positive": [True], "known_from_ms": [9]}).to_parquet(
        base["observed_labels"]
    )
    assert dataset.preparation_fingerprint(base) != before
    manifest = {"source": {"preparation": view}}
    assert dataset.preparation_mismatches(base, manifest) == ["observed_labels"]
    # A missing label source is reported as missing, not as a changed setting.
    Path(base["observed_labels"]).unlink()
    with pytest.raises(ValueError, match="Observed-label source file not found at .*labels"):
        dataset.preparation_mismatches(base, manifest)


LINKED_COUNTS = {
    "shared_internal": 0,
    "shared_external": 40,
    "independent_internal": 7,
    "independent_external": 0,
    "linked_internal": 90,
    "linked_external": 0,
    "shared_ledger": 4,
    "ledger_accounts": 4,
    "members": 500,
}


def policy_counts(policy: str) -> dict[str, int]:
    """Membership classes a scope created with `policy` reports."""
    if policy == "linked":
        return dict(LINKED_COUNTS)
    if policy == "shared":
        return {**LINKED_COUNTS, "independent_internal": 97, "linked_internal": 0}
    if policy == "independent":  # every scope created before the policy, strict_mule_v1
        return {
            **LINKED_COUNTS,
            "shared_external": 0,
            "independent_external": 40,
            "independent_internal": 97,
            "linked_internal": 0,
            "shared_ledger": 0,
        }
    return {**LINKED_COUNTS, "shared_internal": 97, "linked_internal": 0}  # a retired draft


class ScopeServer:
    """A TigerGraph fake for scope headers, scope creation and temporal_scope_policy."""

    def __init__(self, header: dict[str, Any] | None, policy: str) -> None:
        self.header, self.policy = header, policy
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.client = SimpleNamespace(conn=SimpleNamespace(getVerticesById=self.vertices))

    def vertices(self, *args: Any) -> list[dict[str, Any]]:
        if self.header is None:
            raise TigerGraphException("vertex not found", "601")
        return [{"attributes": dict(self.header)}]

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        if name == "temporal_scope_policy":
            return [{"status": "ok", "scope_id": params["scope_id"], **policy_counts(self.policy)}]
        if name == "temporal_create_training_scope":
            self.policy = params["unowned_policy"]
            return [{"status": "ok", "expected_members": 3}]
        if name == "temporal_finalize_training_scope":
            self.header = {"ready": True, "source_id": "unit_snapshot", "split_seed": 42}
        return [{"status": "ok"}]


def test_source_counts_ignore_experiment_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"Account": 10, "Party": 4, "Temporal_Training_Scope": 1}
    header = {"ready": True, "source_id": "snap", "split_seed": 42}
    conn = SimpleNamespace(
        getVertexCount=lambda *a, **k: dict(counts),
        getVerticesById=lambda *a: [{"attributes": dict(header)}],
        runInstalledQuery=lambda name, params, **k: [{"status": "ok", **policy_counts("linked")}],
    )
    tg = executor(conn)
    assert installation.source_counts(tg) == {"Account": 10, "Party": 4}
    monkeypatch.setattr(installation, "verify_sources", lambda executor: [])
    manifest: dict[str, Any] = {
        "config": {
            "dataset_id": "snap",
            "scope_id": "s",
            "evaluation_protocol": "strict_inductive",
        },
        # Older manifests recorded the scope vertex count too.
        "source": {"source_counts": {"Account": 10, "Party": 4, "Temporal_Training_Scope": 1}},
    }
    installation.verify_frozen_source(tg, manifest)
    counts["Temporal_Training_Scope"] = 3  # another experiment created scopes
    installation.verify_frozen_source(tg, manifest)
    # The scope's unowned rule is rechecked on every streamed run.
    manifest["config"]["scope_unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope_unowned = 'linked'"):
        installation.verify_frozen_source(tg, manifest)
    manifest["config"]["scope_unowned"] = "linked"
    counts["Account"] = 11
    with pytest.raises(ValueError, match="counts changed"):
        installation.verify_frozen_source(tg, manifest)


def test_missing_scope_is_created_unless_forbidden(tmp_path: Path) -> None:
    config = live_config(tmp_path)
    server = ScopeServer(None, "independent")
    with pytest.raises(ValueError, match="create_scope = false"):
        scope.ensure_scope(cast(Any, server), {**config, "create_scope": False})
    assert not server.calls
    scope.ensure_scope(cast(Any, server), config)
    names = [call[0] for call in server.calls]
    assert names == [
        "temporal_create_training_scope",
        "temporal_finalize_training_scope",
        "temporal_scope_policy",
    ]
    create = server.calls[0]
    assert create[1]["unowned_policy"] == "linked" and create[2]["attempts"] == 1
    assert "shared_unowned" not in create[1]
    assert server.calls[1][1] == {"scope_id": "unit_scope", "expected_members": 3}
    for policy in ("independent", "shared"):
        server = ScopeServer(None, "independent")
        changed = {**config, "scope_unowned": policy}
        scope.ensure_scope(cast(Any, server), changed)
        assert server.calls[0][1]["unowned_policy"] == policy


def test_existing_scope_must_have_the_configured_unowned_policy(tmp_path: Path) -> None:
    header = {"ready": True, "source_id": "unit_snapshot", "split_seed": 42}
    config = live_config(tmp_path)
    for policy in ("independent", "shared", "linked"):
        assert scope.inferred_scope_policy(policy_counts(policy)) == policy
        server = ScopeServer(header, policy)
        scope.ensure_scope(cast(Any, server), {**config, "scope_unowned": policy})
        assert [call[0] for call in server.calls] == ["temporal_scope_policy"]
    assert scope.inferred_scope_policy(policy_counts("retired")) is None
    # Without unowned external accounts a linked scope is still recognised by its links.
    no_external = {**policy_counts("linked"), "shared_external": 0, "independent_external": 0}
    assert scope.inferred_scope_policy(no_external) == "linked"
    alone = {**no_external, "linked_internal": 0, "shared_ledger": 0}
    assert scope.inferred_scope_policy(alone) == "independent"
    # Bank ledger accounts are shared exactly when external accounts are: all or none.
    assert scope.inferred_scope_policy({**alone, "shared_ledger": 4}) == "shared"
    for ledger_shared in (0, 3):  # a linked scope that left some ledger books partitioned
        mixed = {**policy_counts("linked"), "shared_ledger": ledger_shared}
        assert scope.inferred_scope_policy(mixed) is None
    # Linking without sharing the external accounts is no rule.
    unshared = {**policy_counts("linked"), "shared_external": 0, "independent_external": 3}
    assert scope.inferred_scope_policy(unshared) is None
    partly = {**policy_counts("shared"), "independent_external": 1}
    assert scope.inferred_scope_policy(partly) is None
    # A pre-policy scope (strict_mule_v1) under the default "linked" configuration.
    with pytest.raises(ValueError, match=r"created with scope_unowned = 'independent'.*set a new"):
        scope.ensure_scope(cast(Any, ScopeServer(header, "independent")), config)
    with pytest.raises(ValueError, match="matches no scope_unowned rule"):
        scope.ensure_scope(cast(Any, ScopeServer(header, "retired")), config)
    with pytest.raises(ValueError, match="different source"):
        scope.ensure_scope(cast(Any, ScopeServer({**header, "split_seed": 7}, "linked")), config)
    with pytest.raises(ValueError, match="lacks"):
        scope.scope_policy_counts(Runner(lambda n, p: [{"status": "ok", "members": 3}]), "s")


def test_scope_policy_query_prints_what_the_client_reads() -> None:
    text = (REPOSITORY_ROOT / "gsql/temporal/training_scope.gsql").read_text()
    queries = installation.definitions(text)
    assert scope.SCOPE_POLICY_QUERY in queries
    query = queries[scope.SCOPE_POLICY_QUERY]
    assert installation.parameter_names(query) == {"scope_id"}
    for name in (*scope.SCOPE_POLICY_COUNTS, "members"):
        assert f"AS {name}" in query, name
    create = installation.parameter_names(queries["temporal_create_training_scope"])
    assert "unowned_policy" in create and "shared_unowned" not in create


# --- labels ----------------------------------------------------------------------------------


def test_label_source_must_be_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No observed-label source"):
        label_source({})
    with pytest.raises(ValueError, match="No observed-label source"):
        label_source({"label_policy": "observed"})
    assert isinstance(label_source({"label_policy": "graph_observed"}), GraphObservedLabels)
    with pytest.raises(ValueError, match="conflicts"):
        label_source({"label_policy": "graph_observed", "observed_labels": "x.parquet"})
    with pytest.raises(ValueError, match="source file not found at .*absent.*observed_labels"):
        label_source({"observed_labels": str(tmp_path / "absent.parquet")})
    config = live_config(tmp_path)
    labels = label_source(config)
    assert isinstance(labels, ParquetObservedLabels) and labels.path == Path(
        config["observed_labels"]
    )
    assert resolve_path("configs/x.toml") == REPOSITORY_ROOT / "configs/x.toml"
    assert resolve_path(tmp_path) == tmp_path


def population_row(account: str, positive: bool, known: int) -> dict[str, Any]:
    return {
        "account_id": account,
        "partition": 1,
        "group_id": "g",
        "first_seen_ts_ms": 1,
        "observed_positive": positive,
        "known_from_ms": known,
    }


def test_only_graph_label_policy_reads_graph_labels(tmp_path: Path) -> None:
    from mule_pattern_learner.temporal.live.cohort import scoped_cohort

    seen = []

    def run(name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        seen.append(params["include_observed"])
        graph = params["include_observed"]
        row = population_row("A1", graph, 5 if graph else 0)
        return [{"status": "ok", "accounts": [row]}]

    fake = Runner(run)
    config = live_config(tmp_path)
    scoped_cohort(fake, config, label_source(config))
    frame, _ = scoped_cohort(fake, config, GraphObservedLabels())
    assert seen == [False, True] and frame.in_marginal.tolist() == [True]
    with pytest.raises(ValueError, match="explicit"):
        scoped_cohort(fake, config, None)
    with pytest.raises(ValueError, match="No observed-label source"):
        dataset.prepare(
            {k: v for k, v in config.items() if k != "observed_labels"},
            tmp_path / "out",
            fake,
            {"Account": 1},
        )
    assert seen == [False, True]  # failed before any query


def test_stale_population_queries_fail_fast(tmp_path: Path) -> None:
    from mule_pattern_learner.temporal.live.cohort import scoped_cohort

    config = live_config(tmp_path)
    # An old query emits the discovery time of hidden or negative labels.
    stale = Runner(lambda n, p: [{"status": "ok", "accounts": [population_row("A1", False, 5)]}])
    with pytest.raises(ValueError, match="predates the masked-label predicate"):
        scoped_cohort(stale, config, GraphObservedLabels())
    # Without include_observed the query must return no label information.
    leaky = Runner(lambda n, p: [{"status": "ok", "accounts": [population_row("A1", True, 5)]}])
    with pytest.raises(ValueError, match="include_observed is false"):
        scoped_cohort(leaky, config, label_source(config))
    metadata = pd.DataFrame(
        {
            "account_id": ["A1", "A2", "A3"],
            "split": ["train", "train", "test"],
            "observed_positive": [True, False, False],
            "known_from_ms": [5, 0, 7],
        }
    )
    with pytest.raises(ValueError, match="1 account.*'A3'.*mule-temporal install"):
        GraphObservedLabels().read(metadata)
    metadata.loc[2, "known_from_ms"] = 0
    labels = GraphObservedLabels().read(metadata)
    assert labels.known_positive.tolist() == [True, False, False]
    assert labels.pu_label.tolist() == [1, 0, 0]


# --- configuration schema ------------------------------------------------------------------


def test_config_schema_rejects_unknown_keys_and_applies_operational_defaults(
    tmp_path: Path,
) -> None:
    base = live_config(tmp_path)
    result = validate_config(base)
    for key, value in OPERATIONAL_DEFAULTS.items():
        assert result[key] == value
    for key in ("fanouts", "feature_groups", "extraction_groups", "per_relation", "prepared_id"):
        assert key not in result
    assert result["sampler"] == base["sampler"] and base == live_config(tmp_path)
    with pytest.raises(ValueError, match="Unknown configuration key.*learnig_rate"):
        validate_config({**base, "learnig_rate": 0.1})
    bad = [
        ({"fanouts": [True, 4]}, "fanouts"),
        ({"fanouts": [8]}, "fanouts"),
        ({"query_concurrency": 17}, "query_concurrency"),
        ({"request_batch_size": 65}, "request_batch_size"),
        ({"deterministic": "yes"}, "deterministic"),
        ({"context_storage": "disk"}, "context_storage"),
        ({"label_policy": "oracle"}, "label_policy"),
        ({"prepared_id": "../escape"}, "prepared_id"),
        ({"sampler": {"recnt": 2}}, "sampler.recnt"),
        ({"sampler": {"children": {"older": 99}}}, "sampler.children.older"),
        ({"dates": {**base["dates"], "train": ["not a date"]}}, "dates"),
        ({"feature_groups": ["entity_meta", "no_such_group"]}, "no_such_group"),
    ]  # fmt: skip
    for change, name in bad:
        with pytest.raises(ValueError, match=name):
            validate_config({**base, **change})
    assert result["scope_unowned"] == "linked" and result["max_outage_s"] == 900
    assert "hub_scan_cap" not in result
    for change, name in [
        ({"scope_unowned": "all"}, "scope_unowned"),
        ({"max_outage_s": -1}, "max_outage_s"),
        ({"max_outage_s": 1.5}, "max_outage_s"),
        ({"hub_scan_cap": 262144}, "Unknown configuration key.*hub_scan_cap"),
    ]:
        with pytest.raises(ValueError, match=name):
            validate_config({**base, **change})
    for policy in ("independent", "shared", "linked"):
        assert validate_config({**base, "scope_unowned": policy})["scope_unowned"] == policy
    # Checkpoint configurations used for scoring need not carry preparation keys.
    assert validate_config({"hidden": 16})["query_concurrency"] == 16
    assert validate_config({**base, "deterministic": "strict"})["deterministic"] == "strict"
    assert validate_config({**base, "positive_weight": "prior"})["positive_weight"] == "prior"


def test_built_in_run_validates_and_only_run_config_applies_the_schema(tmp_path: Path) -> None:
    config = run_config()
    assert config == validate_config(config) and config["request_batch_size"] == 8
    assert {key: config[key] for key in DEFAULT_RUN} == DEFAULT_RUN
    # The field defaults are the operational defaults validate_config fills in.
    for key, value in OPERATIONAL_DEFAULTS.items():
        assert LiveConfig.model_fields[key].default == value, key
    toml = tmp_path / "c.toml"
    toml.write_text('dataset_id = "x"\nstage = "offline-only key"\n')
    assert load_config(toml)["stage"] == "offline-only key"
    with pytest.raises(ValueError, match="Unknown configuration key.*stage"):
        run_config(toml)


def test_override_tables_merge_into_the_built_in_run(tmp_path: Path) -> None:
    def overridden(text: str) -> dict[str, Any]:
        path = tmp_path / "overrides.toml"
        path.write_text(text)
        return run_config(path)

    default = run_config()
    torch_only = overridden('[sampler]\nbackend = "torch"\n')
    assert torch_only["sampler"] == {**DEFAULT_RUN["sampler"], "backend": "torch"}
    assert SamplerPlan.from_config(torch_only).fingerprint() == (
        SamplerPlan.from_config(default).fingerprint()
    )
    # A pool setting keeps the default policy and every other sampler key.
    fewer = overridden("[sampler]\nrecent = 4\n[sampler.children]\nolder = 1\n")
    assert fewer["sampler"]["policy"] == "resample" and fewer["sampler"]["recent"] == 4
    assert fewer["sampler"]["children"] == {**DEFAULT_RUN["sampler"]["children"], "older": 1}
    assert fewer["sampler"]["relation_fanouts"] == DEFAULT_RUN["sampler"]["relation_fanouts"]
    # Another policy starts from the override's table alone.
    recent = overridden('[sampler]\npolicy = "recent"\nrecent = 4\n')
    assert recent["sampler"] == {"policy": "recent", "recent": 4}
    dates = overridden('[dates]\ntrain = ["2024-05-01", "2024-07-01"]\n')
    assert dates["dates"] == {**DEFAULT_RUN["dates"], "train": ["2024-05-01", "2024-07-01"]}
    # Lists and scalars replace the default.
    assert overridden("fanouts = [8, 2]\nepochs = 3\n")["fanouts"] == [8, 2]
    assert overridden('observed_labels = "x.parquet"\n')["label_policy"] == "observed"
    assert run_config() == default


def test_transport_settings_come_from_the_training_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    monkeypatch.setattr(source, "verify_frozen_source", lambda executor, manifest: None)

    class Executor:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("mule_pattern_learner.temporal.live.executor.TigerGraphExecutor", Executor)
    prepared = {"dataset_id": "d", "evaluation_protocol": "shared_history"}
    manifest = {"config": prepared, "source": {"context_storage": "stream"}}
    training = {
        **prepared,
        "request_batch_size": 32,
        "query_concurrency": 4,
        "context_lru_capacity": 1024,
        "encoding_check_every": 8,
        "max_query_attempts": 3,
        "max_outage_s": 120,
    }
    store = cast(
        StreamingContextSource, source.open_context_source(Path("unused"), manifest, training)
    )
    assert seen == {"max_attempts": 3, "max_outage_s": 120}
    assert (store.request_batch_size, store.concurrency, store.capacity) == (32, 4, 1024)
    assert store._cadence.every == 8
    store.close()
    changed = {**training, "sampler": {"recent": 5}}
    with pytest.raises(ValueError, match="pools differ"):
        source.open_context_source(Path("unused"), manifest, changed)


# --- installation checks ---------------------------------------------------------------------


def endpoint(parameters: set[str], enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "parameters": {name: {} for name in parameters | {"query", "read_committed"}},
    }


def test_verify_sources_requires_matching_text_and_enabled_endpoints() -> None:
    files = ("gsql/temporal/training_cutoffs.gsql", "gsql/features/temporal_fourier64.gsql")
    expected: dict[str, str] = {}
    for path in files:
        expected.update(installation.definitions((REPOSITORY_ROOT / path).read_text()))
    params = {name: installation.parameter_names(text) for name, text in expected.items()}
    assert params["temporal_training_cutoffs"] == {"cutoff_times"}

    def conn(
        text_for: Callable[[str], str], endpoints: dict[str, dict[str, Any]]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            gsql=lambda text: text_for(text.rsplit(" ", 1)[1]),
            getInstalledQueries=lambda: {
                f"GET /query/Mule_Pattern_Learner/{name}": value
                for name, value in endpoints.items()
            },
        )

    good = {name: endpoint(value) for name, value in params.items()}
    assert set(
        installation.verify_sources(executor(conn(expected.__getitem__, good)), files)
    ) == set(expected)
    disabled = {**good, "temporal_training_cutoffs": endpoint({"cutoff_times"}, False)}
    renamed = {**good, "temporal_fourier64": endpoint({"other"})}
    cases = [
        (conn(expected.__getitem__, disabled), "not installed"),
        (conn(lambda n: expected[n].replace("24", "25"), good), "differs"),
        (conn(expected.__getitem__, renamed), "parameters differ"),
        (conn(lambda n: "", good), "missing"),
    ]
    for fake, message in cases:
        with pytest.raises(ValueError, match=message):
            installation.verify_sources(executor(fake), files)


INSTALL_FILES = ("gsql/features/temporal_fourier64.gsql", "gsql/temporal/training_cutoffs.gsql")


class InstallServer:
    """A GSQL server fake: SHOW QUERY, endpoint listing, CREATE OR REPLACE and install.

    `mode` decides how GET /gsql/v1/queries/install answers: "sync" (TigerGraph
    4.2.5: the reply arrives when compilation is done), "timeout" (the client read
    times out; queries become enabled after `ready_after` endpoint listings) or
    "async" (a requestId polled through getQueryInstallationStatus).
    """

    def __init__(
        self, *, stale: tuple[str, ...] = (), mode: str = "sync", ready_after: int = 0
    ) -> None:
        self.queries = installation.repository_queries(INSTALL_FILES)
        self.shown = {name: text for name, (_, text) in self.queries.items()}
        for name in stale:  # an older definition is installed
            self.shown[name] = self.shown[name].replace("{", "{ INT stale_marker = 0;", 1)
        self.enabled = dict.fromkeys(self.queries, True)
        self.mode, self.ready_after = mode, ready_after
        self.created: list[str] = []
        self.installs: list[tuple[list[str], bool]] = []
        self.listings = 0
        self.statuses: list[dict[str, Any]] = []
        self.pending: list[str] = []

    def getSchema(self, force: bool) -> dict[str, Any]:
        return {"VertexTypes": [{"Name": "Temporal_Training_Scope"}]}

    def gsql(self, text: str) -> str:
        if "SHOW QUERY" in text:
            return self.shown.get(text.rsplit(" ", 1)[1], "Query not found")
        self.created.append(text)
        names = list(installation.definitions(text))
        for name in names:
            self.shown[name] = installation.definitions(text)[name]
            self.enabled[name] = False  # CREATE OR REPLACE disables the endpoint
        return f"Successfully created queries: [{', '.join(names)}]."

    def _enable(self) -> None:
        for name in self.pending:
            self.enabled[name] = True

    def installQueries(self, names: list[str], wait: bool) -> dict[str, Any]:
        self.installs.append((list(names), wait))
        self.pending = list(names)
        if self.mode == "timeout":
            raise requests.ReadTimeout("no reply while the server compiles")
        if self.mode == "async":
            return {"requestId": "r1"}
        self._enable()
        return {"error": False, "message": "Query installation finished: SUCCESS"}

    def getQueryInstallationStatus(self, request: str) -> dict[str, Any]:
        status = self.statuses.pop(0)
        if "SUCCESS" in status["message"]:
            self._enable()
        return status

    def getInstalledQueries(self) -> dict[str, Any]:
        self.listings += 1
        if self.pending and self.mode == "timeout" and self.listings > self.ready_after:
            self._enable()
        return {
            f"GET /query/Mule_Pattern_Learner/{name}": endpoint(
                installation.parameter_names(text), self.enabled[name]
            )
            for name, (_, text) in self.queries.items()
        }


def test_install_creates_and_installs_only_stale_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installation, "TRAINING_QUERY_FILES", INSTALL_FILES)
    names = list(installation.repository_queries(INSTALL_FILES))
    assert names == ["temporal_fourier64_values", "temporal_fourier64", "temporal_training_cutoffs"]
    # Everything current: nothing is created or installed.
    server = InstallServer()
    logs = installation.install(executor(server))
    assert logs["installed"] == [] and logs["verified"] == names
    assert not server.created and not server.installs
    # A stale subquery is reinstalled together with its caller, nothing else.
    server = InstallServer(stale=("temporal_fourier64_values",))
    logs = installation.install(executor(server))
    assert logs["installed"] == ["temporal_fourier64_values", "temporal_fourier64"]
    assert logs["up_to_date"] == ["temporal_training_cutoffs"]
    assert server.installs == [(["temporal_fourier64_values", "temporal_fourier64"], False)]
    assert len(server.created) == 1 and "temporal_training_cutoffs" not in server.created[0]
    assert server.created[0].startswith("USE GRAPH Mule_Pattern_Learner\n")
    assert logs["verified"] == names
    # force reinstalls every query.
    server = InstallServer()
    logs = installation.install(executor(server), force=True)
    assert server.installs == [(names, False)] and len(server.created) == 2
    # A disabled endpoint is stale even when the text matches.
    server = InstallServer()
    server.enabled["temporal_training_cutoffs"] = False
    assert installation.install(executor(server))["installed"] == ["temporal_training_cutoffs"]
    # Callers are found in the repository queries too.
    queries = installation.repository_queries(installation.QUERY_FILES)
    assert "temporal_training_context" in installation._with_callers(
        {"temporal_fourier64_values"}, queries
    )
    assert installation._with_callers({"temporal_hub_registry"}, queries) == {
        "temporal_hub_registry"
    }


def test_install_polls_endpoints_when_the_install_request_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installation, "TRAINING_QUERY_FILES", INSTALL_FILES)
    # Listing 1 finds the stale query; listings 2 and 3 still see it compiling.
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="timeout", ready_after=3)
    tg = executor(server)
    logs = installation.install(tg, sleep=tg.clock.sleep, clock=tg.clock.time, poll_s=30)
    assert logs["installed"] == ["temporal_training_cutoffs"] and logs["install"] is None
    assert tg.sleeps == [30, 30] and all(server.enabled.values())
    # Still compiling at the deadline: an actionable timeout, and a later run installs
    # only what is still stale.
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="timeout", ready_after=99)
    tg = executor(server)
    with pytest.raises(TimeoutError, match="still not installed.*re-run `mule-temporal install`"):
        installation.install(
            tg, sleep=tg.clock.sleep, clock=tg.clock.time, poll_s=30, deadline_s=100
        )
    # Other failures of the install request propagate.
    server = InstallServer(stale=("temporal_training_cutoffs",))
    server.installQueries = lambda names, wait: (_ for _ in ()).throw(KeyError("bad"))
    with pytest.raises(KeyError):
        installation.install(executor(server))


def test_install_follows_an_asynchronous_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installation, "TRAINING_QUERY_FILES", INSTALL_FILES)
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="async")
    server.statuses = [{"message": "RUNNING"}, {"message": "Query installation SUCCESS"}]
    sleeps: list[float] = []
    logs = installation.install(executor(server), sleep=sleeps.append, poll_s=5)
    assert logs["install"]["message"].endswith("SUCCESS") and sleeps == [5, 5]
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="async")
    server.statuses = [{"message": "FAILED: type check"}]
    with pytest.raises(RuntimeError, match="failed"):
        installation.install(executor(server), sleep=sleeps.append)
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="async")
    server.statuses = [{"message": "RUNNING"}] * 5
    clock = iter([0.0, 10.0, 99999.0])
    with pytest.raises(TimeoutError, match="still running"):
        installation.install(
            executor(server), sleep=sleeps.append, deadline_s=60, clock=lambda: next(clock)
        )
    server = InstallServer(stale=("temporal_training_cutoffs",))
    server.gsql = lambda text: (
        "Semantic Check Error" if "SHOW QUERY" not in text else "Query not found"
    )
    with pytest.raises(RuntimeError, match="Semantic Check"):
        installation.install(executor(server))


def test_sent_parameters_match_the_repository_query_signatures() -> None:
    def signature(path: str, name: str) -> set[str]:
        text = installation.definitions((REPOSITORY_ROOT / path).read_text())[name]
        return installation.parameter_names(text)

    context = signature("gsql/temporal/training_context.gsql", "temporal_training_context")
    for plan in (PLAN, FeaturePlan(), FeaturePlan(LEGACY_GROUPS, "split")):
        server = ContextServer()
        store = StreamingContextSource(server, plan=plan, sampler=SAMPLER)
        store.fetch([root(0)], hop=1)
        store.fetch([root(0)], hop=2)
        store.close()
        assert all(set(call) == context for call in server.calls)
    calls = []
    query_hub_registry(
        Runner(lambda n, p: calls.append(p) or hub_rows([1000, 2000])),
        [1000, 2000],
        threshold=1024,
    )
    assert set(calls[0]) == signature("gsql/temporal/hub_registry.gsql", "temporal_hub_registry")
    creation = signature("gsql/temporal/training_scope.gsql", "temporal_create_training_scope")
    assert {"scope_id", "source_id", "split_seed", "unowned_policy"} <= creation
    policy = signature("gsql/temporal/training_scope.gsql", scope.SCOPE_POLICY_QUERY)
    assert policy == {"scope_id"}
