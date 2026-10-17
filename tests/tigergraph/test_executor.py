"""The query executor: failure classes, retry budgets, the shared backoff, the size refusal
and request halving, and the entry points that run only the queries their role allows."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

import pytest
from pyTigerGraph.common.exception import TigerGraphException
import requests

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.tigergraph.executor import (
    AVAILABILITY,
    DETERMINISTIC,
    SERVER_TIMEOUT,
    AllowedQueries,
    BenchExecutor,
    EvaluationExecutor,
    FetchExecutor,
    ParityExecutor,
    QueryNotAllowed,
    ResponseTooLarge,
    ServerTimeoutError,
    TigerGraphExecutor,
    TigerGraphUnavailableError,
    TransientQueryError,
    error_summary,
    failure_class,
    failure_summary,
    run_in_halves,
    short_reason,
)
from mule_pattern_learner.tigergraph.queries import (
    BENCH_ONLY,
    EVALUATION_ONLY,
    FETCH_ALLOWLIST,
    QUERY_FILES,
)

# A query that takes a batch of keys, so a retry names how many it asked for.
QUERY = "fetch_subgraphs"


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


class FakeClient:
    """A connected client of a graph over a scripted connection."""

    def __init__(self, conn: Any, graphname: str) -> None:
        self.conn = conn
        self.graphname = graphname


class RecordingExecutor(TigerGraphExecutor):
    """The real retry policy over a fake connection and a fake clock; sleeps are recorded."""

    def __init__(self, conn: Any, graph: str = "Test_Graph", **kwargs: Any) -> None:
        self.clock = FakeClock()
        self.sleeps = self.clock.sleeps
        client = FakeClient(conn, graph)
        super().__init__(client=client, sleep=self.clock.sleep, clock=self.clock.time, **kwargs)


def executor(conn: Any, **kwargs: Any) -> RecordingExecutor:
    return RecordingExecutor(conn, **kwargs)


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
    assert tg.run("q", {"dates": [1]}) == [{"status": "ok"}]
    assert len(conn.calls) == 6
    assert tg.retries["q"] == 5 and tg.calls == 1
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
        (TigerGraphException("Query fetch_x is not installed", "REST-1000"), None),
        (ValueError("Returned context differs"), None),
        (KeyError("results"), None),
    ],
)
def test_failure_classes(error: BaseException, kind: str | None) -> None:
    assert failure_class(error) == kind
    assert (failure_class(error) is not None) is (kind is not None)


# The page a TigerGraph Cloud workspace answers while it starts, as pyTigerGraph raises it.
STARTING = TigerGraphException(
    "Cannot parse json: <html> <head><title>Starting workspace</title></head> <body>"
    + "<div class='spinner'></div>" * 50
    + "</body></html>"
)


@pytest.mark.parametrize(
    "error, reason",
    [
        (STARTING, "starting workspace"),
        (
            http_error(502, b"<html><head><title>502 Bad Gateway</title></head><p>nginx</p>"),
            "bad gateway, HTTP 502",
        ),
        (http_error(504), "HTTP 504"),
        (
            http_error(503, {"error": True, "message": "not ready", "code": "REST-0005"}),
            "not ready, HTTP 503",
        ),
        (json.JSONDecodeError("x", "<html><body>resuming</body></html>", 0), "an HTML page"),
        (
            requests.ConnectionError(
                "HTTPSConnectionPool(host='tg.example', port=443): Max retries exceeded with "
                "url: /restpp/query/x (Caused by NewConnectionError('<urllib3.connection."
                "HTTPSConnection object at 0x10>: Failed to establish a new connection: "
                "[Errno 61] Connection refused'))"
            ),
            "connection refused",
        ),
        (
            requests.ReadTimeout(
                "HTTPSConnectionPool(host='tg.example', port=443): Read timed out. (read "
                "timeout=330)"
            ),
            "read timed out",
        ),
        # Capitalised words are lowercased, so the reason reads inside a line; an acronym
        # or a name keeps its capitals.
        (TigerGraphException("Query timeout exceeded", "REST-3002"), "query timeout exceeded"),
        (
            TigerGraphException("The graph engine is not ready yet, come back in a few minutes"),
            "the graph engine is not ready yet, come...",
        ),
        (
            TigerGraphException("GSQL Server is restarting on TigerGraph"),
            "GSQL server is restarting on TigerGraph",
        ),
    ],
)
def test_a_failure_is_named_in_a_few_words_never_by_its_page(
    error: BaseException, reason: str
) -> None:
    assert short_reason(error) == reason


def test_retries_and_the_outage_that_ends_them_name_the_reason_without_the_page(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tg = executor(FakeConn([STARTING] * 100), max_attempts=3, max_outage_s=120)
    with pytest.raises(TigerGraphUnavailableError) as raised:
        tg.run("q", {})
    message = str(raised.value)
    assert "<" not in message and len(message) < 200
    assert message.endswith("TigerGraphException: starting workspace")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(tg.sleeps) > 1
    for line in lines:
        assert line.startswith("TigerGraph is not answering yet (starting workspace): attempt ")


# TigerGraph's own words for a failure, longer than a retry's line keeps.
OUT_OF_MEMORY = TigerGraphException(
    "Runtime Error: the query fetch_subgraphs ran out of memory on partition 3"
)
URLLIB3_REFUSED = (
    "HTTPSConnectionPool(host='tg.example', port=443): Max retries exceeded with url: "
    "/restpp/query/x (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection "
    "object at 0x10>: Failed to establish a new connection: [Errno 61] Connection refused'))"
)


@pytest.mark.parametrize(
    "error, summary",
    [
        (
            OUT_OF_MEMORY,
            "TigerGraphException: Runtime Error: the query fetch_subgraphs ran out of memory "
            "on partition 3",
        ),
        (STARTING, "TigerGraphException: starting workspace"),
        (
            http_error(502, b"<html><head><title>502 Bad Gateway</title></head><p>nginx</p>"),
            "HTTPError: bad gateway, HTTP 502",
        ),
        (
            http_error(503, {"error": True, "message": "not ready", "code": "REST-0005"}),
            "HTTPError: not ready, HTTP 503",
        ),
        (
            http_error(504, b"upstream request timeout"),
            "HTTPError: upstream request timeout, HTTP 504",
        ),
        (requests.ConnectionError(URLLIB3_REFUSED), "ConnectionError: connection refused"),
        (
            requests.ReadTimeout(
                "HTTPSConnectionPool(host='tg.example', port=443): Read timed out. (read "
                "timeout=330)"
            ),
            "ReadTimeout: read timed out",
        ),
        # A message of TigerGraph's own is kept, even where it says what timed out.
        (
            TigerGraphException("The query timed out after 300 s on partition 3", "REST-3002"),
            "TigerGraphException: The query timed out after 300 s on partition 3",
        ),
        # About 200 characters of a longer message are kept.
        (
            TigerGraphException("word " * 60),
            "TigerGraphException: " + ("word " * 39) + "wo...",
        ),
    ],
)
def test_the_error_that_ends_the_retries_keeps_its_cause_but_never_a_page(
    error: BaseException, summary: str
) -> None:
    assert failure_summary(error) == summary
    assert "<" not in summary and len(summary) < 230


def test_retries_that_run_out_keep_the_cause_tigergraph_gave(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tg = executor(FakeConn([OUT_OF_MEMORY] * 2))
    with pytest.raises(TransientQueryError) as raised:
        tg.run(QUERY, {"node_ids": ["a"] * 512})
    # The retry's line names the reason in a few words; the error keeps TigerGraph's.
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith(
        "TigerGraph failed fetch_subgraphs, 512 keys (runtime error: the query fetch_subgraphs "
        "ran out of...): attempt 1, retrying in "
    )
    assert str(raised.value) == (
        "fetch_subgraphs (512 keys) failed after 2 attempts (suspected deterministic "
        "failure, retried once): TigerGraphException: Runtime Error: the query "
        "fetch_subgraphs ran out of memory on partition 3"
    )
    # A write has one attempt, so no retry records its error: the error itself names it.
    conn = FakeConn([requests.ConnectionError(URLLIB3_REFUSED)])
    with pytest.raises(TigerGraphUnavailableError) as raised:
        executor(conn).gsql("CREATE QUERY q() {}", what="CREATE QUERY q", attempts=1)
    assert str(raised.value) == (
        "CREATE QUERY q failed after 1 attempt (1 attempt allowed): ConnectionError: "
        "connection refused"
    )
    # A summary of that error, as a suite records the run it failed, keeps it whole.
    whole = f"TigerGraphUnavailableError: {raised.value}"
    assert error_summary(raised.value) == whole
    assert error_summary(ValueError("x" * 300)) == f"ValueError: {'x' * 200}"


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
            tg.run(QUERY, {"node_ids": ["a", "b"]})
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
    with pytest.raises(TigerGraphUnavailableError, match=r"unavailable for 120s"):
        tg.run("q", {})
    assert tg.clock.now - started == pytest.approx(120) and len(conn.calls) > 3
    assert sum(tg.sleeps) == pytest.approx(120)
    # max_outage_s = 0 fails on the first availability failure.
    conn = FakeConn([requests.ConnectionError("refused"), [{"status": "ok"}]])
    with pytest.raises(TigerGraphUnavailableError, match="after 1 attempt"):
        executor(conn, max_outage_s=0).run("q", {})


def test_slow_failing_attempts_count_toward_max_query_attempts() -> None:
    tg = executor(FakeConn([]), max_attempts=3, max_outage_s=3600)
    gateway = http_error(504, b"upstream request timeout")
    tg.client.conn.outcomes = [slow(tg.clock, 45, gateway) for _ in range(10)]
    with pytest.raises(TigerGraphUnavailableError, match="after 3 attempt.*max_query_attempts = 3"):
        tg.run("q", {})
    assert len(tg.client.conn.calls) == 3


def test_only_retries_that_end_unavailable_are_an_outage() -> None:
    # An outage: the graph did not answer, whatever the request.
    for error in (requests.ConnectionError("down"), http_error(503)):
        conn = FakeConn([error] * 3)
        with pytest.raises(TigerGraphUnavailableError, match="3 attempt"):
            executor(conn).run("q", {}, attempts=3)
    # The request's own transient failures are not: a timeout, a repeated bare 500.
    for error in (TigerGraphException("x", "REST-3002"), http_error(500)):
        conn = FakeConn([error] * 3)
        with pytest.raises(TransientQueryError) as raised:
            executor(conn).run("q", {})
        assert not isinstance(raised.value, TigerGraphUnavailableError), error
    # The failure that ends the retries decides: a bare 500, then a refused connection.
    conn = FakeConn([http_error(500), requests.ConnectionError("down")])
    with pytest.raises(TigerGraphUnavailableError):
        executor(conn).run("q", {}, attempts=2)


def test_permanent_errors_and_writes_are_not_retried() -> None:
    for error in (
        http_error(404),
        http_error(401),
        http_error(400, {"error": True, "message": "bad parameter"}),
        TigerGraphException("Query fetch_x is not installed", "REST-1000"),
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
        executor(conn).run("write_q", {}, attempts=1)
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


def test_resume_page_in_gsql_output_is_transient() -> None:
    conn = FakeConn(["<!DOCTYPE html><title>Starting workspace</title>", "CREATE QUERY q() {}"])
    tg = executor(conn)
    assert tg.gsql("SHOW QUERY q") == "CREATE QUERY q() {}"
    assert len(tg.sleeps) == 1
    assert failure_class(json.JSONDecodeError("x", "<html>", 0)) is not None
    assert failure_class(ValueError("Returned context differs")) is None


def test_executor_connects_only_with_the_settings_it_is_given() -> None:
    # Its caller reads .env (ConnectionSettings); the executor never does.
    with pytest.raises(ValueError, match="connected client or its settings"):
        TigerGraphExecutor()


def test_the_graph_is_the_one_the_connection_names() -> None:
    # No graph name is built in: GRAPHNAME in .env chooses it.
    for graph in ("Mule_Pattern_Learner", "Sensitivity_World"):
        assert executor(FakeConn([]), graph=graph).graph_name == graph


def test_a_count_of_one_in_a_failure_reads_in_the_singular() -> None:
    timeout = TigerGraphException("Query timeout exceeded", "REST-3002")
    with pytest.raises(ServerTimeoutError) as raised:
        executor(FakeConn([timeout] * 2)).run(QUERY, {"node_ids": ["a"]})
    assert str(raised.value).startswith(
        "fetch_subgraphs (1 key) failed after 2 attempts (server timeout, 1 retry allowed): "
    )
    with pytest.raises(ServerTimeoutError) as raised:
        executor(FakeConn([timeout])).run("q", {}, timeout_retries=0)
    assert str(raised.value).startswith("q failed after 1 attempt (server timeout, 0 retries")


# How TigerGraph words a refusal for the response's size, as far as is known.
TOO_LARGE = [
    TigerGraphException("The response size 70000000 exceeds the size limit 64000000"),
    http_error(500, {"error": True, "message": "Response size exceeds RESPONSE-LIMIT"}),
]


@pytest.mark.parametrize("error", TOO_LARGE)
def test_a_response_refused_for_its_size_raises_at_once(error: BaseException) -> None:
    conn = FakeConn([error, [{"status": "ok"}]])
    tg = executor(conn, size_limit=64_000_000)
    with pytest.raises(ResponseTooLarge, match="more than 64000000 bytes"):
        tg.run("read_payment_events", {"accounts": ["A1", "A2"]})
    assert len(conn.calls) == 1 and not tg.sleeps
    assert conn.calls[0][2]["sizeLimit"] == 64_000_000


def test_requests_are_sized_under_half_of_the_limit_the_executor_asks_for() -> None:
    conn = FakeConn([[{"status": "ok"}]])
    executor(conn).run("q", {})
    assert conn.calls[0][2]["sizeLimit"] == 2 * DEFAULT_CONFIG.transport.response_budget_bytes


def _ok(name: str, params: dict[str, Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"status": "ok", "query": name}]


# Names no entry point but parity may run: the loader's verification and the owner's demos.
ELSEWHERE = ("verify_load", "demo_account_web")


@pytest.mark.parametrize(
    ("entry", "allowed"),
    [
        (FetchExecutor, FETCH_ALLOWLIST),
        (EvaluationExecutor, EVALUATION_ONLY),
        (BenchExecutor, FETCH_ALLOWLIST | BENCH_ONLY),
        (ParityExecutor, frozenset(QUERY_FILES)),
    ],
)
def test_each_entry_point_runs_only_the_queries_of_its_role(
    entry: Callable[[TigerGraphExecutor], AllowedQueries], allowed: frozenset[str]
) -> None:
    conn = FakeConn([_ok] * 20)
    runner = entry(executor(conn))
    assert runner.allowed == allowed
    refused = [*ELSEWHERE, *sorted(set(QUERY_FILES) - allowed)]
    for name in refused:
        with pytest.raises(QueryNotAllowed, match=f"does not run {name}"):
            runner.run(name, {})
    # Refused before any request.
    assert conn.calls == []
    for name in sorted(allowed):
        assert runner.run(name, {"t_ms": 1}) == [{"status": "ok", "query": name}]
    assert [call[0] for call in conn.calls] == sorted(allowed)
    # It offers no GSQL statement and no connection operation.
    assert not hasattr(runner, "gsql") and not hasattr(runner, "call")


def test_fetch_refuses_the_answer_key_and_the_loaders_verification() -> None:
    conn = FakeConn([])
    fetch = FetchExecutor(executor(conn))
    for name in ("read_answer_key", "verify_load", "profile_hubs"):
        with pytest.raises(QueryNotAllowed):
            fetch.run(name, {})
    assert conn.calls == []


def _accounts(chunk: tuple[str, ...]) -> dict[str, Any]:
    return {"t_ms": 1, "accounts": list(chunk)}


def _refusing(error: BaseException, most: int) -> Any:
    """An answer refusing, with error, any request of more than most accounts."""

    def answer(name: str, params: dict[str, Any], kwargs: dict[str, Any]) -> Any:
        if len(params["accounts"]) > most:
            raise error
        return [{"status": "ok", "accounts": params["accounts"]}]

    return answer


@pytest.mark.parametrize(
    "error",
    [*TOO_LARGE, TigerGraphException("Query timeout exceeded", "REST-3002")],
)
def test_a_refused_or_timed_out_request_is_halved_down_to_one_account(
    error: BaseException,
) -> None:
    conn = FakeConn([_refusing(error, 2)] * 20)
    tg = executor(conn)
    answers = run_in_halves(FetchExecutor(tg), "read_payment_events", "abcde", _accounts)
    # 5 is refused, then 3 of its halves; 2, 1 and 2 accounts are answered.
    assert [chunk for chunk, _ in answers] == [("a", "b"), ("c",), ("d", "e")]
    assert [rows[0]["accounts"] for _, rows in answers] == [["a", "b"], ["c"], ["d", "e"]]
    assert [len(call[1]["accounts"]) for call in conn.calls] == [5, 3, 2, 1, 2]
    # A timeout is not repeated before the request is halved.
    assert not tg.sleeps


def test_one_account_still_refused_is_an_error_never_a_gap() -> None:
    conn = FakeConn([_refusing(TOO_LARGE[0], 0)] * 20)
    with pytest.raises(ResponseTooLarge) as raised:
        run_in_halves(FetchExecutor(executor(conn)), "read_payment_events", "ab", _accounts)
    assert raised.value.__notes__ == ["read_payment_events failed for the account a alone"]
    assert [call[1]["accounts"] for call in conn.calls] == [["a", "b"], ["a"]]
    with pytest.raises(ValueError, match="needs an account"):
        run_in_halves(FetchExecutor(executor(conn)), "read_payment_events", [], _accounts)
    # Any other failure is not halved.
    conn = FakeConn([http_error(400)] * 2)
    with pytest.raises(requests.HTTPError):
        run_in_halves(executor(conn), "read_payment_events", "ab", _accounts)
    assert len(conn.calls) == 1
