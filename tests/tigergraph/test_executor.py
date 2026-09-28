"""The query executor: failure classes, retry budgets and the shared backoff."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

import pytest
from pyTigerGraph.common.exception import TigerGraphException
import requests

from mule_pattern_learner.contract.server import CONTEXT_QUERY, CREATE_SCOPE_QUERY, CUTOFF_QUERY
from mule_pattern_learner.testing.fake_connection import FakeClock, FakeConn, executor
from mule_pattern_learner.tigergraph.executor import (
    AVAILABILITY,
    DETERMINISTIC,
    SERVER_TIMEOUT,
    ServerTimeoutError,
    TigerGraphExecutor,
    TigerGraphUnavailableError,
    TransientQueryError,
    failure_class,
)


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
    assert tg.run(CUTOFF_QUERY, {"cutoff_times": [1]}) == [{"status": "ok"}]
    assert len(conn.calls) == 6
    assert tg.retries[CUTOFF_QUERY] == 5 and tg.calls == 1
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
    assert (failure_class(error) is not None) is (kind is not None)


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
            tg.run(CONTEXT_QUERY, {"node_ids": ["a", "b"]})
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
        executor(conn).run(CREATE_SCOPE_QUERY, {}, attempts=1)
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
    # The pipeline reads .env (pipeline.connect.connect); the executor never does.
    with pytest.raises(ValueError, match="connected client or its settings"):
        TigerGraphExecutor()
