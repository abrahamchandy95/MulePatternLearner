"""The pyTigerGraph client: status errors and default timeouts."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from typing import Any

import pytest
from pyTigerGraph.common.exception import TigerGraphException
import requests

from mule_pattern_learner.contract.server import GRAPH_NAME
from mule_pattern_learner.testing.fake_connection import executor
from mule_pattern_learner.tigergraph.client import Client, _status_error, _TimeoutConnection
from mule_pattern_learner.tigergraph.executor import ServerTimeoutError


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
    conn = _TimeoutConnection(host="http://127.0.0.1", graphname=GRAPH_NAME)
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


def test_connection_default_timeout_is_effective_and_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def fake(self: requests.Session, method: str, url: str, **kwargs: Any) -> None:
        seen.append(kwargs.get("timeout"))
        raise requests.ConnectionError("no network in tests")

    monkeypatch.setattr(requests.Session, "request", fake)
    conn = _TimeoutConnection(host="http://127.0.0.1", graphname=GRAPH_NAME)
    client = Client.__new__(Client)
    client.conn = conn
    for timeout in (None, 7):
        with pytest.raises(requests.ConnectionError):
            conn._session.request("GET", "http://127.0.0.1:9/x", timeout=timeout)
    with client.request_timeout(read_s=3600):
        with pytest.raises(requests.ConnectionError):
            conn._session.request("GET", "http://127.0.0.1:9/x", timeout=None)
    assert seen == [(30.0, 600.0), 7, (30.0, 3600)]
