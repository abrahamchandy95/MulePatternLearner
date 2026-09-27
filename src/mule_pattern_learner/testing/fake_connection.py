"""Fake pyTigerGraph connections for the client, executor and transport tests.

RecordingExecutor runs the real retry policy over a scripted connection and a fake
clock, so retries, backoff and budgets are tested without a network.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor


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
    """A connected client of the training graph over a scripted connection.

    ``timeouts`` records the read timeouts set with ``request_timeout``.
    """

    graphname = "Mule_Pattern_Learner"

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.timeouts: list[float] = []

    @contextmanager
    def request_timeout(self, read_s: float, connect_s: float = 30.0) -> Generator[None]:
        self.timeouts.append(read_s)
        yield


class RecordingExecutor(TigerGraphExecutor):
    """The real retry policy over a fake connection and a fake clock; sleeps are recorded."""

    def __init__(self, conn: Any, **kwargs: Any) -> None:
        self.clock = FakeClock()
        self.sleeps = self.clock.sleeps
        client = FakeClient(conn)
        super().__init__(client=client, sleep=self.clock.sleep, clock=self.clock.time, **kwargs)


def executor(conn: Any, **kwargs: Any) -> RecordingExecutor:
    return RecordingExecutor(conn, **kwargs)
