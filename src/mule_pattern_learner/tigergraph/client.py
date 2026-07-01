from functools import partial
import socket
from typing import cast, override

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import requests
from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.tigergraph.settings import Settings

_READ_TIMEOUT_S = 600.0
_CONNECT_TIMEOUT_S = 30.0
# Hard wall-clock ceiling for a single installed-query call. Distinct from the
# socket/session read timeout: pyTigerGraph's runInstalledQuery can stall in a
# way the socket timeout does not interrupt, so this deadline is enforced from
# the OUTSIDE via a worker thread. Set above the slowest legitimate batch query
# but well under "hung forever" -- a batch that exceeds this is treated as a
# transient failure and retried by the training loop.
_QUERY_TIMEOUT_S = 300.0


class ClientQueryTimeoutError(requests.exceptions.ReadTimeout):
    """A single installed-query call exceeded _QUERY_TIMEOUT_S.

    Subclasses requests.exceptions.ReadTimeout so the training loop's existing
    transient-error handling (_resilient_batches) catches and retries it without
    needing to know about this client-specific type.
    """


class Client:
    """
    Client that connects to TigerGraph
    """

    _settings: Settings
    conn: TigerGraphConnection
    _executor: ThreadPoolExecutor

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        socket.setdefaulttimeout(_READ_TIMEOUT_S)
        self.conn = TigerGraphConnection(
            host=settings.host,
            graphname=settings.graphname,
            gsqlSecret=settings.secret.get_secret_value(),
        )
        _ = self.conn.getToken(settings.secret.get_secret_value())
        self._install_default_timeout()
        # Single reusable worker thread for deadline-enforced query calls. A
        # query that overruns the deadline is abandoned here (the thread keeps
        # running until the underlying read finally returns or errors, but the
        # caller does not block on it); a fresh thread is used for the next call.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tg-query")

    def _install_default_timeout(self) -> None:
        session = cast(object, getattr(self.conn, "_session", None))
        if not isinstance(session, requests.Session):
            return
        if getattr(session.request, "_has_default_timeout", False):
            return
        wrapped = partial(session.request, timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S))
        setattr(wrapped, "_has_default_timeout", True)
        setattr(session, "request", wrapped)

    def run_with_timeout(
        self, query_name: str, params: dict[str, object], timeout_s: float = _QUERY_TIMEOUT_S
    ) -> list[object]:
        """Run an installed query under a hard wall-clock deadline.

        Submits conn.runInstalledQuery to a worker thread and waits at most
        timeout_s for it. On overrun, raises ClientQueryTimeoutError (a
        ReadTimeout subclass) so callers' transient-retry logic handles it; the
        abandoned thread is left to unwind on its own and a fresh executor is
        created so the next call is not blocked behind it. This enforces a
        timeout even when pyTigerGraph's own HTTP path ignores the socket /
        session read timeout.
        """
        future = self._executor.submit(self.conn.runInstalledQuery, query_name, params)
        try:
            return cast("list[object]", future.result(timeout=timeout_s))
        except FuturesTimeoutError as exc:
            # Abandon the stuck call: drop the executor (without waiting on the
            # running thread) and spin up a fresh one so the retried query is
            # not queued behind the hung thread.
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tg-query")
            raise ClientQueryTimeoutError(
                f"installed query {query_name!r} exceeded {timeout_s:.0f}s deadline"
            ) from exc

    @property
    def graphname(self) -> str:
        return self._settings.graphname

    @override
    def __repr__(self) -> str:
        return f"Client(graphname={self._settings.graphname!r})"
