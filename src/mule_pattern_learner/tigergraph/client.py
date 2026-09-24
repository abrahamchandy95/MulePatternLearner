from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
import threading
from typing import Any, cast, override

import requests
from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.tigergraph.settings import Settings

_READ_TIMEOUT_S = 600.0
_CONNECT_TIMEOUT_S = 30.0
# Hard wall-clock ceiling for a single installed-query call. Distinct from the
# socket/session read timeout: pyTigerGraph's runInstalledQuery can stall in a
# way the socket timeout does not interrupt, so this deadline is enforced from
# the OUTSIDE via a worker thread. Set above the slowest legitimate batch query
# but well under "hung forever". A batch that exceeds this is treated as a
# transient failure and retried by the training loop.
_QUERY_TIMEOUT_S = 300.0


class ClientQueryTimeoutError(requests.exceptions.ReadTimeout):
    """A single installed-query call exceeded _QUERY_TIMEOUT_S.

    Subclasses requests.exceptions.ReadTimeout so the training loop's existing
    transient-error handling (_resilient_batches) catches and retries it without
    needing to know about this client-specific type.
    """


def _status_error(response: requests.Response) -> requests.HTTPError | None:
    """HTTPError for server-side and throttling statuses, None for every other status.

    pyTigerGraph parses a JSON error body of any non-2xx response into a
    TigerGraphException before it calls raise_for_status, and that exception
    has no HTTP status. A JSON-bodied 503 or 429 would then look like a
    permanent query error. Only 5xx, 408 and 429 are raised here: pyTigerGraph
    itself needs 401 (token refresh), 404 (endpoint fallbacks) and the other 4xx.
    """
    status = response.status_code
    if status < 500 and status not in (408, 429):
        return None
    body = (response.content or b"")[:600].decode("utf-8", "replace")
    return requests.HTTPError(f"HTTP {status}: {' '.join(body.split())[:300]}", response=response)


class _TimeoutConnection(TigerGraphConnection):
    """TigerGraphConnection whose HTTP requests always carry a finite timeout.

    pyTigerGraph passes `timeout=None` for every request without a GSQL-TIMEOUT
    header (vertex counts, SHOW QUERY, endpoint listings, installation polling),
    so a stalled socket could block forever. Sessions are thread-local in
    pyTigerGraph, so the default is installed on each thread's session on first
    use. `Client.request_timeout` overrides it for the calling thread only.
    Responses with status 5xx, 408 or 429 raise requests.HTTPError before
    pyTigerGraph sees them, so callers can classify them by status.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Set before the base initializer, which may already issue requests.
        self.timeout_override = threading.local()
        super().__init__(*args, **kwargs)

    def _default_timeout(self) -> tuple[float, float]:
        return cast(
            "tuple[float, float]",
            getattr(self.timeout_override, "value", (_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S)),
        )

    @property
    @override
    def _session(self) -> requests.Session:
        session = super()._session
        if not getattr(session, "_has_default_timeout", False):
            request = session.request

            def request_with_timeout(method: str, url: str, **kwargs: Any) -> requests.Response:
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = self._default_timeout()
                response = request(method, url, **kwargs)
                error = _status_error(response)
                if error is not None:
                    raise error
                return response

            setattr(session, "request", request_with_timeout)
            setattr(session, "_has_default_timeout", True)
        return session


class Client:
    """
    Client that connects to TigerGraph
    """

    _settings: Settings
    conn: TigerGraphConnection
    _executor: ThreadPoolExecutor

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.conn = _TimeoutConnection(
            host=settings.host,
            graphname=settings.graphname,
            gsqlSecret=settings.secret.get_secret_value(),
        )
        _ = self.conn.getToken(settings.secret.get_secret_value())
        # Single reusable worker thread for deadline-enforced query calls. A
        # query that overruns the deadline is abandoned here (the thread keeps
        # running until the underlying read finally returns or errors, but the
        # caller does not block on it); a fresh thread is used for the next call.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tg-query")

    @contextmanager
    def request_timeout(
        self, read_s: float, connect_s: float = _CONNECT_TIMEOUT_S
    ) -> Generator[None, None, None]:
        """Temporarily change the default HTTP timeout for the calling thread."""
        local = cast(_TimeoutConnection, self.conn).timeout_override
        previous = getattr(local, "value", None)
        local.value = (connect_s, read_s)
        try:
            yield
        finally:
            if previous is None:
                del local.value
            else:
                local.value = previous

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
