"""Installed-query access: failure classification, the retrying executor and response helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
import json
import random
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from ..config import DEFAULT_CONFIG
from ..contract.bounds import OUTAGE_SECONDS, QUERY_ATTEMPTS
from ..contract.server import GRAPH_NAME
from ..runtime.console import brief, plural
from ..runtime.progress import emit

if TYPE_CHECKING:
    from pyTigerGraph import TigerGraphConnection

    from .connection import ConnectionSettings

T = TypeVar("T")


class TransientQueryError(RuntimeError):
    """A transport failure that is expected to clear (restart, overload, resume)."""


class ServerTimeoutError(TransientQueryError):
    """TigerGraph exceeded its query timeout on every allowed attempt of one request."""


class TigerGraphUnavailableError(TransientQueryError):
    """TigerGraph stayed unavailable until an operation's retries ran out: an outage.

    The retries ended on an availability failure (see failure_class), because the outage
    budget or the attempts allowed ran out. Nothing about the request itself failed, so a
    suite of runs stops on it rather than failing one run after another, while the other
    transient errors (a server timeout, a repeated suspected-deterministic failure)
    belong to the request that raised them.
    """


# Failure classes. Availability failures clear when the server (or the path to it)
# recovers and are retried under a wall-clock budget. Suspected-deterministic failures
# would fail the same way again, so they get at most one retry; server timeouts are a
# separate class because a multi-key context request that times out is bisected.
AVAILABILITY = "availability"
SERVER_TIMEOUT = "server_timeout"
DETERMINISTIC = "deterministic"
_TIMEOUT_CODES = frozenset({"REST-3002"})  # query exceeded the server timeout
_AVAILABILITY_MESSAGE = re.compile(
    r"starting workspace|workspace is (?:starting|resuming|paused)"
    r"|service (?:is )?(?:un|not )available|temporarily unavailable|try again"
    r"|(?:engine|service|server|graph) (?:is )?not ready|bad gateway|gateway time"
    r"|too many requests|rate limit|server is busy|queue is full"
    r"|memory usage.*critical|critical.*memory|connection error"
    r"|connection (?:reset|refused|aborted)|remote end closed",
    re.IGNORECASE,
)
_TIMEOUT_MESSAGE = re.compile(r"timed? ?out|timeout", re.IGNORECASE)
_DETERMINISTIC_MESSAGE = re.compile(r"cannot parse json|out of memory", re.IGNORECASE)


def looks_like_html(text: str) -> bool:
    head = text[:2048].lstrip().lower()
    return head.startswith(("<!doctype html", "<html")) or "starting workspace" in head


def _message_class(message: str) -> str | None:
    if _AVAILABILITY_MESSAGE.search(message) or (
        "cannot parse json" in message.lower() and "<html" in message.lower()
    ):
        return AVAILABILITY
    if _TIMEOUT_MESSAGE.search(message):
        return SERVER_TIMEOUT
    if _DETERMINISTIC_MESSAGE.search(message):
        return DETERMINISTIC
    return None


def _http_class(error: Any) -> str | None:
    response = error.response
    if response is None:
        return AVAILABILITY
    status = int(response.status_code)
    body = (response.content or b"")[:4096].decode("utf-8", "replace")
    code, message = "", body
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        code = str(parsed.get("code") or "")
        message = str(parsed.get("message") or "")
    if code in _TIMEOUT_CODES:
        return SERVER_TIMEOUT
    if status != 500:
        return AVAILABILITY if status > 500 or status in (408, 429) else None
    if looks_like_html(body):
        return AVAILABILITY  # a gateway error page, not a query error
    return _message_class(message) or DETERMINISTIC


def failure_class(error: BaseException) -> str | None:
    """AVAILABILITY, SERVER_TIMEOUT, DETERMINISTIC, or None for a permanent error.

    - Availability: connection errors and connect timeouts, HTTP 502/503/504,
      408 and 429 (and every other 5xx except 500), HTML error pages, the
      TigerGraph Cloud "Starting workspace" page, chunked-encoding errors and
      overload or not-ready messages.
    - Server timeout: code REST-3002, a timeout message, and a client read
      timeout (pyTigerGraph sets it 30 s beyond the server-side query timeout).
    - Suspected deterministic: a bare HTTP 500, a response that is neither JSON
      nor an HTML page, and query out-of-memory.
    Contract and validation errors (ValueError and the like) are permanent.
    """
    from pyTigerGraph.common.exception import TigerGraphException
    import requests

    if isinstance(error, ServerTimeoutError):
        return SERVER_TIMEOUT
    if isinstance(error, TransientQueryError):
        return AVAILABILITY
    if isinstance(error, json.JSONDecodeError):
        return AVAILABILITY if looks_like_html(error.doc) else DETERMINISTIC
    if isinstance(error, requests.HTTPError):
        return _http_class(error)
    if isinstance(error, requests.ConnectionError | requests.exceptions.ChunkedEncodingError):
        return AVAILABILITY
    if isinstance(error, requests.Timeout):
        return SERVER_TIMEOUT
    if isinstance(error, TigerGraphException):
        if str(getattr(error, "code", "") or "") in _TIMEOUT_CODES:
            return SERVER_TIMEOUT
        return _message_class(str(getattr(error, "message", "") or ""))
    return None


def error_summary(error: BaseException) -> str:
    """The error's type and the first 200 characters of its message, on one line.

    A TransientQueryError keeps its whole message: the one the executor raises when its
    retries run out names the operation, its attempts and why they ended, then the
    error that ended them (failure_summary), itself cut to about 200 characters.
    """
    message = " ".join(str(getattr(error, "message", None) or error).split())
    kept = message if isinstance(error, TransientQueryError) else message[:200]
    return f"{type(error).__name__}: {kept}"


_PAGE_HEADING = re.compile(r"<(title|h1)[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_STATUS_HEADING = re.compile(r"(\d{3})\s+(.+)")
# What a connection error says in the middle of the urllib3 text around it.
_CONNECTION_FAILURE = re.compile(
    r"connection (?:refused|reset|aborted)|remote end closed connection"
    r"|name or service not known|nodename nor servname provided"
    r"|temporary failure in name resolution|read timed out|timed out",
    re.IGNORECASE,
)


def _failure_text(error: BaseException) -> tuple[str, Any]:
    """What a failed request said, and the HTTP status of its response if it had one.

    The text is the document JSON could not parse, else the response's body, else the
    error's message.
    """
    response = getattr(error, "response", None)
    if isinstance(error, json.JSONDecodeError):
        text = error.doc
    elif response is not None and getattr(response, "content", None):
        text = bytes(response.content[:4096]).decode("utf-8", "replace")
    else:
        text = str(getattr(error, "message", None) or error)
    return text, getattr(response, "status_code", None)


def _page_reason(text: str) -> tuple[str, int | None] | None:
    """What an HTML page says, and the status its heading gives; None for other text.

    A page gives its title or first heading, lowercased ("502 Bad Gateway" gives "bad
    gateway" and 502), or "an HTML page" when it has neither.
    """
    heading = _PAGE_HEADING.search(text)
    if heading is not None:
        reason = " ".join(heading.group(2).split())
        coded = _STATUS_HEADING.fullmatch(reason)
        if coded is not None:
            return coded.group(2).lower(), int(coded.group(1))
        return reason.lower(), None
    if looks_like_html(text) or "<html" in text.lower():
        return "an HTML page", None
    return None


def _message(text: str) -> str:
    """A JSON body's message, or the text, on one line without the URL requests adds."""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("message"):
        text = str(parsed["message"])
    # requests words an HTTPError "502 Server Error: Bad Gateway for url: ...".
    text = re.sub(r"\s+for url:.*", "", " ".join(text.split()))
    return re.sub(r"^\d{3} (?:Client|Server) Error: ", "", text)


def _with_status(reason: str, status: Any, error: BaseException) -> str:
    """A reason with the HTTP status beside it, or the status alone when it says nothing more."""
    if status is not None and reason in ("", str(status)):
        return f"HTTP {status}"
    reason = reason or type(error).__name__
    return reason if status is None else f"{reason}, HTTP {status}"


def _lowercased(text: str) -> str:
    """Text with its capitalised words in lowercase; GSQL, REST-3002 or TigerGraph stay."""
    return " ".join(
        word.lower() if word[1:] == word[1:].lower() else word for word in text.split(" ")
    )


def short_reason(error: BaseException, words: int = 8) -> str:
    """Why a request failed, in a few lowercase words for a person: never a page or a URL.

    An HTML page gives its title or first heading, as a TigerGraph Cloud workspace that
    is starting answers ("starting workspace"); a JSON body its message; a connection
    error what failed; anything else its first ``words`` words. Capitalised words are
    lowercased, so the reason reads inside a line; acronyms and names keep their
    capitals. A known HTTP status is added, from the response or from a page heading
    such as "502 Bad Gateway".
    """
    text, status = _failure_text(error)
    page = _page_reason(text)
    if page is not None:
        reason, coded = page
        status = status or coded
    else:
        text = _message(text)
        failure = _CONNECTION_FAILURE.search(text)
        if failure is not None:
            reason = failure.group(0).lower()
        else:
            kept = text.split()[:words]
            reason = _lowercased(" ".join(kept)) + ("..." if len(text.split()) > words else "")
    return _with_status(reason, status, error)


def failure_summary(error: BaseException) -> str:
    """The error's type and why it failed, in about 200 characters on one line.

    It ends the error the executor raises once its retries run out, so the stderr line
    of the command it stops and the record of a suite's run it fails keep the cause
    TigerGraph gave, such as "out of memory": a JSON body's message, the response's body
    or the error's message, cut to about 200 characters, with the HTTP status. An HTML
    page is its short reason (its title or heading) and a connection error what failed
    (short_reason), never the page's markup or urllib3's text around the failure.
    """
    from requests import RequestException

    text, status = _failure_text(error)
    transport = isinstance(error, RequestException) and status is None
    if _page_reason(text) is not None or (transport and _CONNECTION_FAILURE.search(text)):
        return f"{type(error).__name__}: {short_reason(error)}"
    return f"{type(error).__name__}: {_with_status(brief(_message(text), 200), status, error)}"


class QueryExecutor(Protocol):
    """Installed-query access with the keywords of TigerGraphExecutor.run.

    Callers pass timeouts, write attempts and timeout retries as keywords; a test
    double may ignore them.
    """

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]: ...


class ConnectionExecutor(QueryExecutor, Protocol):
    """A QueryExecutor that also runs operations on its connection and GSQL statements.

    The scope, provenance and installer functions use these; TigerGraphExecutor
    retries both under its budgets, and an operation that writes passes attempts=1.
    ``graph_name`` is the graph the connection uses.
    """

    client: Any
    graph_name: str

    def call(
        self, operation: Callable[[Any], T], *, what: str, attempts: int | None = None
    ) -> T: ...
    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str: ...


class TigerGraphExecutor:
    """Installed-query access with per-class retry budgets (see failure_class).

    It runs on a connected `client`, or connects with `settings` (read from the
    repository .env by pipeline.connect); credentials never enter cache metadata.
    - Availability failures are retried with capped, jittered exponential backoff
      until `max_outage_s` seconds have passed since the operation's first such
      failure, and then raise TigerGraphUnavailableError. Worker threads share one
      "backoff until" time, so they pause together instead of hammering a resuming
      workspace.
    - Server timeouts are retried `timeout_retries` times (default once) and then
      raise ServerTimeoutError; other suspected-deterministic failures are
      retried once.
    - `max_attempts` (transport.max_query_attempts) caps the attempts that count:
      every attempt except an availability failure that failed within
      `slow_attempt_s`, so a fast-failing outage is bounded by the wall clock
      and a request that hangs on every attempt by the attempt cap. An explicit
      `attempts` argument is an absolute cap on all attempts (1 for writes).
    Anything else, including every error raised while validating a response,
    propagates on the first attempt. The connection keeps one HTTP session per
    thread, so `run` is thread-safe.
    """

    def __init__(
        self,
        *,
        max_attempts: int = DEFAULT_CONFIG.transport.max_query_attempts,
        max_outage_s: float = DEFAULT_CONFIG.transport.max_outage_s,
        timeout_s: float = 300.0,
        base_delay_s: float = 4.0,
        max_delay_s: float = 60.0,
        slow_attempt_s: float = 30.0,
        size_limit: int = 64_000_000,
        client: Any = None,
        settings: ConnectionSettings | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        if (
            not QUERY_ATTEMPTS.holds(max_attempts)
            or max_outage_s < OUTAGE_SECONDS.low
            or timeout_s <= 0
            or not 0 <= base_delay_s <= max_delay_s
            or slow_attempt_s < 0
        ):
            raise ValueError("Invalid retry policy")
        self.max_attempts, self.max_outage_s = max_attempts, float(max_outage_s)
        self.timeout_s, self.slow_attempt_s = timeout_s, slow_attempt_s
        self.base_delay_s, self.max_delay_s = base_delay_s, max_delay_s
        self.size_limit = size_limit
        self._sleep, self._clock = sleep, clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._backoff_until = 0.0
        self.calls = 0
        self.retries: Counter[str] = Counter()
        if client is None:
            if settings is None:
                raise ValueError("TigerGraphExecutor needs a connected client or its settings")
            from .client import Client

            client = self._retry(lambda: Client(settings), what="connect", attempts=None)
        self.client = client
        if self.client.graphname != GRAPH_NAME:
            raise ValueError(f"Training queries require {GRAPH_NAME}")
        self.graph_name: str = GRAPH_NAME

    def delay(self, attempt: int) -> float:
        """Exponential backoff with equal jitter, capped at max_delay_s."""
        base = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        with self._lock:
            jitter = self._rng.random()
        return base * (0.5 + 0.5 * jitter)

    def _shared_pause(self, now: float, pause: float) -> float:
        """Extend the shared backoff to now + pause; return this thread's wait."""
        with self._lock:
            self._backoff_until = max(self._backoff_until, now + pause)
            return self._backoff_until - now

    def _retry(
        self,
        operation: Callable[[], T],
        *,
        what: str,
        attempts: int | None,
        timeout_retries: int = 1,
        detail: str = "",
    ) -> T:
        with self._lock:
            waiting = self._backoff_until - self._clock()
        if waiting > 0:  # another thread saw the server unavailable
            self._sleep(waiting)
        label = f"{what} ({detail})" if detail else what
        total = counted = timeouts = deterministic = outages = 0
        outage_started: float | None = None
        while True:
            total += 1
            started = self._clock()
            try:
                result = operation()
            except Exception as error:
                kind = failure_class(error)
                if kind is None:
                    raise
                now = self._clock()
                if kind != AVAILABILITY or now - started >= self.slow_attempt_s:
                    counted += 1
                if kind == SERVER_TIMEOUT:
                    timeouts += 1
                    exhausted = timeouts > timeout_retries
                    allowed = plural(timeout_retries, "retry", "retries")
                    reason = f"server timeout, {allowed} allowed"
                elif kind == DETERMINISTIC:
                    deterministic += 1
                    exhausted = deterministic > 1
                    reason = "suspected deterministic failure, retried once"
                else:
                    outage_started = started if outage_started is None else outage_started
                    exhausted = now - outage_started >= self.max_outage_s
                    reason = (
                        f"TigerGraph unavailable for {now - outage_started:.0f}s "
                        f"(max_outage_s = {self.max_outage_s:.0f})"
                    )
                if attempts is not None and total >= attempts:
                    exhausted, reason = True, f"{plural(attempts, 'attempt')} allowed"
                elif attempts is None and counted >= self.max_attempts:
                    exhausted, reason = True, f"max_query_attempts = {self.max_attempts}"
                if exhausted:
                    failure = {
                        SERVER_TIMEOUT: ServerTimeoutError,
                        AVAILABILITY: TigerGraphUnavailableError,
                    }.get(kind, TransientQueryError)
                    raise failure(
                        f"{label} failed after {plural(total, 'attempt')} ({reason}): "
                        f"{failure_summary(error)}"
                    ) from error
                if kind == AVAILABILITY:
                    outages += 1
                    assert outage_started is not None
                    budget = outage_started + self.max_outage_s - now
                    pause = self._shared_pause(now, min(self.delay(outages), max(budget, 0.0)))
                else:
                    pause = self.delay(1)
                with self._lock:
                    self.retries[what] += 1
                emit(
                    {
                        "event": "retry",
                        "failure": kind,
                        "operation": label,
                        "attempt": total,
                        "retry_in_s": round(pause, 1),
                        "error": error_summary(error),
                        "reason": short_reason(error),
                    }
                )
                self._sleep(pause)
            else:
                with self._lock:
                    self.calls += 1
                    self._backoff_until = 0.0
                return result

    def call(
        self,
        operation: Callable[[TigerGraphConnection], T],
        *,
        what: str,
        attempts: int | None = None,
        timeout_retries: int = 1,
        detail: str = "",
    ) -> T:
        """Run a connection operation under the retry policy; attempts=1 for writes."""
        return self._retry(
            lambda: operation(self.client.conn),
            what=what,
            attempts=attempts,
            timeout_retries=timeout_retries,
            detail=detail,
        )

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        """Run an installed query. Pass attempts=1 for queries that write.

        timeout_retries=0 raises ServerTimeoutError on the first server timeout,
        for callers that split the request instead of repeating it.
        """
        timeout_ms = int(1000 * (self.timeout_s if timeout_s is None else timeout_s))

        def operation(conn: TigerGraphConnection) -> list[dict[str, Any]]:
            result: Any = conn.runInstalledQuery(
                name, params, usePost=True, timeout=timeout_ms, sizeLimit=self.size_limit
            )
            if not isinstance(result, list):
                raise ValueError(f"{name} returned {type(result).__name__}, not a result list")
            return result

        # Parameter sizes name the request in retry events (context requests: key count).
        sizes = {key: len(value) for key, value in params.items() if isinstance(value, list)}
        detail = (
            plural(sizes["node_ids"], "key")
            if "node_ids" in sizes
            else ", ".join(f"{key}={size}" for key, size in sizes.items())
        )
        return self.call(
            operation,
            what=name,
            attempts=attempts,
            timeout_retries=timeout_retries,
            detail=detail,
        )

    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str:
        """A GSQL statement with resume detection: SHOW QUERY, or a write with attempts=1."""

        def operation(conn: TigerGraphConnection) -> str:
            output = str(conn.gsql(text))
            if looks_like_html(output):
                raise TransientQueryError("TigerGraph returned an HTML page instead of GSQL output")
            return output

        return self.call(operation, what=what, attempts=attempts)


# A named tuple of exception types: formatters targeting Python 3.14 rewrite
# `except (A, B):` as `except A, B:`, which Python 3.12 and 3.13 cannot parse.
CONVERSION_ERRORS = (TypeError, ValueError)


def checked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "status" in row and row["status"] != "ok":
            raise ValueError(f"TigerGraph rejected request: {row}")
    if not any(row.get("status") == "ok" for row in rows):
        raise ValueError("Query did not return a success status")
    return rows


def merged_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A query's PRINT rows as one dict (a later row wins on a repeated field)."""
    merged: dict[str, Any] = {}
    for row in rows:
        merged.update(row)
    return merged


def printed(rows: list[dict[str, Any]], name: str) -> Any:
    """The first PRINT row's value of field name; a missing field is a malformed response."""
    for row in rows:
        if name in row:
            return row[name]
    raise ValueError(f"{name} missing from response")


def account_pages(
    executor: QueryExecutor,
    query: str,
    params: dict[str, Any],
    *,
    page_size: int = 10_000,
    timeout_s: float | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Pages of an after_id/batch_size account query, each row a flat attribute dict.

    A page larger than page_size, or an account_id that does not strictly increase
    within and across pages, is a transport contract violation. Paging stops after
    an empty or short page.
    """
    after = ""
    while True:
        rows = checked_rows(
            executor.run(
                query, {**params, "after_id": after, "batch_size": page_size}, timeout_s=timeout_s
            )
        )
        page = [dict(item.get("attributes", item)) for item in printed(rows, "accounts")]
        if len(page) > page_size:
            raise ValueError(f"{query} page exceeds the transport contract")
        if not page:
            return
        for row in page:
            account = str(row["account_id"])
            if account <= after:
                raise ValueError(f"{query} pagination is not strictly increasing")
            after = account
        yield page
        if len(page) < page_size:
            return
