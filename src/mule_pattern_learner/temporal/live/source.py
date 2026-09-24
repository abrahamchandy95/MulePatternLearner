"""Read-only query access with retries, bounded streaming and optional durable caching."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import asdict
import json
import logging
from pathlib import Path
import queue
import random
import re
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
import weakref
import zlib

import numpy as np

from ..encoding import BASIS_ID, fourier64
from .config_schema import TRANSPORT_DEFAULTS
from .contract import (
    AMOUNT_RATIO_CAP,
    AMOUNT_RATIO_FEATURES,
    CHANNELS,
    CLIENT_GROUPS,
    CONTRACT_VERSION,
    FEATURE_GROUPS,
    LEGACY_GROUPS,
    NODE_TYPES,
    RAILS,
    RELATIONS,
    STRATA,
    ContextKey,
    FeaturePlan,
    SamplerPlan,
    contract_fingerprint,
    fingerprint,
)

if TYPE_CHECKING:
    from pyTigerGraph import TigerGraphConnection

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

CONTEXT_QUERY = "temporal_training_context"
# Per-request statuses: the query continues with the next request and the
# client receives None for that key. Every other non-ok status is call-level.
PER_REQUEST_STATUSES = frozenset(
    {
        "invalid_request",
        "missing_entity",
        "invisible_entity",
        "history_capacity_exceeded",
        "nonmonotonic_pair_clock",
        "invalid_payment_fields",
        "invalid_event_roles",
    }
)
MAX_FETCH_KEYS = 2048
# A named tuple of exception types: formatters targeting Python 3.14 rewrite
# `except (A, B):` as `except A, B:`, which Python 3.12 and 3.13 cannot parse.
CONVERSION_ERRORS = (TypeError, ValueError)
# Node features TigerGraph may return; client groups (hub_indicator) never come from it.
KNOWN_NODE_FEATURES = frozenset(
    name
    for group, spec in FEATURE_GROUPS.items()
    if spec.path in ("node", "summary") and group not in CLIENT_GROUPS
    for name in spec.names
)


class TransientQueryError(RuntimeError):
    """A transport failure that is expected to clear (restart, overload, resume)."""


class ServerTimeoutError(TransientQueryError):
    """TigerGraph exceeded its query timeout on every allowed attempt of one request."""


class ContextTimeoutError(RuntimeError):
    """One context keeps timing out on the server, even as a single-key request.

    Fatal on purpose: whether a key times out depends on server load, so
    dropping it would make the training data differ between runs.
    """

    def __init__(self, key: ContextKey, hop: int, cause: str) -> None:
        super().__init__(
            f"TigerGraph timed out on context {key} (hop {hop}) even as a single-key request "
            f"({cause}). The context cannot be skipped without making the data depend on "
            "server load: retry when the server is less busy, or lower the sampler's "
            "max_history for this preparation."
        )
        self.key, self.hop = key, hop


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
    import requests
    from pyTigerGraph.common.exception import TigerGraphException

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


def is_transient(error: BaseException) -> bool:
    """True for failures worth retrying; contract and validation errors never are."""
    return failure_class(error) is not None


def _summary(error: BaseException) -> str:
    message = str(getattr(error, "message", None) or error)
    return f"{type(error).__name__}: {' '.join(message.split())[:200]}"


class QueryExecutor(Protocol):
    def run(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


class TigerGraphExecutor:
    """Installed-query access with per-class retry budgets (see failure_class).

    Credentials come from the repository .env and never enter cache metadata.
    - Availability failures are retried with capped, jittered exponential backoff
      until `max_outage_s` seconds have passed since the operation's first such
      failure. Worker threads share one "backoff until" time, so they pause
      together instead of hammering a resuming workspace.
    - Server timeouts are retried `timeout_retries` times (default once) and then
      raise ServerTimeoutError; other suspected-deterministic failures are
      retried once.
    - `max_attempts` (config max_query_attempts) caps the attempts that count:
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
        max_attempts: int = TRANSPORT_DEFAULTS["max_query_attempts"],
        max_outage_s: float = TRANSPORT_DEFAULTS["max_outage_s"],
        timeout_s: float = 300.0,
        base_delay_s: float = 4.0,
        max_delay_s: float = 60.0,
        slow_attempt_s: float = 30.0,
        size_limit: int = 64_000_000,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        if (
            not 1 <= max_attempts <= 20
            or max_outage_s < 0
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
            from mule_pattern_learner.tigergraph.client import Client
            from mule_pattern_learner.tigergraph.settings import Settings

            settings = Settings()
            client = self._retry(lambda: Client(settings), what="connect", attempts=None)
        self.client = client
        if self.client.graphname != "Mule_Pattern_Learner":
            raise ValueError("Training queries require Mule_Pattern_Learner")

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
                    reason = f"server timeout, {timeout_retries} retry(ies) allowed"
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
                    exhausted, reason = True, f"{attempts} attempt(s) allowed"
                elif attempts is None and counted >= self.max_attempts:
                    exhausted, reason = True, f"max_query_attempts = {self.max_attempts}"
                if exhausted:
                    failure = ServerTimeoutError if kind == SERVER_TIMEOUT else TransientQueryError
                    raise failure(
                        f"{label} failed after {total} attempt(s) ({reason}): {_summary(error)}"
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
                LOGGER.warning(
                    "TigerGraph %s failure in %s (attempt %d, retry in %.1fs): %s",
                    kind,
                    label,
                    total,
                    pause,
                    _summary(error),
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
        """Run a read-only connection operation under the retry policy."""
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

        # Parameter sizes name the request in retry logs (context requests: key count).
        sizes = {key: len(value) for key, value in params.items() if isinstance(value, list)}
        detail = (
            f"{sizes['node_ids']} keys"
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

    def gsql(self, text: str, *, what: str = "gsql") -> str:
        """Read-only GSQL statement (for example SHOW QUERY) with resume detection."""

        def operation(conn: TigerGraphConnection) -> str:
            output = str(conn.gsql(text))
            if looks_like_html(output):
                raise TransientQueryError("TigerGraph returned an HTML page instead of GSQL output")
            return output

        return self.call(operation, what=what)


def run_query(
    executor: QueryExecutor, name: str, params: dict[str, Any], *, timeout_s: float | None = None
) -> list[dict[str, Any]]:
    """Run with an explicit timeout when the executor supports it (test fakes need not)."""
    if timeout_s is not None and isinstance(executor, TigerGraphExecutor):
        return executor.run(name, params, timeout_s=timeout_s)
    return executor.run(name, params)


def checked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "status" in row and row["status"] != "ok":
            raise ValueError(f"TigerGraph rejected request: {row}")
    if not any(row.get("status") == "ok" for row in rows):
        raise ValueError("Query did not return a success status")
    return rows


def response_bound(sampler: SamplerPlan, hop: int) -> int:
    """Largest message count a context may contain at this hop."""
    return int(sampler.pool(hop).response_bound)


def validate_context(
    key: ContextKey,
    row: dict[str, Any],
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    *,
    require_encodings: bool = False,
) -> int:
    """Check one context against its key, clocks and the feature contract.

    Encodings are optional. When age_encoding is present (or required for a
    spot-check request), every vector is compared with the shared fourier64
    basis in one vectorized call. Returns the number of messages whose channel
    is outside CHANNELS; those are allowed and mapped to "other" downstream.
    """
    if any(row.get(name) != value for name, value in asdict(key).items()):
        raise ValueError("Returned context differs from requested entity/cutoff")
    if row.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Missing current feature contract; install the current context query")
    if row.get("basis_id") != BASIS_ID:
        raise ValueError("Fourier basis mismatch")
    messages: list[dict[str, Any]] = row["messages"]
    if len(messages) > response_bound(sampler, hop):
        raise ValueError("Query response exceeds the neighborhood bound")
    features: dict[str, Any] = row["features"]
    if "amount_ratios" in plan.groups and any(
        name not in features for name in AMOUNT_RATIO_FEATURES
    ):
        raise ValueError(
            "GSQL response is missing amount ratios; install the current context query"
        )
    if set(features) - KNOWN_NODE_FEATURES:
        raise ValueError("Unknown node feature in response")
    if features and not _finite_nonnegative(list(features.values())):
        raise ValueError("Features must be finite and nonnegative")
    if any(features.get(name, 0) > AMOUNT_RATIO_CAP for name in AMOUNT_RATIO_FEATURES):
        raise ValueError("GSQL amount ratio exceeds the feature contract")
    numeric = [
        (group, name)
        for group in ("flow_timing", "pair_history", "device_ip_context")
        if group in plan.groups
        for name in FEATURE_GROUPS[group].names
    ]
    if numeric and messages:
        table = [[message.get(name) for _, name in numeric] for message in messages]
        if not _finite_nonnegative(table):
            for message in messages:
                for group, name in numeric:
                    if not _finite_nonnegative([message.get(name)]):
                        raise ValueError(f"Missing or invalid {group} field: {name}")
    flow = "flow_timing" in plan.groups
    unknown_channels = 0
    events: list[dict[str, Any]] = []
    for message in messages:
        if message.get("stratum", "recent") not in STRATA:
            raise ValueError("Unknown sampling stratum")
        if message["node_type"] not in NODE_TYPES or message["relation"] not in RELATIONS:
            raise ValueError("Unknown entity or relation in neighborhood")
        if message["rail"] not in RAILS:
            raise ValueError("Unknown payment rail")
        if message.get("channel", "unknown") not in CHANNELS:
            unknown_channels += 1
        if flow:
            if message["flow_present"] and message["flow_censored"]:
                raise ValueError("Observed forward event cannot be censored")
            if not message["flow_present"] and (
                message["flow_delay_seconds"] or message["flow_ratio_present"]
            ):
                raise ValueError("Missing flow event must not have delay/amount evidence")
        if message["event_id"]:
            if not 0 < message["event_seq"] < key.cutoff_seq:
                raise ValueError("Future or invalid event sequence")
            if not 0 < message["event_ts_ms"] <= key.cutoff_ms:
                raise ValueError("Future or invalid event timestamp")
            if message["age_ms"] != key.cutoff_ms - message["event_ts_ms"]:
                raise ValueError("Event age differs from cutoff")
            if message["gap_ms"] < 0:
                raise ValueError("Negative predecessor gap")
            if not message["gap_present"] and message["gap_ms"]:
                raise ValueError("Missing predecessor must not have an encoding or gap")
            events.append(message)
        elif (message["event_seq"], message["event_ts_ms"]) != (key.cutoff_seq, key.cutoff_ms):
            raise ValueError("Association context changed the cutoff")
    age_map: dict[str, Any] = row.get("age_encoding") or {}
    gap_map: dict[str, Any] = row.get("gap_encoding") or {}
    if age_map or gap_map or (require_encodings and "time_encoding" in plan.groups):
        _check_encodings(events, age_map, gap_map)
    return unknown_channels


def _finite_nonnegative(values: list[Any]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except CONVERSION_ERRORS:
        return False
    return bool(np.isfinite(array).all() and (array >= 0).all())


def _check_encodings(
    events: list[dict[str, Any]], age_map: dict[str, Any], gap_map: dict[str, Any]
) -> None:
    keys = [message["relation"] + ":" + message["event_id"] for message in events]
    gap_keys = [k for k, message in zip(keys, events, strict=True) if message["gap_present"]]
    if set(gap_map) - set(gap_keys):
        raise ValueError("Missing predecessor must not have an encoding")
    if set(age_map) != set(keys) or set(gap_map) != set(gap_keys):
        raise ValueError("GSQL time encodings do not cover the returned events")
    if not keys:
        return
    deltas = [int(message["age_ms"]) for message in events] + [
        int(message["gap_ms"]) for message in events if message["gap_present"]
    ]
    try:
        vectors = np.asarray(
            [age_map[k] for k in keys] + [gap_map[k] for k in gap_keys], dtype=np.float64
        )
    except CONVERSION_ERRORS:
        raise ValueError("GSQL time encoding has an invalid shape") from None
    expected = fourier64(np.asarray(deltas, dtype=np.int64))
    if vectors.shape != expected.shape or not np.allclose(vectors, expected, atol=1e-5):
        raise ValueError("GSQL time encoding does not match the shared basis")


def _canonical(row: dict[str, Any]) -> dict[str, Any]:
    """Drop what depends on how keys were grouped into requests.

    request_index is a position inside one REST call. Verified Fourier vectors
    are dropped because batches recompute them on the device, so a context is
    identical whether or not its request was a spot check.
    """
    row = {name: value for name, value in row.items() if name != "request_index"}
    if row.get("status") == "ok":
        row["age_encoding"], row["gap_encoding"] = {}, {}
    return row


def query_context_rows(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
    diagnostics: Counter[str] | None = None,
    timeout_retries: int = 1,
) -> list[dict[str, Any]]:
    """One REST call; validated ok rows or per-request status rows, in key order."""
    if not batch or len({(k.scope_id, k.visibility_phase) for k in batch}) != 1:
        raise ValueError("Query batch must have one visibility scope and phase")
    if len(batch) > 64:
        raise ValueError("A context request carries at most 64 keys")
    params = {
        "node_types": [key.node_type for key in batch],
        "node_ids": [key.node_id for key in batch],
        "cutoff_seqs": [key.cutoff_seq for key in batch],
        "cutoff_times": [key.cutoff_ms for key in batch],
        **sampler.query_params(hop),
        "emit_encodings": emit_encodings,
        **plan.query_flags(hop),
        "scope_id": batch[0].scope_id,
        "visibility_phase": batch[0].visibility_phase,
    }
    if isinstance(executor, TigerGraphExecutor):
        result = executor.run(CONTEXT_QUERY, params, timeout_retries=timeout_retries)
    else:
        result = executor.run(CONTEXT_QUERY, params)
    indexed: dict[int, dict[str, Any]] = {}
    for row in result:
        if "request_index" not in row:
            if row.get("status", "ok") != "ok":
                raise ValueError(f"TigerGraph rejected the context call: {row}")
            continue
        index = int(row["request_index"])
        if index in indexed or not 0 <= index < len(batch):
            raise ValueError("Incomplete or duplicated query response")
        indexed[index] = row
    if len(indexed) != len(batch):
        raise ValueError("Incomplete or duplicated query response")
    rows = []
    for index, key in enumerate(batch):
        row = indexed[index]
        status = row.get("status")
        if status == "ok":
            unknown = validate_context(
                key, row, plan, sampler, hop, require_encodings=emit_encodings
            )
            if diagnostics is not None and unknown:
                diagnostics["unknown_channel"] += unknown
        elif status not in PER_REQUEST_STATUSES:
            raise ValueError(f"Unknown per-request status from TigerGraph: {row}")
        rows.append(row)
    if diagnostics is not None and emit_encodings:
        diagnostics["encoding_checks"] += 1
    return rows


def query_context_split(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
    diagnostics: Counter[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """query_context_rows that bisects a block TigerGraph times out on; (rows, calls).

    A multi-key request that exceeds the server timeout is split in two halves
    at once instead of being repeated, which isolates a slow key in about
    log2(block) extra calls. A single key is retried once and then raises
    ContextTimeoutError naming it; it is never mapped to None, because which
    keys time out depends on server load.
    """
    try:
        rows = query_context_rows(
            executor,
            batch,
            plan=plan,
            sampler=sampler,
            hop=hop,
            emit_encodings=emit_encodings,
            diagnostics=diagnostics,
            timeout_retries=0 if len(batch) > 1 else 1,
        )
    except ServerTimeoutError as error:
        if len(batch) == 1:
            raise ContextTimeoutError(batch[0], hop, _summary(error)) from error
        LOGGER.warning(
            "TigerGraph timed out on a %d-key context request (hop %d); splitting it",
            len(batch),
            hop,
        )
        if diagnostics is not None:
            diagnostics["timeout_splits"] += 1
        middle = len(batch) // 2
        options: dict[str, Any] = {
            "plan": plan,
            "sampler": sampler,
            "hop": hop,
            "emit_encodings": emit_encodings,
            "diagnostics": diagnostics,
        }
        left, left_calls = query_context_split(executor, batch[:middle], **options)
        right, right_calls = query_context_split(executor, batch[middle:], **options)
        return left + right, left_calls + right_calls
    return rows, 1


def query_context_batch(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
) -> list[dict[str, Any] | None]:
    """Validated contexts in key order; None where TigerGraph rejected one request."""
    return [
        row if row.get("status") == "ok" else None
        for row in query_context_rows(
            executor, batch, plan=plan, sampler=sampler, hop=hop, emit_encodings=emit_encodings
        )
    ]


def _check_source_limits(keys: list[ContextKey], hop: int) -> None:
    if len(keys) > MAX_FETCH_KEYS:
        raise ValueError(f"Context fetch is bounded to {MAX_FETCH_KEYS} items")
    if hop not in (1, 2):
        raise ValueError("Context hop must be 1 (roots) or 2 (children)")


def _default_sampler(per_relation: int | None) -> SamplerPlan:
    return SamplerPlan.from_config({} if per_relation is None else {"per_relation": per_relation})


class _EncodingCadence:
    """Requests 0, n, 2n, ... carry emit_encodings=True and are verified."""

    def __init__(self, every: int) -> None:
        if not 1 <= every <= 1_000_000:
            raise ValueError("encoding_check_every must be in [1,1000000]")
        self.every, self.requests = every, 0

    def next(self) -> bool:
        emit = self.requests % self.every == 0
        self.requests += 1
        return emit


class ContextStore:
    """SQLite cache keyed by immutable dataset identity, query contract, hop and clocks.

    A missing entry in offline mode is an error. Training cannot silently query
    a mutable live graph. Per-request rejections are cached as status rows and
    returned as None. The small LRU bounds decompressed host memory. One lock
    serializes access so batch-builder threads can share the store.
    """

    def __init__(
        self,
        path: Path,
        metadata: dict[str, Any],
        executor: QueryExecutor | None = None,
        *,
        per_relation: int | None = None,
        request_batch_size: int = TRANSPORT_DEFAULTS["request_batch_size"],
        plan: FeaturePlan = FeaturePlan(),
        sampler: SamplerPlan | None = None,
        encoding_check_every: int = TRANSPORT_DEFAULTS["encoding_check_every"],
        capacity: int = 256,
    ) -> None:
        if not 1 <= request_batch_size <= 64 or not 0 <= capacity <= 4096:
            raise ValueError("Unsupported query batch or cache capacity")
        self.plan = plan
        self.sampler = sampler or _default_sampler(per_relation)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.executor = path, executor
        self.request_batch_size, self.capacity = request_batch_size, capacity
        self.metadata = {
            **{
                k: v
                for k, v in metadata.items()
                if k
                not in (
                    "config_sha256",
                    "preparation",
                    "preparation_sha256",
                    "observed_labels_sha256",
                )
            },
            "contract": contract_fingerprint(),
            "extraction_groups": sorted(plan.groups),
            "sampler": {str(hop): self.sampler.query_params(hop) for hop in (1, 2)},
            "flags": {str(hop): plan.query_flags(hop) for hop in (1, 2)},
        }
        self._cadence = _EncodingCadence(encoding_check_every)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, value TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contexts (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        existing = self.conn.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if existing and json.loads(existing[0]) != json.loads(json.dumps(self.metadata)):
            self.conn.close()
            raise ValueError("Cache provenance mismatch; use a new dataset/cache directory")
        if not existing:
            self.conn.execute(
                "INSERT INTO metadata VALUES (1, ?)", (json.dumps(self.metadata, sort_keys=True),)
            )
            self.conn.commit()
        self.memory: OrderedDict[tuple[int, ContextKey], dict[str, Any]] = OrderedDict()
        self.query_calls = 0
        self.rejections: Counter[str] = Counter()
        self.rejections_by_hop: dict[int, Counter[str]] = {}
        self.diagnostics: Counter[str] = Counter()

    @staticmethod
    def _key(hop: int, key: ContextKey) -> str:
        return fingerprint({**asdict(key), "hop": hop})

    def close(self) -> None:
        with self._lock:
            self.conn.close()
            self.memory.clear()

    def _remember(self, hop: int, key: ContextKey, row: dict[str, Any]) -> None:
        self.memory[(hop, key)] = row
        self.memory.move_to_end((hop, key))
        while len(self.memory) > self.capacity:
            self.memory.popitem(last=False)

    def _read(self, hop: int, key: ContextKey) -> dict[str, Any] | None:
        cached = self.memory.get((hop, key))
        if cached is not None:
            self.memory.move_to_end((hop, key))
            return cached
        found = self.conn.execute(
            "SELECT value FROM contexts WHERE key=?", (self._key(hop, key),)
        ).fetchone()
        if not found:
            return None
        row = json.loads(zlib.decompress(found[0]))
        if row.get("status") == "ok":
            validate_context(key, row, self.plan, self.sampler, hop)
        elif row.get("status") not in PER_REQUEST_STATUSES:
            raise ValueError("Cached context has an unknown status")
        self._remember(hop, key, row)
        return row

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        _check_source_limits(keys, hop)
        with self._lock:
            rows: dict[ContextKey, dict[str, Any]] = {}
            missing = []
            for key in dict.fromkeys(keys):
                row = self._read(hop, key)
                if row is None:
                    missing.append(key)
                else:
                    rows[key] = row
            if missing and self.executor is None:
                raise ValueError(
                    f"Offline cache lacks {len(missing)} contexts; prepare this cohort first"
                )
            for start in range(0, len(missing), self.request_batch_size):
                batch = missing[start : start + self.request_batch_size]
                assert self.executor is not None
                result, calls = query_context_split(
                    self.executor,
                    batch,
                    plan=self.plan,
                    sampler=self.sampler,
                    hop=hop,
                    emit_encodings=self._cadence.next(),
                    diagnostics=self.diagnostics,
                )
                self.query_calls += calls
                for key, row in zip(batch, result, strict=True):
                    row = _canonical(row)
                    data = zlib.compress(json.dumps(row, allow_nan=False).encode())
                    self.conn.execute(
                        "INSERT OR REPLACE INTO contexts VALUES (?, ?)", (self._key(hop, key), data)
                    )
                    rows[key] = row
                    self._remember(hop, key, row)
                self.conn.commit()
            return _resolve(keys, rows, self.rejections, self.rejections_by_hop, hop)


def _resolve(
    keys: list[ContextKey],
    rows: dict[ContextKey, dict[str, Any]],
    rejections: Counter[str],
    by_hop: dict[int, Counter[str]],
    hop: int,
) -> list[dict[str, Any] | None]:
    """Rows in key order, None for rejected keys; count each rejected key once per fetch."""
    for key in rows:
        status = rows[key].get("status")
        if status != "ok":
            rejections[str(status)] += 1
            by_hop.setdefault(hop, Counter())[str(status)] += 1
    return [rows[key] if rows[key].get("status") == "ok" else None for key in keys]


_Work = tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]


def _pool_worker(work: queue.SimpleQueue[_Work | None]) -> None:
    """Run submitted calls until the None sentinel; never holds a reference to the pool."""
    while True:
        item = work.get()
        if item is None:
            return
        future, function, args = item
        if not future.set_running_or_notify_cancel():
            continue
        try:
            result = function(*args)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)
        del item, future, function, args


def _stop_workers(work: queue.SimpleQueue[_Work | None], threads: list[threading.Thread]) -> None:
    for _ in threads:
        work.put(None)


class _DaemonPool:
    """At most `workers` daemon threads running submitted calls (a ThreadPoolExecutor subset).

    ThreadPoolExecutor joins its workers when the interpreter exits, so a request
    abandoned after an error or Ctrl-C (a REST retry chain may last max_outage_s)
    would still hold up the exiting process. These workers are daemon threads:
    `shutdown(wait=False, cancel_futures=True)` cancels queued calls and returns at
    once, and the process may exit while a request is still in flight. Workers
    start on demand and stop when the pool is shut down or garbage collected.
    """

    def __init__(self, workers: int, name: str) -> None:
        self._workers, self._name = workers, name
        self._queue: queue.SimpleQueue[_Work | None] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._release = weakref.finalize(self, _stop_workers, self._queue, self._threads)

    def submit(self, function: Callable[..., T], /, *args: Any) -> Future[T]:
        future: Future[T] = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.put((future, function, args))
            if len(self._threads) < self._workers:
                thread = threading.Thread(
                    target=_pool_worker,
                    args=(self._queue,),
                    name=f"{self._name}_{len(self._threads)}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is not None:
                            item[0].cancel()
                self._release()
            threads = list(self._threads)
        if wait:
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join()


class ContextSource(Protocol):
    """Model-facing port independent of SQLite, HTTP or future streaming transports.

    fetch returns rows in key order, None where TigerGraph rejected a request
    (counted by status in rejections, once per rejected key and fetch).
    Implementations are thread-safe. Optional extras that callers look up with
    getattr: `rejections_by_hop` (the same counts per hop, 1 roots and 2 children)
    and a `wait` keyword on close (False: do not wait for in-flight requests).
    """

    plan: FeaturePlan
    sampler: SamplerPlan
    query_calls: int
    rejections: Counter[str]

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]: ...
    def close(self) -> None: ...


class StreamingContextSource:
    """Fetch only requested batch contexts with a bounded in-memory LRU; no disk.

    The LRU is keyed by (hop, ContextKey) because roots and children use
    different candidate pools and feature flags. Several batch-builder threads
    may call fetch concurrently: they share one request pool of `concurrency`
    workers, a lock guards the LRU and counters, a key already being fetched by
    another thread is awaited rather than requested twice, and each fetch keeps
    at most `concurrency` of its own requests in flight. The first request and
    every `encoding_check_every`-th request ask TigerGraph for Fourier vectors
    and verify them. This adapter bounds client memory, not TigerGraph scan work.
    Train against a frozen source for reproducibility.
    """

    def __init__(
        self,
        executor: QueryExecutor,
        *,
        plan: FeaturePlan = FeaturePlan(),
        sampler: SamplerPlan | None = None,
        capacity: int = TRANSPORT_DEFAULTS["context_lru_capacity"],
        request_batch_size: int = TRANSPORT_DEFAULTS["request_batch_size"],
        concurrency: int = TRANSPORT_DEFAULTS["query_concurrency"],
        encoding_check_every: int = TRANSPORT_DEFAULTS["encoding_check_every"],
        per_relation: int | None = None,
    ) -> None:
        if not 1 <= concurrency <= 16:
            raise ValueError("Query concurrency must be in [1,16]")
        if not 0 <= capacity <= 4096 or not 1 <= request_batch_size <= 64:
            raise ValueError("Invalid context source capacity or query size")
        self.plan = plan
        self.sampler = sampler or _default_sampler(per_relation)
        self.executor = executor
        self.capacity, self.request_batch_size = capacity, request_batch_size
        self.concurrency = concurrency
        self._cadence = _EncodingCadence(encoding_check_every)
        self.pool = _DaemonPool(concurrency, "temporal-context")
        self.memory: OrderedDict[tuple[int, ContextKey], dict[str, Any]] = OrderedDict()
        self.query_calls = 0
        self.rejections: Counter[str] = Counter()
        self.rejections_by_hop: dict[int, Counter[str]] = {}
        self.diagnostics: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._inflight: dict[tuple[int, ContextKey], Future[dict[ContextKey, dict[str, Any]]]] = {}
        self._closed = False

    def __enter__(self) -> StreamingContextSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        _check_source_limits(keys, hop)
        unique = list(dict.fromkeys(keys))
        rows: dict[ContextKey, dict[str, Any]] = {}
        shared: dict[ContextKey, Future[dict[ContextKey, dict[str, Any]]]] = {}
        blocks: list[tuple[list[ContextKey], Future[dict[ContextKey, dict[str, Any]]]]] = []
        with self._lock:
            if self._closed:
                raise RuntimeError("Context source is closed")
            missing = []
            for key in unique:
                cached = self.memory.get((hop, key))
                if cached is not None:
                    rows[key] = cached
                elif (hop, key) in self._inflight:
                    shared[key] = self._inflight[(hop, key)]
                else:
                    missing.append(key)
            self.diagnostics["lru_hits"] += len(rows)
            self.diagnostics["shared_inflight"] += len(shared)
            for start in range(0, len(missing), self.request_batch_size):
                block = missing[start : start + self.request_batch_size]
                holder: Future[dict[ContextKey, dict[str, Any]]] = Future()
                holder.set_running_or_notify_cancel()
                for key in block:
                    self._inflight[(hop, key)] = holder
                blocks.append((block, holder))
        fetched: dict[ContextKey, dict[str, Any]] = {}
        try:
            self._run_window(deque(blocks), hop, fetched)
        finally:
            with self._lock:
                # Cache in key order, so LRU recency does not depend on thread timing.
                for key in unique:
                    if key in fetched and self.capacity:
                        self.memory[(hop, key)] = fetched[key]
                    if (hop, key) in self.memory:
                        self.memory.move_to_end((hop, key))
                while len(self.memory) > self.capacity:
                    self.memory.popitem(last=False)
                for block, holder in blocks:
                    for key in block:
                        if self._inflight.get((hop, key)) is holder:
                            del self._inflight[(hop, key)]
        rows.update(fetched)
        for key, holder in shared.items():
            rows[key] = holder.result()[key]
        with self._lock:
            return _resolve(keys, rows, self.rejections, self.rejections_by_hop, hop)

    def _run_window(
        self,
        blocks: deque[tuple[list[ContextKey], Future[dict[ContextKey, dict[str, Any]]]]],
        hop: int,
        fetched: dict[ContextKey, dict[str, Any]],
    ) -> None:
        """Keep at most `concurrency` of this fetch's requests in flight."""
        running: dict[Future[list[dict[str, Any]]], tuple[list[ContextKey], Future[Any]]] = {}
        try:
            while blocks or running:
                while blocks and len(running) < self.concurrency:
                    block, holder = blocks.popleft()
                    with self._lock:
                        emit = self._cadence.next()
                    try:
                        future = self.pool.submit(self._query, block, hop, emit)
                    except BaseException:
                        blocks.appendleft((block, holder))
                        raise
                    running[future] = (block, holder)
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    block, holder = running.pop(future)
                    try:
                        entries = dict(zip(block, future.result(), strict=True))
                    except BaseException:
                        blocks.appendleft((block, holder))
                        raise
                    fetched.update(entries)
                    holder.set_result(entries)
        except BaseException as error:
            for future, pair in running.items():
                future.cancel()
                blocks.append(pair)
            for _, holder in blocks:
                holder.set_exception(error)
            raise

    def _query(self, block: list[ContextKey], hop: int, emit: bool) -> list[dict[str, Any]]:
        if self._closed:  # queued before close(wait=False) and not cancelled in time
            raise RuntimeError("Context source is closed")
        diagnostics: Counter[str] = Counter()
        try:
            result, calls = query_context_split(
                self.executor,
                block,
                plan=self.plan,
                sampler=self.sampler,
                hop=hop,
                emit_encodings=emit,
                diagnostics=diagnostics,
            )
        finally:
            with self._lock:
                self.diagnostics.update(diagnostics)
        with self._lock:
            self.query_calls += calls
        return [_canonical(row) for row in result]

    def close(self, *, wait: bool = True) -> None:
        """Cancel queued requests and free the LRU.

        wait=True also waits for requests already in flight. wait=False (after an
        error or Ctrl-C) returns at once; in-flight requests finish, or are dropped
        at interpreter exit, on their daemon worker threads.
        """
        with self._lock:
            self._closed = True
        self.pool.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            self.memory.clear()


def transport_settings(config: dict[str, Any]) -> dict[str, int]:
    """Transport knobs from a training config, with documented defaults."""
    result = {}
    for name, default in TRANSPORT_DEFAULTS.items():
        value = config.get(name, default)
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        result[name] = value
    return result


def live_executor(config: dict[str, Any]) -> TigerGraphExecutor:
    """A connected executor with the retry budgets of a training or preparation config."""
    transport = transport_settings(config)
    return TigerGraphExecutor(
        max_attempts=transport["max_query_attempts"], max_outage_s=transport["max_outage_s"]
    )


def sampler_pools(sampler: SamplerPlan) -> dict[str, dict[str, Any]]:
    """The query-relevant part of a sampler: what TigerGraph returns per hop."""
    return {"roots": sampler.query_params(1), "children": sampler.query_params(2)}


def open_context_source(
    dataset: Path, manifest: dict[str, Any], config: dict[str, Any] | None = None
) -> ContextSource:
    """Open the prepared transport; `config` is the training config (default: prepared).

    Streaming requests the extraction plan of the training model: the prepared
    extraction groups (training compares them with PREPARATION_KEYS) and the hop-2
    flags of the model's architecture. A SQLite cache holds exactly what
    preparation requested, so it is read with the prepared plan.
    """
    prepared: dict[str, Any] = manifest["config"]
    training = prepared if config is None else config
    streaming = manifest["source"].get("context_storage") == "stream"
    plan = extraction_plan(training if streaming else prepared)
    if sorted(plan.groups) != sorted(extraction_plan(prepared).groups):
        raise ValueError(
            "Extraction groups differ from preparation; prepare a new dataset "
            "(set prepared_id) or restore the prepared extraction_groups"
        )
    sampler = SamplerPlan.from_config(training)
    if sampler_pools(sampler) != sampler_pools(SamplerPlan.from_config(prepared)):
        raise ValueError(
            "Sampler candidate pools differ from preparation; prepare a new dataset "
            "(set prepared_id) or restore the prepared [sampler] pools"
        )
    transport = transport_settings(training)
    if streaming:
        from .installation import verify_frozen_source

        executor = live_executor(training)
        verify_frozen_source(executor, manifest)
        return StreamingContextSource(
            executor,
            plan=plan,
            sampler=sampler,
            capacity=transport["context_lru_capacity"],
            request_batch_size=transport["request_batch_size"],
            concurrency=transport["query_concurrency"],
            encoding_check_every=transport["encoding_check_every"],
        )
    return ContextStore(
        dataset / "contexts.sqlite",
        manifest["source"],
        plan=plan,
        sampler=sampler,
        request_batch_size=transport["request_batch_size"],
        encoding_check_every=transport["encoding_check_every"],
        capacity=transport["context_lru_capacity"],
    )


def extraction_groups(config: dict[str, Any]) -> tuple[str, ...]:
    """The configured extraction superset without client groups.

    `extraction_groups`, else `feature_groups`, else the legacy groups. Model
    variants (no_fourier, tabular) do not change it, so they share a preparation.
    """
    groups = config.get("extraction_groups") or config.get("feature_groups") or LEGACY_GROUPS
    return tuple(g for g in groups if g not in CLIENT_GROUPS)


def extraction_plan(config: dict[str, Any]) -> FeaturePlan:
    """What the context source asks TigerGraph for.

    Groups are `extraction_groups(config)`; client groups are computed locally.
    The architecture is the model's, so a split model skips summary groups at
    hop 2 while a single model keeps them.
    """
    model = FeaturePlan.from_config(config)
    groups = extraction_groups(config)
    missing = set(model.groups) - set(groups) - CLIENT_GROUPS
    if missing:
        raise ValueError(
            f"Extraction groups must cover all model inputs; missing {sorted(missing)}"
        )
    return FeaturePlan(groups, model.architecture)
