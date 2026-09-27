"""Context sources: the model-facing port, its streaming adapter, and its opener.

StreamingContextSource requests each batch's contexts through a ContextFetcher
(ports.py) and keeps a bounded LRU. It returns rows in key order, with None where
TigerGraph rejected a request. check_coverage and close_source work with any
ContextSource. A ContextOpener opens the source of a prepared dataset; the pipeline
passes pipeline.connect.open_context_source to the use cases that need one.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, wait
import threading
from typing import Any, Protocol

from ..config import DEFAULT_CONFIG, RunConfig, TransportConfig
from ..contract.bounds import (
    BATCH_CONTEXTS,
    CONTEXT_LRU_CAPACITY,
    ENCODING_CHECK_EVERY,
    QUERY_CONCURRENCY,
    REQUEST_KEYS,
)
from ..contract.feature_groups import FeaturePlan
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..paths import DatasetPaths
from ..runtime.workers import DaemonPool
from .ports import ContextFetcher


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


def _check_source_limits(keys: list[ContextKey], hop: int) -> None:
    if len(keys) > BATCH_CONTEXTS:
        raise ValueError(f"Context fetch is bounded to {BATCH_CONTEXTS} items")
    if hop not in (1, 2):
        raise ValueError("Context hop must be 1 (roots) or 2 (children)")


class _EncodingCadence:
    """Requests 0, n, 2n, ... carry emit_encodings=True and are verified."""

    def __init__(self, every: int) -> None:
        if not ENCODING_CHECK_EVERY.holds(every):
            raise ValueError(
                "encoding_check_every must be in "
                f"[{ENCODING_CHECK_EVERY.low},{ENCODING_CHECK_EVERY.high}]"
            )
        self.every, self.requests = every, 0

    def next(self) -> bool:
        emit = self.requests % self.every == 0
        self.requests += 1
        return emit


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


class ContextSource(Protocol):
    """Model-facing port independent of the transport (HTTP today, a disk cache later).

    fetch returns rows in key order, None where TigerGraph rejected a request
    (counted by status in rejections, once per rejected key and fetch, and in
    rejections_by_hop per hop, 1 roots and 2 children). close with wait=False does not
    wait for requests in flight. Implementations are thread-safe.
    """

    @property
    def plan(self) -> FeaturePlan: ...
    @property
    def sampler(self) -> SamplerPlan: ...
    @property
    def query_calls(self) -> int: ...
    @property
    def rejections(self) -> Counter[str]: ...
    @property
    def rejections_by_hop(self) -> dict[int, Counter[str]]: ...

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]: ...
    def close(self, *, wait: bool = True) -> None: ...


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
        fetcher: ContextFetcher,
        *,
        plan: FeaturePlan = FeaturePlan(),
        sampler: SamplerPlan = SamplerPlan(),
        capacity: int = DEFAULT_CONFIG.transport.context_lru_capacity,
        request_batch_size: int = DEFAULT_CONFIG.transport.request_batch_size,
        concurrency: int = DEFAULT_CONFIG.transport.query_concurrency,
        encoding_check_every: int = DEFAULT_CONFIG.transport.encoding_check_every,
    ) -> None:
        if not QUERY_CONCURRENCY.holds(concurrency):
            raise ValueError(
                f"Query concurrency must be in [{QUERY_CONCURRENCY.low},{QUERY_CONCURRENCY.high}]"
            )
        if not CONTEXT_LRU_CAPACITY.holds(capacity) or not REQUEST_KEYS.holds(request_batch_size):
            raise ValueError("Invalid context source capacity or query size")
        self.plan = plan
        self.sampler = sampler
        self.fetcher = fetcher
        self.capacity, self.request_batch_size = capacity, request_batch_size
        self.concurrency = concurrency
        self._cadence = _EncodingCadence(encoding_check_every)
        self.pool = DaemonPool(concurrency, "temporal-context")
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
            result, calls = self.fetcher.request(
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


def streaming_source(
    fetcher: ContextFetcher, plan: FeaturePlan, sampler: SamplerPlan, transport: TransportConfig
) -> StreamingContextSource:
    """Live source with the LRU, request size and concurrency of a transport section."""
    return StreamingContextSource(
        fetcher,
        plan=plan,
        sampler=sampler,
        capacity=transport.context_lru_capacity,
        request_batch_size=transport.request_batch_size,
        concurrency=transport.query_concurrency,
        encoding_check_every=transport.encoding_check_every,
    )


class ContextOpener(Protocol):
    """Opens the source of a prepared dataset for a training or scoring configuration.

    Use cases call it once their own checks passed, so a refused run never connects.
    """

    def __call__(
        self, dataset: DatasetPaths, manifest: dict[str, Any], config: RunConfig
    ) -> ContextSource: ...


def check_coverage(store: ContextSource, plan: FeaturePlan, sampler: SamplerPlan) -> None:
    """The source must request every input the model reads, with the model's pools."""
    source_plan, source_sampler = store.plan, store.sampler
    # A summary model never fetches children, so only its first hop matters.
    for hop in (1,) if plan.architecture == "summary" else (1, 2):
        have = source_plan.query_flags(hop)
        missing = sorted(k for k, v in plan.query_flags(hop).items() if v and not have.get(k))
        if missing:
            raise ValueError(f"Context source does not request {missing} at hop {hop}")
    if (source_sampler.roots, source_sampler.children) != (sampler.roots, sampler.children):
        raise ValueError("Context source candidate pools differ from the model sampler")


def close_source(store: ContextSource, *, failed: bool) -> None:
    """Close a context source; after a failure, do not wait for its in-flight requests.

    After an error or KeyboardInterrupt the source is closed with ``wait=False``, so the
    error surfaces without waiting for REST retries.
    """
    store.close(wait=not failed)
