"""Context sources: the model-facing port, its streaming adapter, and factories.

StreamingContextSource requests each batch's contexts from TigerGraph through
context_query and keeps a bounded LRU. It returns rows in key order, with None where
TigerGraph rejected a request. check_coverage and close_source work with any
ContextSource.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, wait
import inspect
from pathlib import Path
import queue
import threading
from typing import Any, Protocol, TypeVar
import weakref

from ..config import TRANSPORT_DEFAULTS
from ..contract.feature_groups import FeaturePlan, extraction_plan
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan, sampler_pools
from ..tigergraph.context_query import query_context_split
from ..tigergraph.executor import QueryExecutor, live_executor, transport_settings
from ..tigergraph.provenance import verify_frozen_source

T = TypeVar("T")

MAX_FETCH_KEYS = 2048


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
    if len(keys) > MAX_FETCH_KEYS:
        raise ValueError(f"Context fetch is bounded to {MAX_FETCH_KEYS} items")
    if hop not in (1, 2):
        raise ValueError("Context hop must be 1 (roots) or 2 (children)")


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
    """Model-facing port independent of the transport (HTTP today, a disk cache later).

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
        sampler: SamplerPlan = SamplerPlan(),
        capacity: int = TRANSPORT_DEFAULTS["context_lru_capacity"],
        request_batch_size: int = TRANSPORT_DEFAULTS["request_batch_size"],
        concurrency: int = TRANSPORT_DEFAULTS["query_concurrency"],
        encoding_check_every: int = TRANSPORT_DEFAULTS["encoding_check_every"],
    ) -> None:
        if not 1 <= concurrency <= 16:
            raise ValueError("Query concurrency must be in [1,16]")
        if not 0 <= capacity <= 4096 or not 1 <= request_batch_size <= 64:
            raise ValueError("Invalid context source capacity or query size")
        self.plan = plan
        self.sampler = sampler
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


def streaming_source(
    executor: QueryExecutor, plan: FeaturePlan, sampler: SamplerPlan, config: dict[str, Any]
) -> StreamingContextSource:
    """Live source with the transport settings of a training configuration."""
    transport = transport_settings(config)
    return StreamingContextSource(
        executor,
        plan=plan,
        sampler=sampler,
        capacity=transport["context_lru_capacity"],
        request_batch_size=transport["request_batch_size"],
        concurrency=transport["query_concurrency"],
        encoding_check_every=transport["encoding_check_every"],
    )


def open_context_source(
    dataset: Path, manifest: dict[str, Any], config: dict[str, Any] | None = None
) -> ContextSource:
    """Open the live source of a prepared dataset; `config` is the training config.

    It requests the training model's groups and hop-2 flags (extraction_plan) with the
    prepared candidate pools. `config` defaults to the prepared configuration.
    """
    prepared: dict[str, Any] = manifest["config"]
    training = prepared if config is None else config
    plan = extraction_plan(training)
    sampler = SamplerPlan.from_config(training)
    if sampler_pools(sampler) != sampler_pools(SamplerPlan.from_config(prepared)):
        raise ValueError(
            f"Sampler candidate pools differ from the preparation in {dataset}; prepare a "
            "new dataset (set prepared_id) or restore the prepared [sampler] pools"
        )
    executor = live_executor(training)
    verify_frozen_source(executor, manifest)
    return streaming_source(executor, plan, sampler, training)


def check_coverage(store: ContextSource, plan: FeaturePlan, sampler: SamplerPlan) -> None:
    """The source must request every input the model reads, with the model's pools."""
    source_plan = getattr(store, "plan", None)
    source_sampler = getattr(store, "sampler", None)
    if not isinstance(source_plan, FeaturePlan) or not isinstance(source_sampler, SamplerPlan):
        raise ValueError("Context source must expose its FeaturePlan and SamplerPlan")
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

    Sources whose ``close`` accepts ``wait`` are closed with ``wait=False`` after an
    error or KeyboardInterrupt, so the error surfaces without waiting for REST retries.
    """
    close = store.close
    if failed and "wait" in inspect.signature(close).parameters:
        close(wait=False)  # pyright: ignore[reportCallIssue]
    else:
        close()
