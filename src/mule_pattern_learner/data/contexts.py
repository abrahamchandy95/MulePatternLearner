"""Context sources: the model-facing port, the source that streams contexts, its opener.

Batching, training and scoring read contexts through a ContextReader. ContextSource
is the one that reads them from the graph: it requests each batch's contexts through
a ContextFetcher (ports.py), keeps a bounded LRU and, given a dataset's context cache,
reads and writes a disk tier (context_cache.py). It returns rows in key order, with
None where TigerGraph rejected a request. Every reader counts what it was asked for in
a ContextCounts. check_coverage and close_source work with any ContextReader, and a
parameter that takes one is named ``contexts``. A ContextOpener opens the
source of a prepared dataset; the pipeline passes pipeline.connect.open_context_source
to the use cases that need one.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
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
from ..contract.fingerprints import hash64
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..paths import DatasetPaths
from ..runtime.workers import DaemonPool
from .context_cache import ContextCache, ContextTier, DiskTier, MemoryTier
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


def context_hash(key: ContextKey, hop: int) -> int:
    """A 63-bit hash of one context, the same in every process (fits an int64 tensor)."""
    parts = (key.node_type, key.node_id, key.cutoff_seq, key.cutoff_ms, key.scope_id)
    return hash64(hop, *parts, key.visibility_phase) >> 1


@dataclass
class ContextCounts:
    """What a context source was asked for: every context, the distinct ones, cache hits.

    requested counts the contexts fetches asked for (a key repeated within one fetch
    once), memory_hits those served from memory and disk_hits those read from the disk
    tier, both without a request. A context that a fetch awaited from another counts as
    a disk hit when the other read it from disk. seen holds the context_hash of every
    distinct (hop, key) asked for; a resumed run restores it, so distinct counts every
    segment of the run.
    """

    requested: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    seen: set[int] = field(default_factory=set[int])

    def ask(self, keys: Iterable[ContextKey], hop: int) -> None:
        """Count one fetch's distinct keys."""
        for key in keys:
            self.requested += 1
            self.seen.add(context_hash(key, hop))

    @property
    def distinct(self) -> int:
        return len(self.seen)


@dataclass(frozen=True)
class _Obtained:
    """What a fetch obtained for the keys it claimed: their rows, and those read from disk."""

    rows: dict[ContextKey, dict[str, Any]]
    from_disk: frozenset[ContextKey]


class ContextReader(Protocol):
    """Model-facing port independent of the transport (HTTP, with or without a disk tier).

    fetch returns rows in key order, None where TigerGraph rejected a request
    (counted by status in rejections, once per rejected key and fetch, and in
    rejections_by_hop per hop, 1 roots and 2 children). counts holds what fetches
    asked for (ContextCounts). close with wait=False does not wait for requests in
    flight. Implementations are thread-safe.
    """

    @property
    def plan(self) -> FeaturePlan: ...
    @property
    def sampler(self) -> SamplerPlan: ...
    @property
    def database_calls(self) -> int: ...
    @property
    def rejections(self) -> Counter[str]: ...
    @property
    def rejections_by_hop(self) -> dict[int, Counter[str]]: ...
    @property
    def counts(self) -> ContextCounts: ...

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]: ...
    def close(self, *, wait: bool = True) -> None: ...


class ContextSource:
    """Fetch only requested batch contexts, through a bounded LRU and an optional disk tier.

    The LRU is ``memory``, a context_cache.MemoryTier of ``capacity`` rows keyed by
    (hop, ContextKey). Given a dataset's context cache (``cache``), ``disk`` is its
    context_cache.DiskTier: a context memory lacks is read from disk before it is
    requested, and every row TigerGraph returns is written there as it came, so a later
    source of the same dataset, feature flags and pools requests it no more while its
    entry lasts. Rows read from disk are served exactly as requested ones, and both go
    into the LRU. Several batch-builder threads may call fetch concurrently: they share
    one request pool of `concurrency` workers, and a lock guards the LRU and counters.
    Under the lock a fetch claims the keys that neither memory holds nor another fetch
    has claimed; it reads them from disk outside the lock and requests only those the
    disk lacks, keeping at most `concurrency` of its own requests in flight. A fetch
    that asks for a claimed key awaits the claim rather than reading or requesting the
    key again, so each context is requested once while its entry lasts, however the LRU
    churns. The first request and every `encoding_check_every`-th request ask
    TigerGraph for Fourier vectors and verify them, counted over every fetch the source
    serves (both audits of an evaluation share one source). A row read from disk is not
    checked again: it was validated when its request wrote it, so a source that reads
    every context from disk verifies no encoding. This adapter bounds client memory,
    not TigerGraph scan work. Train against a frozen source for reproducibility: the
    disk tier's entries are the frozen source's rows.
    """

    def __init__(
        self,
        fetcher: ContextFetcher,
        *,
        plan: FeaturePlan,
        sampler: SamplerPlan,
        capacity: int = DEFAULT_CONFIG.transport.context_lru_capacity,
        request_batch_size: int = DEFAULT_CONFIG.transport.request_batch_size,
        concurrency: int = DEFAULT_CONFIG.transport.query_concurrency,
        encoding_check_every: int = DEFAULT_CONFIG.transport.encoding_check_every,
        cache: ContextCache | None = None,
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
        self.concurrency, self.encoding_check_every = concurrency, encoding_check_every
        self._cadence = _EncodingCadence(encoding_check_every)
        self.pool = DaemonPool(concurrency, "context-requests")
        self.memory = MemoryTier(capacity)
        self.disk = DiskTier(cache, plan=plan, sampler=sampler) if cache is not None else None
        self.database_calls = 0
        self.rejections: Counter[str] = Counter()
        self.rejections_by_hop: dict[int, Counter[str]] = {}
        self.counts = ContextCounts()
        self.diagnostics: Counter[str] = Counter()
        self._lock = threading.Lock()
        # The claim of every key a fetch is reading from disk or requesting.
        self._inflight: dict[tuple[int, ContextKey], Future[_Obtained]] = {}
        self._closed = False

    def __enter__(self) -> ContextSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        _check_source_limits(keys, hop)
        unique = list(dict.fromkeys(keys))
        shared: dict[ContextKey, Future[_Obtained]] = {}
        claimed: list[ContextKey] = []
        claim: Future[_Obtained] = Future()
        claim.set_running_or_notify_cancel()
        with self._lock:
            if self._closed:
                raise RuntimeError("Context source is closed")
            rows = self.memory.get(hop, unique)
            for key in unique:
                if key in rows:
                    continue
                if (hop, key) in self._inflight:
                    shared[key] = self._inflight[(hop, key)]
                else:
                    claimed.append(key)
                    self._inflight[(hop, key)] = claim
            self.counts.ask(unique, hop)
            self.counts.memory_hits += len(rows)
            self.diagnostics["shared_inflight"] += len(shared)
        rows.update(self._obtain(unique, claimed, hop, claim).rows)
        from_disk = 0
        for key, holder in shared.items():
            other = holder.result()
            rows[key] = other.rows[key]
            from_disk += key in other.from_disk
        with self._lock:
            # A context the claiming fetch read from disk reached this one without a request.
            self.counts.disk_hits += from_disk
            return _resolve(keys, rows, self.rejections, self.rejections_by_hop, hop)

    def _obtain(
        self,
        unique: list[ContextKey],
        claimed: list[ContextKey],
        hop: int,
        claim: Future[_Obtained],
    ) -> _Obtained:
        """Read the claimed keys from disk, request those it lacks, then answer the claim.

        The disk is read outside the lock: no other fetch reads or requests a claimed
        key, it awaits the claim. The rows obtained go into the LRU with every key of
        the fetch (``unique``), and the claim is released with them, or with the error.
        """
        stored: dict[ContextKey, dict[str, Any]] = {}
        fetched: dict[ContextKey, dict[str, Any]] = {}
        try:
            if self.disk is not None and claimed:
                stored = {key: _canonical(row) for key, row in self.disk.get(hop, claimed).items()}
                with self._lock:
                    self.counts.disk_hits += len(stored)
            missing = [key for key in claimed if key not in stored]
            size = self.request_batch_size
            blocks = deque(missing[start : start + size] for start in range(0, len(missing), size))
            self._run_window(blocks, hop, fetched)
        except BaseException as error:
            claim.set_exception(error)
            raise
        finally:
            with self._lock:
                self.memory.put(hop, unique, {**stored, **fetched})
                for key in claimed:
                    if self._inflight.get((hop, key)) is claim:
                        del self._inflight[(hop, key)]
        obtained = _Obtained({**stored, **fetched}, frozenset(stored))
        claim.set_result(obtained)
        return obtained

    def _run_window(
        self, blocks: deque[list[ContextKey]], hop: int, fetched: dict[ContextKey, dict[str, Any]]
    ) -> None:
        """Request the blocks, keeping at most `concurrency` of this fetch's requests in flight."""
        running: dict[Future[list[dict[str, Any]]], list[ContextKey]] = {}
        try:
            while blocks or running:
                while blocks and len(running) < self.concurrency:
                    block = blocks.popleft()
                    with self._lock:
                        emit = self._cadence.next()
                    running[self.pool.submit(self._query, block, hop, emit)] = block
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    fetched.update(zip(running.pop(future), future.result(), strict=True))
        except BaseException:
            for future in running:
                future.cancel()
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
            self.database_calls += calls
        if self.disk is not None:
            # TigerGraph's own rows, before _canonical drops what serving does not use.
            self.disk.put(hop, block, dict(zip(block, result, strict=True)))
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
            for tier in self._tiers():
                tier.close()

    def _tiers(self) -> tuple[ContextTier, ...]:
        """The cache tiers, in the order fetch consults them before TigerGraph."""
        return (self.memory,) if self.disk is None else (self.memory, self.disk)


def build_context_source(
    fetcher: ContextFetcher,
    plan: FeaturePlan,
    sampler: SamplerPlan,
    transport: TransportConfig,
    cache: ContextCache | None = None,
) -> ContextSource:
    """A source with the LRU, request size and concurrency of a transport section.

    ``cache`` gives it the disk tier of a dataset's context cache.
    """
    return ContextSource(
        fetcher,
        plan=plan,
        sampler=sampler,
        capacity=transport.context_lru_capacity,
        request_batch_size=transport.request_batch_size,
        concurrency=transport.query_concurrency,
        encoding_check_every=transport.encoding_check_every,
        cache=cache,
    )


class ContextOpener(Protocol):
    """Opens the source of a prepared dataset for a training or scoring configuration.

    Use cases call it once their own checks passed, so a refused run never connects.
    """

    def __call__(
        self, dataset: DatasetPaths, manifest: dict[str, Any], config: RunConfig
    ) -> ContextReader: ...


def check_coverage(contexts: ContextReader, plan: FeaturePlan, sampler: SamplerPlan) -> None:
    """The source must request every input the model reads, with the model's pools."""
    source_plan, source_sampler = contexts.plan, contexts.sampler
    # A model of the root's own inputs never fetches children, so only its first hop
    # matters.
    for hop in (1,) if plan.root_only else (1, 2):
        have = source_plan.query_flags(hop)
        missing = sorted(k for k, v in plan.query_flags(hop).items() if v and not have.get(k))
        if missing:
            raise ValueError(f"Context source does not request {missing} at hop {hop}")
    if (source_sampler.roots, source_sampler.children) != (sampler.roots, sampler.children):
        raise ValueError("Context source candidate pools differ from the model sampler")


def close_source(contexts: ContextReader, *, failed: bool) -> None:
    """Close a context source; after a failure, do not wait for its in-flight requests.

    After an error or KeyboardInterrupt the source is closed with ``wait=False``, so the
    error surfaces without waiting for REST retries.
    """
    contexts.close(wait=not failed)
