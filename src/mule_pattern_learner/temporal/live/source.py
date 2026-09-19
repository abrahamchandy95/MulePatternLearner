"""Read-only query access with bounded streaming and optional durable caching."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Protocol
import zlib

import numpy as np

from ..encoding import BASIS_ID, fourier64
from .contract import (
    ContextKey,
    NODE_TYPES,
    RELATIONS,
    RAILS,
    contract_fingerprint,
    fingerprint,
    AMOUNT_RATIO_FEATURES,
    AMOUNT_RATIO_CAP,
)


class QueryExecutor(Protocol):
    def run(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


class TigerGraphExecutor:
    """Uses the existing .env connection; credentials never enter cache metadata."""

    def __init__(self) -> None:
        from mule_pattern_learner.tigergraph.client import Client
        from mule_pattern_learner.tigergraph.settings import Settings

        self.client = Client(Settings())
        if self.client.graphname != "Mule_Pattern_Learner":
            raise ValueError("Training queries require Mule_Pattern_Learner")

    def run(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self.client.conn.runInstalledQuery(
            name,
            params,
            usePost=True,
            timeout=300_000,
            sizeLimit=32_000_000,
        )


def checked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "status" in row and row["status"] != "ok":
            raise ValueError(f"TigerGraph rejected request: {row}")
    if not any(row.get("status") == "ok" for row in rows):
        raise ValueError("Query did not return a success status")
    return rows


def validate_context(key: ContextKey, row: dict[str, Any]) -> None:
    if any(row.get(name) != value for name, value in asdict(key).items()):
        raise ValueError("Returned context differs from requested entity/cutoff")
    if row.get("basis_id") != BASIS_ID:
        raise ValueError("Fourier basis mismatch")
    if len(row["messages"]) > len(RELATIONS) * 8:
        raise ValueError("Query response exceeds the neighborhood bound")
    if any(name not in row["features"] for name in AMOUNT_RATIO_FEATURES):
        raise ValueError(
            "GSQL response is missing amount ratios; install the current context query"
        )
    if any(row["features"][name] > AMOUNT_RATIO_CAP for name in AMOUNT_RATIO_FEATURES):
        raise ValueError("GSQL amount ratio exceeds the feature contract")
    for value in row["features"].values():
        if not np.isfinite(value) or value < 0:
            raise ValueError("Features must be finite and nonnegative")
    for message in row["messages"]:
        if message["node_type"] not in NODE_TYPES or message["relation"] not in RELATIONS:
            raise ValueError("Unknown entity or relation in neighborhood")
        if message["rail"] not in RAILS:
            raise ValueError("Unknown payment rail")
        if message["event_id"]:
            if not 0 < message["event_seq"] < key.cutoff_seq:
                raise ValueError("Future or invalid event sequence")
            if not 0 < message["event_ts_ms"] <= key.cutoff_ms:
                raise ValueError("Future or invalid event timestamp")
            if message["age_ms"] != key.cutoff_ms - message["event_ts_ms"]:
                raise ValueError("Event age differs from cutoff")
            event_key = message["relation"] + ":" + message["event_id"]
            checks = [(row["age_encoding"][event_key], message["age_ms"])]
            if message["gap_present"]:
                checks.append((row["gap_encoding"][event_key], message["gap_ms"]))
            elif message["gap_ms"] or event_key in row["gap_encoding"]:
                raise ValueError("Missing predecessor must not have an encoding")
            for vector, delta in checks:
                expected = fourier64(np.array([delta], dtype=np.int64))[0]
                if np.shape(vector) != (64,) or not np.allclose(vector, expected, atol=1e-5):
                    raise ValueError("GSQL time encoding does not match the shared basis")
        elif (message["event_seq"], message["event_ts_ms"]) != (key.cutoff_seq, key.cutoff_ms):
            raise ValueError("Association context changed the cutoff")


class ContextStore:
    """SQLite cache keyed by immutable dataset identity, query contract and clocks.

    A missing entry in offline mode is an error. Training cannot silently query
    a mutable live graph. The small LRU bounds decompressed host memory.
    """

    def __init__(
        self,
        path: Path,
        metadata: dict[str, Any],
        executor: QueryExecutor | None = None,
        *,
        per_relation: int = 2,
        request_batch_size: int = 16,
    ) -> None:
        if not 1 <= per_relation <= 8 or not 1 <= request_batch_size <= 16:
            raise ValueError("Unsupported query batch or fanout")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.executor = path, executor
        self.per_relation, self.request_batch_size = per_relation, request_batch_size
        self.metadata = {
            **metadata,
            "contract": contract_fingerprint(),
            "per_relation": per_relation,
        }
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, value TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contexts (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        existing = self.conn.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if existing and json.loads(existing[0]) != self.metadata:
            self.conn.close()
            raise ValueError("Cache provenance mismatch; use a new dataset/cache directory")
        if not existing:
            self.conn.execute(
                "INSERT INTO metadata VALUES (1, ?)", (json.dumps(self.metadata, sort_keys=True),)
            )
            self.conn.commit()
        self.memory: OrderedDict[ContextKey, dict[str, Any]] = OrderedDict()
        self.query_calls = 0

    def close(self) -> None:
        self.conn.close()

    def _read(self, key: ContextKey) -> dict[str, Any] | None:
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        found = self.conn.execute(
            "SELECT value FROM contexts WHERE key=?", (fingerprint(asdict(key)),)
        ).fetchone()
        if not found:
            return None
        row = json.loads(zlib.decompress(found[0]))
        validate_context(key, row)
        self.memory[key] = row
        if len(self.memory) > 256:
            self.memory.popitem(last=False)
        return row

    def fetch(self, keys: list[ContextKey]) -> list[dict[str, Any]]:
        if len(keys) > 2048:
            raise ValueError("Context fetch is bounded to 2048 items")
        unique = list(dict.fromkeys(keys))
        missing = [key for key in unique if self._read(key) is None]
        if missing and self.executor is None:
            raise ValueError(
                f"Offline cache lacks {len(missing)} contexts; prepare this cohort first"
            )
        for start in range(0, len(missing), self.request_batch_size):
            batch = missing[start : start + self.request_batch_size]
            assert self.executor is not None
            rows = query_context_batch(self.executor, batch, self.per_relation)
            self.query_calls += 1
            for key, row in zip(batch, rows, strict=True):
                data = zlib.compress(json.dumps(row, allow_nan=False).encode())
                self.conn.execute(
                    "INSERT INTO contexts VALUES (?, ?)", (fingerprint(asdict(key)), data)
                )
            self.conn.commit()
        result = [self._read(key) for key in keys]
        assert all(row is not None for row in result)
        return result  # type: ignore[return-value]


class ContextSource(Protocol):
    """Model-facing port independent of SQLite, HTTP or future streaming transports."""

    query_calls: int

    def fetch(self, keys: list[ContextKey]) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def query_context_batch(
    executor: QueryExecutor, batch: list[ContextKey], per_relation: int
) -> list[dict[str, Any]]:
    if not batch or len({(k.scope_id, k.visibility_phase) for k in batch}) != 1:
        raise ValueError("Query batch must have one visibility scope and phase")
    result = checked_rows(
        executor.run(
            "temporal_training_context",
            {
                "node_types": [key.node_type for key in batch],
                "node_ids": [key.node_id for key in batch],
                "cutoff_seqs": [key.cutoff_seq for key in batch],
                "cutoff_times": [key.cutoff_ms for key in batch],
                "per_relation": per_relation,
                "scope_id": batch[0].scope_id,
                "visibility_phase": batch[0].visibility_phase,
            },
        )
    )
    indexed = {int(row["request_index"]): row for row in result if "request_index" in row}
    if len(result) != len(batch) or set(indexed) != set(range(len(batch))):
        raise ValueError("Incomplete or duplicated query response")
    rows = []
    for index, key in enumerate(batch):
        row = indexed[index]
        validate_context(key, row)
        rows.append(row)
    return rows


class StreamingContextSource:
    """Fetch only requested batch contexts with a bounded in-memory LRU; no disk.

    This HTTP adapter bounds client context storage, not TigerGraph scan work.
    Seed metadata is paged separately; query concurrency and prefetch are bounded.
    Broker transport and temporal indexes remain separate production work.
    Train against a frozen source for reproducibility.
    """

    def __init__(
        self,
        executor: QueryExecutor,
        *,
        per_relation: int = 2,
        capacity: int = 64,
        request_batch_size: int = 16,
        concurrency: int = 2,
    ) -> None:
        if not 1 <= concurrency <= 4:
            raise ValueError("Query concurrency must be in [1,4]")
        if (
            not 0 <= capacity <= 256
            or not 1 <= per_relation <= 8
            or not 1 <= request_batch_size <= 16
        ):
            raise ValueError("Invalid context source capacity or query size")
        self.executor, self.per_relation = executor, per_relation
        self.capacity, self.request_batch_size = capacity, request_batch_size
        self.pool = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="temporal-sampler"
        )
        self.concurrency = concurrency
        self.memory: OrderedDict[ContextKey, dict[str, Any]] = OrderedDict()
        self.query_calls = 0

    def fetch(self, keys: list[ContextKey]) -> list[dict[str, Any]]:
        if len(keys) > 2048:
            raise ValueError("Streaming fetch is bounded to 2048 batch contexts")
        resolved = {}
        missing = []
        for key in dict.fromkeys(keys):
            if key in self.memory:
                self.memory.move_to_end(key)
                resolved[key] = self.memory[key]
            else:
                missing.append(key)
        blocks = [
            missing[start : start + self.request_batch_size]
            for start in range(0, len(missing), self.request_batch_size)
        ]
        fetch = partial(query_context_batch, self.executor, per_relation=self.per_relation)
        responses = self.pool.map(fetch, blocks, buffersize=self.concurrency)
        for batch, rows in zip(blocks, responses, strict=True):
            self.query_calls += 1
            for key, row in zip(batch, rows, strict=True):
                resolved[key] = row
                self.memory[key] = row
                while len(self.memory) > self.capacity:
                    self.memory.popitem(last=False)
        return [resolved[key] for key in keys]

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.memory.clear()


def open_context_source(dataset: Path, manifest: dict[str, Any]) -> ContextSource:
    per_relation = int(manifest["config"].get("per_relation", 2))
    if manifest["source"].get("context_storage") == "stream":
        from .installation import verify_frozen_source

        executor = TigerGraphExecutor()
        verify_frozen_source(executor, manifest)
        return StreamingContextSource(executor, per_relation=per_relation)
    return ContextStore(dataset / "contexts.sqlite", manifest["source"], per_relation=per_relation)
