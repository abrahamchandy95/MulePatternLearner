"""The cache tiers of a ContextSource: its in-memory LRU and its disk tier.

A tier holds context rows by hop and ContextKey (ContextTier). The source asks its
tiers before TigerGraph and gives them the rows it obtained, so every tier has the same
two operations: get the rows it holds of some keys, and put the rows a fetch obtained.
MemoryTier is the bounded LRU of the rows the source serves. DiskTier keeps the rows
TigerGraph returned, as it returned them, in a prepared dataset's context cache
(ContextCache, under data/<dataset id>/contexts/), so a later run of the dataset reads
them from disk instead of requesting them again.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
import contextlib
from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import threading
from typing import Any, Protocol
import zlib

from ..artifacts import atomic_write
from ..contract.bounds import CONTEXT_CACHE_ENTRIES
from ..contract.feature_groups import FeaturePlan
from ..contract.fingerprints import fingerprint
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..contract.server import CONTEXT_CONTRACT
from ..paths import DatasetPaths
from ..runtime.progress import warn
from .manifest import recorded_dataset_id, source_fingerprint

# The file name ending of a disk tier's entries.
ENTRY_SUFFIX = ".json.gz"
# A disk tier over its capacity removes its least recently used entries until this share
# of the capacity remains, so that it does not scan its directory on every write.
EVICTED_DOWN_TO = 0.9


class ContextTier(Protocol):
    """One cache tier of a ContextSource, keyed by (hop, ContextKey).

    get returns the rows the tier holds of keys, and put hands it the rows one fetch
    obtained (``rows``) with every key the fetch asked for, in the fetch's order
    (``keys``). close frees what the tier holds in memory. The source calls a tier from
    one thread at a time unless the tier says otherwise.
    """

    def get(self, hop: int, keys: Iterable[ContextKey]) -> dict[ContextKey, dict[str, Any]]: ...
    def put(
        self, hop: int, keys: Sequence[ContextKey], rows: Mapping[ContextKey, dict[str, Any]]
    ) -> None: ...
    def close(self) -> None: ...


class MemoryTier:
    """The bounded in-memory LRU of the rows a ContextSource serves.

    It is keyed by (hop, ContextKey), because roots and children use different
    candidate pools and feature flags. put keeps the new rows, then marks every key of
    the fetch it holds as the most recent, in the fetch's key order, so recency does
    not depend on thread timing; only then are the least recent rows beyond
    ``capacity`` dropped. A capacity of 0 keeps nothing.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.rows: OrderedDict[tuple[int, ContextKey], dict[str, Any]] = OrderedDict()

    def get(self, hop: int, keys: Iterable[ContextKey]) -> dict[ContextKey, dict[str, Any]]:
        """The rows held of keys; their recency changes only with put."""
        found: dict[ContextKey, dict[str, Any]] = {}
        for key in keys:
            row = self.rows.get((hop, key))
            if row is not None:
                found[key] = row
        return found

    def put(
        self, hop: int, keys: Sequence[ContextKey], rows: Mapping[ContextKey, dict[str, Any]]
    ) -> None:
        for key in keys:
            if key in rows and self.capacity:
                self.rows[(hop, key)] = rows[key]
            if (hop, key) in self.rows:
                self.rows.move_to_end((hop, key))
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def close(self) -> None:
        self.rows.clear()


@dataclass(frozen=True)
class ContextCache:
    """A prepared dataset's context cache on disk: its directory and what names its entries.

    ``dataset_id`` is the dataset's id, and ``source`` the fingerprint of the frozen
    source its manifest records (data.manifest.source_fingerprint): the vertex counts and
    the scope that tigergraph.provenance.verify_frozen_source checks against the live
    graph before a source opens the cache. Both name every entry, so that check is what
    invalidates the cache: once the graph's counts change, a run stops before it reads
    an entry, and a dataset prepared again from the changed graph never finds the old
    graph's entries. ``capacity`` is the most entries the cache keeps.
    """

    directory: Path
    dataset_id: str
    source: str
    capacity: int = CONTEXT_CACHE_ENTRIES

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"A context cache keeps at least one entry, not {self.capacity}")

    @classmethod
    def of(cls, dataset: DatasetPaths, manifest: dict[str, Any]) -> ContextCache:
        """The context cache of a prepared dataset, in its contexts/ directory."""
        return cls(dataset.contexts, recorded_dataset_id(manifest), source_fingerprint(manifest))


def entry_row(data: bytes, name: str, key: ContextKey) -> dict[str, Any]:
    """The row an entry's bytes hold; a ValueError says why the entry is refused.

    An entry is refused when it cannot be read (gzip checks its own CRC), when the name
    it records is another one (the entry of another context), or when it holds no row
    with a status, or an ok row of another key or feature contract.
    """
    try:
        entry = json.loads(gzip.decompress(data))
    except (OSError, EOFError, ValueError, zlib.error) as error:
        raise ValueError(f"it cannot be read ({type(error).__name__})") from None
    if not isinstance(entry, dict) or entry.get("entry") != name:
        raise ValueError("it records another context")
    row = entry.get("row")
    if not isinstance(row, dict) or not isinstance(row.get("status"), str):
        raise ValueError("it holds no context row")
    if row["status"] == "ok" and (
        any(row.get(field) != value for field, value in asdict(key).items())
        or row.get("contract_version") != CONTEXT_CONTRACT
    ):
        raise ValueError("its row is of another context or feature contract")
    return row


class DiskTier:
    """The disk tier of a ContextSource: TigerGraph's rows, compressed, one file per context.

    An entry is named by the sha256 of what its row depends on: the hop and ContextKey,
    the group flags and candidate pool the source requests at that hop, the
    CONTEXT_CONTRACT and the cache's dataset id and frozen source. It holds the row as
    the ContextFetcher returned it (its request position and any Fourier vectors of a
    spot check included) beside its own name, as gzip-compressed JSON in
    <directory>/<first two hex digits>/<name>.json.gz, written atomically. get refuses
    an entry that entry_row refuses, with a warning: the source requests that context
    again and put replaces the entry. Beyond the cache's capacity, the least recently
    used entries go (a hit refreshes an entry's time) until EVICTED_DOWN_TO of the
    capacity remain. A directory that cannot be written is warned about once, then only
    read. The source's request workers call get and put at once, so the counts take a
    lock: ``refused`` entries and ``evicted`` ones.
    """

    def __init__(self, cache: ContextCache, *, plan: FeaturePlan, sampler: SamplerPlan) -> None:
        self.cache, self.plan, self.sampler = cache, plan, sampler
        self._requests = {
            hop: {"flags": plan.query_flags(hop), "pools": fingerprint(sampler.query_params(hop))}
            for hop in (1, 2)
        }
        self._lock = threading.Lock()
        # The entries on disk: counted when the tier first writes one, then kept up.
        self._entries: int | None = None
        self.writable = True
        self.refused = self.evicted = 0

    def name(self, hop: int, key: ContextKey) -> str:
        """The name of a context's entry: the fingerprint of what its row depends on."""
        identity = {
            "hop": hop,
            "key": asdict(key),
            **self._requests[hop],
            "contract": CONTEXT_CONTRACT,
            "dataset_id": self.cache.dataset_id,
            "source": self.cache.source,
        }
        return fingerprint(identity)

    def path(self, name: str) -> Path:
        return self.cache.directory / name[:2] / (name + ENTRY_SUFFIX)

    def get(self, hop: int, keys: Iterable[ContextKey]) -> dict[ContextKey, dict[str, Any]]:
        found: dict[ContextKey, dict[str, Any]] = {}
        for key in keys:
            name = self.name(hop, key)
            path = self.path(name)
            try:
                found[key] = entry_row(path.read_bytes(), name, key)
            except (FileNotFoundError, NotADirectoryError):
                continue  # no entry
            except (OSError, ValueError) as error:
                self._refuse(path, str(error))
                continue
            with contextlib.suppress(OSError):
                os.utime(path)
        return found

    def put(
        self, hop: int, keys: Sequence[ContextKey], rows: Mapping[ContextKey, dict[str, Any]]
    ) -> None:
        for key in keys:
            if key not in rows or not self.writable:
                continue
            name = self.name(hop, key)
            path = self.path(name)
            entry = json.dumps({"entry": name, "row": rows[key]}, separators=(",", ":"))
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                new = not path.exists()
                with atomic_write(path) as pending:
                    pending.write_bytes(gzip.compress(entry.encode(), mtime=0))
            except FileNotFoundError:
                continue  # another writer of the same entry replaced it first
            except OSError as error:
                self._unwritable(error)
                return
            if new:
                self._added()

    def close(self) -> None:
        """Nothing to free: the tier holds no rows in memory."""

    def _files(self) -> Iterator[Path]:
        return self.cache.directory.glob(f"??/*{ENTRY_SUFFIX}")

    def _added(self) -> None:
        with self._lock:
            if self._entries is None:
                self._entries = sum(1 for _ in self._files())
            else:
                self._entries += 1
            if self._entries > self.cache.capacity:
                self._evict()

    def _evict(self) -> None:
        """Remove the least recently used entries until EVICTED_DOWN_TO remain."""
        dated: list[tuple[int, str, Path]] = []
        for path in self._files():
            with contextlib.suppress(FileNotFoundError):
                dated.append((path.stat().st_mtime_ns, path.name, path))
        dated.sort()
        keep = int(self.cache.capacity * EVICTED_DOWN_TO)
        for _, _, path in dated[: max(len(dated) - keep, 0)]:
            path.unlink(missing_ok=True)
            self.evicted += 1
        self._entries = min(len(dated), keep)

    def _refuse(self, path: Path, reason: str) -> None:
        with self._lock:
            self.refused += 1
        warn(
            "context_cache_refused",
            f"Refused the cached context {path.name}: {reason}. It is requested from "
            "TigerGraph again.",
        )

    def _unwritable(self, error: OSError) -> None:
        with self._lock:
            if not self.writable:
                return
            self.writable = False
        warn(
            "context_cache_unwritable",
            f"Cannot write the context cache in {self.cache.directory} ({error}); contexts "
            "are requested from TigerGraph without being kept on disk.",
        )
