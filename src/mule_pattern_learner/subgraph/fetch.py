"""Subgraphs fetched by as-of date through the executor, with a disk cache (docs/design.md,
"The subgraph query", and "Caching" under "Training").

Seeds that share an as-of time are fetched together, at most batch_size to a request, so
TigerGraph computes each identity node's features once per request. The cache keeps one
file per (world, phase, as-of date) holding each seed's subgraph as fetch_subgraphs printed
it, under the key it was fetched with: the query's version and the subgraph settings. A
file under another key holds nothing the current code may read, and the next fetch of its
date replaces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, timedelta
import gzip
import json
from pathlib import Path
import re
from typing import Any

from ..config import SubgraphConfig
from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import (
    FETCH_SUBGRAPHS,
    SUBGRAPH_VERSION,
    Phase,
    Subgraph,
    SubgraphBatch,
    SubgraphRequest,
    parse_subgraphs,
    subgraph_params,
)

# Seeds per fetch_subgraphs request, a guess until it is measured on a live graph.
BATCH_SIZE = 32
DAY_MS = 86_400_000
_EPOCH = date(1970, 1, 1)
# A world's name, which is one directory of the cache.
_WORLD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def cache_key(config: SubgraphConfig) -> dict[str, Any]:
    """What a cached subgraph depends on: the query's version and the subgraph settings."""
    return {"version": SUBGRAPH_VERSION, "settings": asdict(config)}


class SubgraphCache:
    """One world's printed subgraphs by seed, in one file per (phase, as-of date) under
    root/world, each file with the key its subgraphs were fetched with."""

    def __init__(self, root: Path, world: str) -> None:
        if not _WORLD.fullmatch(world):
            raise ValueError(f"A world is named by letters, digits, '.', '_' and '-': {world!r}")
        self.directory = root / world

    def path(self, as_of_ms: int, phase: Phase) -> Path:
        days, rest = divmod(as_of_ms, DAY_MS)
        if rest:
            raise ValueError(f"The cache holds as-of dates at 00:00 UTC, not {as_of_ms}")
        day = _EPOCH + timedelta(days=days)
        return self.directory / phase.value / f"{day.isoformat()}.json.gz"

    def load(self, as_of_ms: int, phase: Phase, key: Mapping[str, Any]) -> dict[str, Any]:
        """The printed subgraph of each seed the file holds under key, none under another."""
        path = self.path(as_of_ms, phase)
        if not path.exists():
            return {}
        with gzip.open(path, "rt", encoding="utf-8") as file:
            stored = json.load(file)
        return dict(stored["subgraphs"]) if stored["key"] == key else {}

    def save(
        self, as_of_ms: int, phase: Phase, key: Mapping[str, Any], subgraphs: Mapping[str, Any]
    ) -> None:
        """Replaces the file with subgraphs under key, so no reader sees half of it."""
        path = self.path(as_of_ms, phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial")
        with gzip.open(partial, "wt", encoding="utf-8") as file:
            json.dump({"key": key, "subgraphs": subgraphs}, file)
        partial.replace(path)


def fetch(
    executor: QueryExecutor,
    seeds: Iterable[tuple[str, int]],
    phase: Phase,
    config: SubgraphConfig,
    *,
    cache: SubgraphCache | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[int, SubgraphBatch]:
    """The subgraph of each (account, as-of time) of seeds, seen in phase: one batch per
    as-of time, in time order, holding its accounts in the order seeds first gives them.

    A seed given twice is fetched once. The seeds of an as-of time that the cache does not
    hold are fetched batch_size to a request and then cached with the others of the date.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, not {batch_size}")
    dates: dict[int, dict[str, None]] = {}
    for account_id, as_of_ms in seeds:
        dates.setdefault(as_of_ms, {})[account_id] = None
    key = cache_key(config)
    batches: dict[int, SubgraphBatch] = {}
    for as_of_ms in sorted(dates):
        accounts = list(dates[as_of_ms])
        printed = {} if cache is None else cache.load(as_of_ms, phase, key)
        found = _cached(printed, [seed for seed in accounts if seed in printed], as_of_ms, phase)
        missing = [seed for seed in accounts if seed not in printed]
        for start in range(0, len(missing), batch_size):
            request = SubgraphRequest(tuple(missing[start : start + batch_size]), as_of_ms, phase)
            rows = executor.run(FETCH_SUBGRAPHS, subgraph_params(request, config))
            found.update((sub.seed, sub) for sub in parse_subgraphs(rows, request).subgraphs)
            printed.update(zip(request.seed_ids, rows[1:], strict=True))
        if cache is not None and missing:
            cache.save(as_of_ms, phase, key, printed)
        subgraphs = tuple(found[seed] for seed in accounts)
        batches[as_of_ms] = SubgraphBatch(SUBGRAPH_VERSION, as_of_ms, phase, subgraphs)
    return batches


def _cached(
    printed: Mapping[str, Any], seeds: Sequence[str], as_of_ms: int, phase: Phase
) -> dict[str, Subgraph]:
    """The cached subgraphs of seeds, parsed and checked as a response to them would be."""
    if not seeds:
        return {}
    request = SubgraphRequest(tuple(seeds), as_of_ms, phase)
    head = {
        "status": "ok",
        "version": SUBGRAPH_VERSION,
        "as_of_ms": as_of_ms,
        "held_out_visible": phase.held_out_visible,
    }
    batch = parse_subgraphs([head, *(printed[seed] for seed in seeds)], request)
    return {sub.seed: sub for sub in batch.subgraphs}
