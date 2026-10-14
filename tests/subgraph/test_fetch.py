"""Fetching subgraphs from the reference graph's standard world (testing.world) by as-of
date, through the disk cache.

T is Monday 2024-07-01 and LATER four weeks on, after New's account opened.
"""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, SubgraphConfig
from mule_pattern_learner.subgraph.build import build
from mule_pattern_learner.subgraph.fetch import SubgraphCache, cache_key, fetch
from mule_pattern_learner.testing.reference_graph import ReferenceGraph
from mule_pattern_learner.testing.world import LATER, T, standard_world
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    SUBGRAPH_VERSION,
    WRITE_SPLIT,
    Phase,
    SubgraphBatch,
    SubgraphRequest,
    parse_subgraphs,
    split_params,
    subgraph_params,
)

CONFIG = DEFAULT_CONFIG.subgraph
TRAINING = Phase.TRAINING
# The hold-out seed the standard world's split stands for: the world writes it by hand.
HOLDOUT = DEFAULT_CONFIG.data.seed


class _Recording:
    """The reference graph, keeping the as-of time and seeds of each fetch_subgraphs."""

    def __init__(self, graph: ReferenceGraph) -> None:
        self.graph = graph
        self.requests: list[tuple[int, list[str]]] = []

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        if name == FETCH_SUBGRAPHS:
            self.requests.append((params["as_of_ms"], list(params["seed_ids"])))
        return self.graph.run(name, params)


@pytest.fixture
def graph() -> _Recording:
    return _Recording(standard_world().graph())


def _direct(
    graph: _Recording,
    seeds: list[str],
    t: int,
    phase: Phase = TRAINING,
    config: SubgraphConfig = CONFIG,
) -> SubgraphBatch:
    """The seeds' subgraphs in one request of their own, outside fetch and its cache."""
    request = SubgraphRequest(tuple(seeds), t, phase)
    rows = graph.graph.run(FETCH_SUBGRAPHS, subgraph_params(request, config))
    return parse_subgraphs(rows, request)


def _accounts(batch: SubgraphBatch) -> list[str]:
    """The accounts of the batch's only subgraph, sorted."""
    [subgraph] = batch.subgraphs
    return sorted(node.id for node in subgraph.accounts)


def _stored(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def test_seeds_are_fetched_by_as_of_date_batch_size_to_a_request(graph: _Recording) -> None:
    seeds = [
        ("A_MIA", T),
        ("A_NEW", LATER),
        ("A_ANN", T),
        ("A_QUY", T),
        ("A_ANN", T),
        ("A_ROY", LATER),
    ]
    batches = fetch(graph, seeds, TRAINING, CONFIG, batch_size=2)
    assert graph.requests == [
        (T, ["A_MIA", "A_ANN"]),
        (T, ["A_QUY"]),
        (LATER, ["A_NEW", "A_ROY"]),
    ]
    assert list(batches) == [T, LATER]
    assert batches[T] == _direct(graph, ["A_MIA", "A_ANN", "A_QUY"], T)
    assert batches[LATER] == _direct(graph, ["A_NEW", "A_ROY"], LATER)
    # The batches of both dates make one HeteroData, each seed with its as-of time.
    data = build(list(batches.values()))
    assert data.as_of_ms.tolist() == [T, T, T, LATER, LATER]


def test_a_cached_date_is_read_back_without_a_request(graph: _Recording, tmp_path: Path) -> None:
    cache = SubgraphCache(tmp_path, "dev", HOLDOUT)
    seeds = [("A_MIA", T), ("A_ANN", T)]
    first = fetch(graph, seeds, TRAINING, CONFIG, cache=cache)
    assert graph.requests == [(T, ["A_MIA", "A_ANN"])]
    path = tmp_path / "dev" / "training" / "2024-07-01.json.gz"
    assert cache.path(T, TRAINING) == path
    assert sorted(_stored(path)["subgraphs"]) == ["A_ANN", "A_MIA"]
    again = fetch(graph, seeds[::-1], TRAINING, CONFIG, cache=cache)
    assert len(graph.requests) == 1
    assert again[T].subgraphs == first[T].subgraphs[::-1]


def test_only_the_seeds_the_cache_lacks_are_fetched(graph: _Recording, tmp_path: Path) -> None:
    cache = SubgraphCache(tmp_path, "dev", HOLDOUT)
    fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, cache=cache)
    batches = fetch(graph, [("A_ANN", T), ("A_MIA", T)], TRAINING, CONFIG, cache=cache)
    assert graph.requests == [(T, ["A_MIA"]), (T, ["A_ANN"])]
    assert batches[T] == _direct(graph, ["A_ANN", "A_MIA"], T)
    assert sorted(_stored(cache.path(T, TRAINING))["subgraphs"]) == ["A_ANN", "A_MIA"]


def test_a_changed_query_version_misses_the_cache(graph: _Recording, tmp_path: Path) -> None:
    """A file written by another version of fetch_subgraphs is fetched again and replaced."""
    cache = SubgraphCache(tmp_path, "dev", HOLDOUT)
    fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, cache=cache)
    path = cache.path(T, TRAINING)
    stored = _stored(path)
    stored["key"]["version"] = "subgraphs-000000000000"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(stored, file)
    assert cache.load(T, TRAINING, CONFIG) == {}
    batches = fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, cache=cache)
    assert graph.requests == [(T, ["A_MIA"]), (T, ["A_MIA"])]
    assert batches[T] == _direct(graph, ["A_MIA"], T)
    assert _stored(path)["key"] == {
        "version": SUBGRAPH_VERSION,
        "settings": stored["key"]["settings"],
        "holdout_seed": HOLDOUT,
    }


def test_changed_settings_miss_the_cache(graph: _Recording, tmp_path: Path) -> None:
    """Under caps of one payment each way, Ann's subgraph keeps fewer of her payments."""
    cache = SubgraphCache(tmp_path, "dev", HOLDOUT)
    fetch(graph, [("A_ANN", T)], TRAINING, CONFIG, cache=cache)
    fewer = replace(CONFIG, seed_payments=1, other_payments=1)
    batches = fetch(graph, [("A_ANN", T)], TRAINING, fewer, cache=cache)
    assert graph.requests == [(T, ["A_ANN"]), (T, ["A_ANN"])]
    assert batches[T] == _direct(graph, ["A_ANN"], T, config=fewer)
    assert batches[T] != _direct(graph, ["A_ANN"], T)


def test_a_split_written_with_another_seed_misses_the_cache(
    graph: _Recording, tmp_path: Path
) -> None:
    """Holding out Max hides him, and the cash-out account reached only through him, from
    Mia's training subgraph: the subgraphs cached under the old split are not read."""
    before = fetch(
        graph, [("A_MIA", T)], TRAINING, CONFIG, cache=SubgraphCache(tmp_path, "dev", HOLDOUT)
    )
    assert _accounts(before[T]) == ["A_MAX", "A_MIA", "A_QUY", "A_VIC", "X_CASH"]
    [written] = graph.graph.run(WRITE_SPLIT, split_params(standard_world().splits() | {"P_MAX": 1}))
    assert written["status"] == "ok"
    cache = SubgraphCache(tmp_path, "dev", HOLDOUT + 1)
    assert cache.load(T, TRAINING, CONFIG) == {}
    batches = fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, cache=cache)
    assert graph.requests == [(T, ["A_MIA"]), (T, ["A_MIA"])]
    assert batches[T] == _direct(graph, ["A_MIA"], T)
    assert _accounts(batches[T]) == ["A_MIA", "A_QUY", "A_VIC"]
    assert _stored(cache.path(T, TRAINING))["key"] == cache_key(CONFIG, HOLDOUT + 1)


def test_each_world_and_phase_has_files_of_its_own(graph: _Recording, tmp_path: Path) -> None:
    dev = SubgraphCache(tmp_path, "dev", HOLDOUT)
    other = SubgraphCache(tmp_path, "other-1", HOLDOUT)
    fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, cache=dev)
    fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, cache=other)
    test = fetch(graph, [("A_MIA", T)], Phase.TEST, CONFIG, cache=dev)
    assert len(graph.requests) == 3
    assert test[T] == _direct(graph, ["A_MIA"], T, Phase.TEST)
    assert dev.path(T, Phase.TEST) == tmp_path / "dev" / "test" / "2024-07-01.json.gz"
    assert all(path.exists() for path in (dev.path(T, Phase.TEST), other.path(T, TRAINING)))


def test_a_refused_request_stops_the_fetch(graph: _Recording, tmp_path: Path) -> None:
    cache = SubgraphCache(tmp_path, "dev", HOLDOUT)
    with pytest.raises(ValueError, match="unknown_seed"):
        fetch(graph, [("A_MIA", T), ("A_NOBODY", T)], TRAINING, CONFIG, cache=cache)
    assert not cache.path(T, TRAINING).exists()
    with pytest.raises(ValueError, match="batch_size"):
        fetch(graph, [("A_MIA", T)], TRAINING, CONFIG, batch_size=0)


def test_the_cache_refuses_a_world_that_is_no_name_and_a_time_inside_a_day(
    tmp_path: Path,
) -> None:
    for world in ("", "../dev", "dev/one", ".hidden"):
        with pytest.raises(ValueError, match="world"):
            SubgraphCache(tmp_path, world, HOLDOUT)
    with pytest.raises(ValueError, match="00:00 UTC"):
        SubgraphCache(tmp_path, "dev", HOLDOUT).path(T + 1, TRAINING)
