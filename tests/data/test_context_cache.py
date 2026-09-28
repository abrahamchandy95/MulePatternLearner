"""The cache tiers of a context source: the in-memory LRU and the disk tier."""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.contract.feature_groups import CORE_GROUPS, FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.server import CONTEXT_QUERY
from mule_pattern_learner.data import context_cache
from mule_pattern_learner.data.context_cache import ContextCache, DiskTier, MemoryTier
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.testing.builders import CORE_PLAN, SMALL_SAMPLER, neighbourhood, root
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

# Roots whose cutoff leaves room for the neighbourhood's hourly payments.
KEYS = [root(i, cutoff_ms=1000 * 3_600_000) for i in range(6)]
# The one key the fake graph rejects.
REJECTED = KEYS[4]


def test_the_memory_tier_marks_a_fetch_in_key_order_before_it_evicts() -> None:
    a, b, c, d = (root(i) for i in range(4))
    tier = MemoryTier(2)
    tier.put(1, [a, b], {a: {"id": "a"}, b: {"id": "b"}})
    assert tier.get(1, [a, b, c]) == {a: {"id": "a"}, b: {"id": "b"}}
    # get leaves recency alone; a fetch of c, a (held) and d marks all three in that
    # order, and only then drops the least recent: b, then c.
    tier.get(1, [b])
    tier.put(1, [c, a, d], {c: {"id": "c"}, d: {"id": "d"}})
    assert list(tier.rows) == [(1, a), (1, d)]
    # Hops are kept apart, and a tier of capacity 0 holds nothing.
    assert tier.get(2, [a]) == {}
    empty = MemoryTier(0)
    empty.put(1, [a], {a: {"id": "a"}})
    assert not empty.rows
    tier.close()
    assert not tier.rows


def graph() -> FakeTigerGraph:
    return FakeTigerGraph(factory=neighbourhood, statuses={REJECTED: "missing_entity"})


def cache_in(directory: Path, **changes: Any) -> ContextCache:
    return replace(ContextCache(directory, "dataset", "source"), **changes)


def source(executor: FakeTigerGraph, cache: ContextCache | None, **options: Any) -> ContextSource:
    return ContextSource(
        TigerGraphContextFetcher(executor),
        plan=options.pop("plan", CORE_PLAN),
        sampler=options.pop("sampler", SMALL_SAMPLER),
        request_batch_size=2,
        cache=cache,
        **options,
    )


def read_all(contexts: ContextSource) -> list[list[dict[str, Any] | None]]:
    """Every key at both hops, the hop-1 keys twice (the second time from memory)."""
    rows = [contexts.fetch(KEYS), contexts.fetch(KEYS[:3], hop=2), contexts.fetch(KEYS)]
    contexts.close()
    return rows


def entry(tier: DiskTier, key: ContextKey, hop: int = 1) -> Path:
    return tier.path(tier.name(hop, key))


def test_a_second_source_reads_the_rows_of_the_first_from_disk(tmp_path: Path) -> None:
    cache = cache_in(tmp_path / "contexts")
    cold_graph, warm_graph = graph(), graph()
    cold, warm = source(cold_graph, cache), source(warm_graph, cache)
    cold_rows = read_all(cold)
    # The first source requested every context once and wrote each to disk.
    assert len(cold_graph.requested) == len(KEYS) + 3
    assert (cold.counts.memory_hits, cold.counts.disk_hits) == (len(KEYS), 0)
    assert cold.disk is not None
    assert len(list(cache.directory.glob("??/*.json.gz"))) == len(KEYS) + 3
    # The second requests none of them: it serves the same rows, rejections included.
    warm_rows = read_all(warm)
    assert warm_rows == cold_rows and warm_graph.names() == []
    assert (warm.database_calls, warm.counts.disk_hits, warm.counts.memory_hits) == (
        0,
        len(KEYS) + 3,
        len(KEYS),
    )
    assert warm.rejections == cold.rejections == {"missing_entity": 2}
    assert warm.counts.requested == cold.counts.requested
    assert warm.counts.seen == cold.counts.seen
    # An entry is TigerGraph's own row: its request position, and the Fourier vectors of
    # the first request's spot check, which the source does not serve.
    first = json.loads(gzip.decompress(entry(cold.disk, KEYS[0]).read_bytes()))
    assert first["entry"] == cold.disk.name(1, KEYS[0])
    assert first["row"]["request_index"] == 0 and first["row"]["age_encoding"]
    served = cold_rows[0][0]
    assert served is not None and served["age_encoding"] == {} and "request_index" not in served


def test_an_entry_is_named_by_everything_its_row_depends_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = cache_in(tmp_path)
    tier = DiskTier(cache, plan=CORE_PLAN, sampler=SMALL_SAMPLER)
    name = tier.name(1, KEYS[0])
    assert tier.path(name) == tmp_path / name[:2] / f"{name}.json.gz"
    others = [
        tier.name(2, KEYS[0]),
        tier.name(1, KEYS[1]),
        tier.name(1, replace(KEYS[0], visibility_phase=2)),
        # The groups the source requests, and the candidate pool of the hop.
        DiskTier(cache, plan=FeaturePlan((*CORE_GROUPS, "recency")), sampler=SMALL_SAMPLER).name(
            1, KEYS[0]
        ),
        DiskTier(
            cache,
            plan=CORE_PLAN,
            sampler=replace(SMALL_SAMPLER, roots=replace(SMALL_SAMPLER.roots, max_history=1024)),
        ).name(1, KEYS[0]),
        # The dataset and the frozen source the cache belongs to.
        DiskTier(
            cache_in(tmp_path, dataset_id="other"), plan=CORE_PLAN, sampler=SMALL_SAMPLER
        ).name(1, KEYS[0]),
        DiskTier(cache_in(tmp_path, source="other"), plan=CORE_PLAN, sampler=SMALL_SAMPLER).name(
            1, KEYS[0]
        ),
    ]
    assert name not in others and len(set(others)) == len(others)
    # The children's pool does not name a root's entry.
    children = replace(SMALL_SAMPLER, children=replace(SMALL_SAMPLER.children, max_history=1024))
    assert DiskTier(cache, plan=CORE_PLAN, sampler=children).name(1, KEYS[0]) == name
    # A source that requests another summary group reads none of these roots, but shares
    # the children, whose flags are the same: TGAT reads no summary group of a child.
    read_all(source(graph(), cache))
    other = graph()
    read_all(source(other, cache, plan=FeaturePlan((*CORE_GROUPS, "recency"))))
    assert other.requested == KEYS
    with pytest.raises(ValueError, match="at least one entry"):
        cache_in(tmp_path, capacity=0)
    # The feature contract names every entry.
    monkeypatch.setattr(context_cache, "CONTEXT_CONTRACT", "another_contract")
    assert tier.name(1, KEYS[0]) != name


def test_a_corrupted_or_foreign_entry_is_refused_and_requested_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = cache_in(tmp_path)
    expected = read_all(source(graph(), cache))
    tier = DiskTier(cache, plan=CORE_PLAN, sampler=SMALL_SAMPLER)
    truncated, garbage, foreign, other_key, empty = (entry(tier, key) for key in KEYS[:5])
    truncated.write_bytes(truncated.read_bytes()[:-9])
    garbage.write_bytes(b"not gzip at all")
    # The entry of another context under this one's name.
    foreign.write_bytes(entry(tier, KEYS[5]).read_bytes())
    # An entry that records its own name but holds the row of another key.
    moved = json.loads(gzip.decompress(entry(tier, KEYS[5]).read_bytes()))
    moved["entry"] = tier.name(1, KEYS[3])
    other_key.write_bytes(gzip.compress(json.dumps(moved).encode()))
    # The rejected key's entry, without its status row.
    empty.write_bytes(gzip.compress(json.dumps({"entry": tier.name(1, REJECTED)}).encode()))
    capsys.readouterr()
    again = graph()
    contexts = source(again, cache)
    assert read_all(contexts) == expected
    # The five refused contexts were requested again, once each, and nothing else was.
    assert sorted(again.requested, key=str) == sorted(KEYS[:5], key=str)
    assert contexts.disk is not None and contexts.disk.refused == 5
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["warning"] for line in printed] == ["context_cache_refused"] * 5
    assert "cannot be read" in printed[0]["message"]
    # Their requests replaced them, so the next source reads every context from disk.
    last = graph()
    assert read_all(source(last, cache)) == expected and last.names() == []


def test_the_disk_tier_evicts_its_least_recently_used_entries_beyond_its_capacity(
    tmp_path: Path,
) -> None:
    cache = cache_in(tmp_path, capacity=4)
    tier = DiskTier(cache, plan=CORE_PLAN, sampler=SMALL_SAMPLER)
    rows = {key: {"status": "missing_entity"} for key in KEYS}
    tier.put(1, KEYS[:4], rows)
    for age, key in enumerate(KEYS[:4]):
        os.utime(entry(tier, key), ns=(10**9 * (age + 1), 10**9 * (age + 1)))
    # A hit makes the oldest entry the most recent one.
    assert tier.get(1, KEYS[:1]) == {KEYS[0]: rows[KEYS[0]]}
    # A fifth entry is one too many: the tier keeps 90% of 4, the three most recent.
    tier.put(1, KEYS[4:5], rows)
    kept = {key for key in KEYS[:5] if entry(tier, key).exists()}
    assert kept == {KEYS[0], KEYS[3], KEYS[4]} and tier.evicted == 2
    # Another tier on the directory counts the entries already there.
    later = DiskTier(cache, plan=CORE_PLAN, sampler=SMALL_SAMPLER)
    later.put(1, KEYS[5:], rows)
    assert later.evicted == 0
    later.put(1, KEYS[1:2], rows)
    assert later.evicted == 2


def test_a_cache_that_cannot_be_written_is_warned_about_once_and_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("")
    executor = graph()
    contexts = source(executor, cache_in(blocked))
    rows = read_all(contexts)
    assert rows[0][0] is not None and rows[0][4] is None
    assert executor.names().count(CONTEXT_QUERY) == len(executor.calls) > 1
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["warning"] for line in printed] == ["context_cache_unwritable"]
    assert contexts.disk is not None and not contexts.disk.writable
    assert contexts.disk.refused == 0
