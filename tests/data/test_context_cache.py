"""The cache tiers of a context source: the in-memory LRU."""

from __future__ import annotations

from mule_pattern_learner.data.context_cache import MemoryTier
from mule_pattern_learner.testing.builders import root


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
