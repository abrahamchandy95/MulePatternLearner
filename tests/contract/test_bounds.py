"""The configuration, the sampler plan and the context source accept exactly the bounds."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mule_pattern_learner.batching.limits import BatchLimits
from mule_pattern_learner.config import validate_config
from mule_pattern_learner.contract.bounds import (
    BATCH_ROOTS,
    FANOUT,
    POOL,
    QUERY_CONCURRENCY,
    REQUEST_KEYS,
    SEED_LIMIT,
    Bound,
)
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.data.contexts import StreamingContextSource
from mule_pattern_learner.testing.fake_graph import FakeExecutor
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher


def edges(bound: Bound) -> tuple[list[int], list[int]]:
    """The values at each end of a bound, and the values just outside it."""
    return [bound.low, bound.high], [bound.low - 1, bound.high + 1]


@pytest.mark.parametrize("name", POOL)
def test_pool_settings_accept_exactly_their_bound(name: str) -> None:
    inside, outside = edges(POOL[name])
    for value in inside:
        assert getattr(PoolPlan(**{name: value}), name) == value
        validate_config({"sampler": {name: value, "children": {name: value}}})
    for value in outside:
        with pytest.raises(ValueError, match=name):
            PoolPlan(**{name: value})
        with pytest.raises(ValueError, match=name):
            validate_config({"sampler": {name: value}})


# A configuration that sets one bounded setting to a value, and the setting's bound.
SETTINGS: dict[str, tuple[Callable[[int], dict[str, Any]], Bound]] = {
    "fanouts": (lambda v: {"fanouts": [v, v]}, FANOUT),
    "relation_fanouts": (lambda v: {"sampler": {"relation_fanouts": [v, v]}}, FANOUT),
    "batch_size": (lambda v: {"batch_size": v}, BATCH_ROOTS),
    "request_batch_size": (lambda v: {"request_batch_size": v}, REQUEST_KEYS),
    "query_concurrency": (lambda v: {"query_concurrency": v}, QUERY_CONCURRENCY),
    "seed_limits": (
        lambda v: {"seed_limits": {"train": v, "validation": 1, "test": 1}},
        SEED_LIMIT,
    ),
}


@pytest.mark.parametrize("name", SETTINGS)
def test_configuration_accepts_exactly_the_bounds(name: str) -> None:
    config, bound = SETTINGS[name]
    inside, outside = edges(bound)
    for value in inside:
        validate_config(config(value))
    for value in outside:
        with pytest.raises(ValueError, match="Invalid configuration"):
            validate_config(config(value))


def test_sampler_batches_and_context_source_read_the_same_bounds() -> None:
    for value in edges(FANOUT)[1]:
        with pytest.raises(ValueError, match="relation_fanouts"):
            SamplerPlan(relation_fanouts=(value, 1))
    assert BatchLimits().max_roots == BATCH_ROOTS.high
    for size in edges(REQUEST_KEYS)[1]:
        with pytest.raises(ValueError, match="query size"):
            StreamingContextSource(
                TigerGraphContextFetcher(FakeExecutor()), request_batch_size=size
            )
    for workers in edges(QUERY_CONCURRENCY)[1]:
        with pytest.raises(ValueError, match="concurrency"):
            StreamingContextSource(TigerGraphContextFetcher(FakeExecutor()), concurrency=workers)
