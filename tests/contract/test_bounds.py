"""The configuration, the sampler plan and the context source accept exactly the bounds."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mule_pattern_learner.batching.limits import BatchLimits
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.bounds import (
    BATCH_ROOTS,
    FANOUT,
    POOL,
    QUERY_CONCURRENCY,
    REQUEST_KEYS,
    SEED_LIMIT,
    Bound,
)
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher


def edges(bound: Bound) -> tuple[list[int], list[int]]:
    """The values at each end of a bound, and the values just outside it."""
    return [bound.low, bound.high], [bound.low - 1, bound.high + 1]


@pytest.mark.parametrize("name", POOL)
def test_pool_settings_accept_exactly_their_bound(name: str) -> None:
    inside, outside = edges(POOL[name])
    for value in inside:
        assert getattr(PoolPlan(**{name: value}), name) == value
        DEFAULT_CONFIG.with_changes(
            {"sampler": {"roots": {name: value}, "children": {name: value}}}
        )
    for value in outside:
        with pytest.raises(ValueError, match=name):
            PoolPlan(**{name: value})
        with pytest.raises(ValueError, match=name):
            DEFAULT_CONFIG.with_changes({"sampler": {"roots": {name: value}}})


# The changes that set one bounded setting to a value, and the setting's bound.
SETTINGS: dict[str, tuple[Callable[[int], dict[str, Any]], Bound]] = {
    "sampler.fanouts": (lambda v: {"sampler": {"fanouts": [v, v]}}, FANOUT),
    "sampler.relation_fanouts": (lambda v: {"sampler": {"relation_fanouts": [v, v]}}, FANOUT),
    "training.batch_size": (lambda v: {"training": {"batch_size": v}}, BATCH_ROOTS),
    "transport.request_batch_size": (
        lambda v: {"transport": {"request_batch_size": v}},
        REQUEST_KEYS,
    ),
    "transport.query_concurrency": (
        lambda v: {"transport": {"query_concurrency": v}},
        QUERY_CONCURRENCY,
    ),
    "dataset.seed_limits.train": (
        lambda v: {"dataset": {"seed_limits": {"train": v, "validation": 1, "test": 1}}},
        SEED_LIMIT,
    ),
}


@pytest.mark.parametrize("name", SETTINGS)
def test_configuration_accepts_exactly_the_bounds(name: str) -> None:
    changes, bound = SETTINGS[name]
    inside, outside = edges(bound)
    for value in inside:
        DEFAULT_CONFIG.with_changes(changes(value))
    for value in outside:
        with pytest.raises(
            ValueError, match=f"must be an integer in \\[{bound.low},{bound.high}\\]"
        ):
            DEFAULT_CONFIG.with_changes(changes(value))


def test_sampler_batches_and_context_source_read_the_same_bounds() -> None:
    for value in edges(FANOUT)[1]:
        with pytest.raises(ValueError, match="relation_fanouts"):
            SamplerPlan(relation_fanouts=(value, 1))
    assert BatchLimits().max_roots == BATCH_ROOTS.high
    for size in edges(REQUEST_KEYS)[1]:
        with pytest.raises(ValueError, match="query size"):
            ContextSource(
                TigerGraphContextFetcher(FakeTigerGraph()),
                request_batch_size=size,
                plan=FeaturePlan(),
                sampler=SamplerPlan(),
            )
    for workers in edges(QUERY_CONCURRENCY)[1]:
        with pytest.raises(ValueError, match="concurrency"):
            ContextSource(
                TigerGraphContextFetcher(FakeTigerGraph()),
                concurrency=workers,
                plan=FeaturePlan(),
                sampler=SamplerPlan(),
            )
