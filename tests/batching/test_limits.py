"""Batch limits reject oversized candidate pools before any fetch."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.batching.limits import BatchCapacityError, BatchLimits
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.testing.builders import roots
from mule_pattern_learner.testing.fake_graph import FakeStore


def test_batch_limits_bound_candidate_pools_before_fetching() -> None:
    wide = SamplerPlan(roots=PoolPlan(32, 16, 16, 8), relation_fanouts=(8, 4))
    store = FakeStore(wide)
    with pytest.raises(BatchCapacityError, match="Candidate pools"):
        build_batch(store, roots(120), fanouts=(16, 4), sampler=wide, plan=FeaturePlan())
    assert not store.calls
    BatchLimits().validate(120, (16, 4), FeaturePlan(), SamplerPlan())
    BatchLimits().validate(120, (16, 4), FeaturePlan(), replace(wide, children=PoolPlan()))
