"""Batch limits refuse oversized pools and batches before any fetch; batch ids are dense."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.batching.limits import BatchCapacityError, BatchIndex, BatchLimits
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.testing.builders import roots
from mule_pattern_learner.testing.fake_graph import FakeStore, FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher


def test_batch_limits_bound_candidate_pools_before_fetching() -> None:
    wide = SamplerPlan(roots=PoolPlan(32, 16, 16, 8), relation_fanouts=(8, 4))
    store = FakeStore(wide)
    with pytest.raises(BatchCapacityError, match="Candidate pools"):
        build_batch(store, roots(120), fanouts=(16, 4), sampler=wide, plan=FeaturePlan())
    assert not store.calls
    BatchLimits().validate(120, (16, 4), FeaturePlan(), SamplerPlan())
    BatchLimits().validate(120, (16, 4), FeaturePlan(), replace(wide, children=PoolPlan()))


def test_batch_ids_are_dense_per_type_cutoff_and_scope_and_never_global() -> None:
    a = ContextKey("Account", "999999999999999999999999", 100, 1000, "experiment", 1)
    other_type = replace(a, node_type="Token")
    other_time = replace(a, cutoff_seq=90)
    other_scope = replace(a, scope_id="another")
    index = BatchIndex([a, a, other_type, other_time, other_scope])
    assert [index[k] for k in (a, other_type, other_time, other_scope)] == [0, 1, 2, 3]
    assert BatchIndex([other_type])[other_type] == 0
    with pytest.raises(BatchCapacityError):
        BatchIndex([a, other_type], capacity=1)


def test_budget_rejects_before_database_calls_and_tensor_allocation() -> None:
    executor = FakeTigerGraph({})
    source = ContextSource(
        TigerGraphContextFetcher(executor), plan=FeaturePlan(), sampler=SamplerPlan()
    )
    roots = [ContextKey("Account", str(i), 100, 1000) for i in range(129)]
    with pytest.raises(BatchCapacityError):
        build_batch(source, roots, fanouts=(8, 4), plan=FeaturePlan(), sampler=SamplerPlan())
    with pytest.raises(BatchCapacityError):
        build_batch(source, roots[:64], fanouts=(64, 64), plan=FeaturePlan(), sampler=SamplerPlan())
    assert executor.requested == []
