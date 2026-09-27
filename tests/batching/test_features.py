"""Feature matrices refuse messages that lack a required field."""

from __future__ import annotations

import pytest

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS, FeaturePlan
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.testing.builders import context_rng, payment, roots
from mule_pattern_learner.testing.fake_graph import FakeStore


def test_missing_required_message_fields_raise_instead_of_defaulting() -> None:
    sampler = SamplerPlan(roots=PoolPlan(recent=3))
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    key = roots(1)[0]
    rng = context_rng(key)
    for field in ("flow_present", "pair_prior_count", "age_ms", "amount"):
        store = FakeStore(sampler)
        message = payment(key, "zelle_out", 50, "peer", "recent", rng)
        store.row(key)["messages"] = [message]
        build_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
        del message[field]
        with pytest.raises(ValueError, match=field):
            build_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
    store = FakeStore(sampler)
    store.row(key)["messages"] = [
        payment(key, "zelle_out", 50, "peer", "recent", rng) | {"age_ms": -1}
    ]
    with pytest.raises(ValueError, match="Future"):
        build_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
    # Fields of groups outside the plan stay optional.
    store = FakeStore(sampler)
    message = payment(key, "zelle_out", 50, "peer", "recent", rng)
    del message["device_present"]
    store.row(key)["messages"] = [message]
    build_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
