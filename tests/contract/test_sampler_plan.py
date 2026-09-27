"""Candidate pools, sampler plans and their fingerprints."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import pickle

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.fingerprints import fingerprint
from mule_pattern_learner.contract.sampler_plan import SELECTION_KEYS_VERSION, PoolPlan, SamplerPlan
from mule_pattern_learner.testing.builders import RESAMPLE


def test_sampler_plan_pools_bounds_and_values() -> None:
    plan = SamplerPlan(roots=PoolPlan(4, 3, 2, 2, 2048))
    assert plan.children == PoolPlan(4, 3, 2, 0, 2048)
    assert plan.response_bound() == 64 and plan.response_bound(2) == 36
    assert plan.query_params() == {
        "per_relation": 4,
        "k_old": 3,
        "k_div": 2,
        "k_assoc": 2,
        "max_history": 2048,
    }
    assert SamplerPlan().roots == PoolPlan() and SamplerPlan().children == PoolPlan(associations=0)
    assert RESAMPLE.children == replace(RESAMPLE.roots, associations=0)
    assert RESAMPLE.query_params(2)["k_assoc"] == 0 and RESAMPLE.response_bound(2) == 36
    assert RESAMPLE.pool(1) is RESAMPLE.roots and RESAMPLE.pool(2) is RESAMPLE.children
    with pytest.raises(ValueError, match="recent"):
        PoolPlan(recent=33)
    with pytest.raises(ValueError, match="backend"):
        SamplerPlan(backend="gpu")
    with pytest.raises(ValueError, match="one value per hop"):
        SamplerPlan(relation_fanouts=(2,))
    assert PoolPlan(associations=0).response_bound == 8
    assert pickle.loads(pickle.dumps(RESAMPLE)) == RESAMPLE
    assert copy.deepcopy(RESAMPLE) == RESAMPLE and hash(copy.deepcopy(RESAMPLE)) == hash(RESAMPLE)


def test_sampler_plans_hold_the_fanouts_of_both_hops() -> None:
    assert SamplerPlan().fanouts == (16, 4)
    assert SamplerPlan(fanouts=[8, 2]).fanouts == (8, 2)
    for fanouts in ((0, 4), (65, 4), (8.0, 4)):
        with pytest.raises(ValueError, match="Sampler fanouts must be an integer"):
            SamplerPlan(fanouts=fanouts)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="fanouts must have one value per hop"):
        SamplerPlan(fanouts=(8, 4, 2))


def test_sampler_fingerprints_ignore_backend_and_keep_recorded_values() -> None:
    assert RESAMPLE.fingerprint() != replace(RESAMPLE, association_slots=1).fingerprint()
    assert RESAMPLE.fingerprint() == replace(RESAMPLE, backend="torch").fingerprint()
    assert RESAMPLE.fingerprint() != replace(RESAMPLE, evaluation_seed=1).fingerprint()
    assert (
        RESAMPLE.pool_fingerprint() == replace(RESAMPLE, relation_fanouts=(1, 1)).pool_fingerprint()
    )
    assert SamplerPlan().pool_fingerprint() != RESAMPLE.pool_fingerprint()
    # The key scheme is versioned, and the policy name stays in the value.
    assert SELECTION_KEYS_VERSION == 2
    unversioned = {
        **asdict(RESAMPLE.roots),
        "children": asdict(RESAMPLE.children),
        "relation_fanouts": list(RESAMPLE.relation_fanouts),
        "association_fanout": RESAMPLE.association_fanout,
        "association_slots": RESAMPLE.association_slots,
        "evaluation_seed": RESAMPLE.evaluation_seed,
    }
    versioned = unversioned | {"policy": "resample", "selection_keys": 2}
    assert RESAMPLE.fingerprint() == fingerprint(versioned)
    assert RESAMPLE.fingerprint() != fingerprint(unversioned)
    # The built-in sampler's fingerprints as saved models and prepared datasets record
    # them; the fan-outs are not part of them.
    built_in = DEFAULT_CONFIG.sampler
    assert built_in.fingerprint() == replace(built_in, fanouts=(8, 2)).fingerprint()
    assert built_in.fingerprint() == (
        "44c3e909304a808a4052c2f8ab2111c5f7d35aa96446e5a7904927ba7f0c13e6"
    )
    assert built_in.pool_fingerprint() == (
        "28ef7d452dfbc97e974976997ffaf475c78ea7b0ae36660ecee1d3611f27a660"
    )
