"""Feature arms share one preparation and state their slot sum."""

from __future__ import annotations

from mule_pattern_learner.config import run_config
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS, FeaturePlan
from mule_pattern_learner.experiments.variants import feature_experiments
from mule_pattern_learner.model.build import build_model

CONFIG = run_config()


def test_feature_arms_and_model_seeds_share_one_preparation() -> None:
    from mule_pattern_learner.config import validate_config
    from mule_pattern_learner.data.manifest import preparation_view
    from mule_pattern_learner.experiments.variants import feature_experiments
    from mule_pattern_learner.testing.builders import live_config

    base = live_config()
    views = {json_key(preparation_view(arm)) for arm in feature_experiments(base).values()}
    assert len(views) == 1  # arms of any groups and architecture fit the same preparation
    # So do models saved with a variant, and ones that named extraction groups.
    saved = [{**base, "variant": v} for v in ("no_fourier", "tabular")]
    saved.append({**base, "extraction_groups": [*base["feature_groups"], "rolling_windows"]})
    assert {json_key(preparation_view(validate_config(c))) for c in saved} == views
    reseeded = {**base, "seed": 7, "cohort_seed": base["seed"]}
    assert preparation_view(reseeded) == preparation_view(base)
    assert preparation_view({**base, "seed": 7})["cohort_seed"] == 7


def json_key(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def test_every_ablation_arm_states_its_slot_sum() -> None:
    arms = feature_experiments(CONFIG)
    for name, arm in arms.items():
        model = build_model(arm, FeaturePlan.from_config(arm))
        assert (model.slot_sum is not None) == arm["slot_sum"], name
    # The feature-group arms keep the model they were designed on.
    assert not any(arm["slot_sum"] for name, arm in arms.items() if not name.startswith("built_in"))
    keys = ("feature_groups", "architecture", "slot_sum")
    assert {k: arms["built_in"][k] for k in keys} == {k: CONFIG[k] for k in keys}
    differences = {
        name: {k for k in keys if arm[k] != CONFIG[k]}
        for name, arm in arms.items()
        if name.startswith("built_in_")
    }
    assert differences == {
        "built_in_no_slot_sum": {"slot_sum"},
        "built_in_no_pool": {"feature_groups"},
        "built_in_no_internal": {"feature_groups"},
        "built_in_tabular": {"architecture", "slot_sum"},
    }
    assert arms["built_in_no_pool"]["feature_groups"] == list(DEFAULT_GROUPS)
    assert "pool_internal_inflows" not in arms["built_in_no_internal"]["feature_groups"]
