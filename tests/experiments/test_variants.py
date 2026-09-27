"""Feature variants share one dataset and state their slot sum."""

from __future__ import annotations

from typing import Any

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.experiments.variants import feature_experiments
from mule_pattern_learner.inference.saved_model import saved_run_config
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.testing.builders import live_config

CONFIG = DEFAULT_CONFIG


def test_feature_variants_and_model_seeds_share_one_dataset() -> None:
    base = live_config()
    ids = {dataset_id("source", variant) for variant in feature_experiments(base).values()}
    assert ids == {dataset_id("source", base)}  # any groups and architecture fit one dataset
    # So do models saved with a variant, and ones that named extraction groups.
    saved = [saved_run_config({"variant": v}) for v in ("no_fourier", "tabular")]
    saved.append(saved_run_config({"extraction_groups": [*CONFIG.features, "rolling_windows"]}))
    assert {dataset_id("source", config) for config in saved} == {dataset_id("source", CONFIG)}
    # And other model seeds: the reservoir seed is the dataset's own.
    reseeded = base.with_changes({"training": {"seed": 7}})
    assert dataset_id("source", reseeded) == dataset_id("source", base)
    other = base.with_changes({"dataset": {"seed": 7}})
    assert dataset_id("source", other) != dataset_id("source", base)


def model_keys(config: RunConfig) -> dict[str, Any]:
    return {
        "features": config.features,
        "architecture": config.model.architecture,
        "slot_sum": config.model.slot_sum,
    }


def test_every_ablation_variant_states_its_slot_sum() -> None:
    variants = feature_experiments(CONFIG)
    for name, variant in variants.items():
        model = build_model(variant.model, variant.feature_plan(), variant.sampler.fanouts[0])
        # The summary architecture has no slots, so it never has a slot sum.
        has_slot_sum = isinstance(model, TGAT) and model.slot_sum is not None
        assert has_slot_sum == variant.model.slot_sum, name
    # The feature-group variants keep the model they were designed on.
    assert not any(
        v.model.slot_sum for name, v in variants.items() if not name.startswith("built_in")
    )
    built_in = model_keys(CONFIG)
    assert model_keys(variants["built_in"]) == built_in
    differences = {
        name: {k for k, v in model_keys(variant).items() if v != built_in[k]}
        for name, variant in variants.items()
        if name.startswith("built_in_")
    }
    assert differences == {
        "built_in_no_slot_sum": {"slot_sum"},
        "built_in_no_pool": {"features"},
        "built_in_no_internal": {"features"},
        "built_in_tabular": {"architecture", "slot_sum"},
    }
    assert variants["built_in_no_pool"].features == DEFAULT_GROUPS
    assert "pool_internal_inflows" not in variants["built_in_no_internal"].features
