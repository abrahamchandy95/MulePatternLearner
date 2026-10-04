"""Every variant builds offline, shares the baseline's dataset and names a run of its own."""

from __future__ import annotations

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import (
    BUILT_IN_GROUPS,
    CLIENT_GROUPS,
    extraction_plan,
)
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.experiments.variants import (
    BASELINE,
    CONTROLS,
    DEFAULT_SUITE,
    FEATURE_DROPS,
    METHODS,
    SEEDS,
    SUITES,
    VARIANTS,
    Variant,
    describe,
    seeds_text,
    select,
)
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.paths import BASELINE_VARIANT, RunPaths
from mule_pattern_learner.pipeline.train import BASELINE_RUN
from mule_pattern_learner.training.trainer import check_limits

# The settings no variant may change: the seeds of the dataset and of the reveal.
SEED_SETTINGS = ("dataset.seed", "dataset.split_seed", "scope.reveal_salt", "training.seed")


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS))
def test_each_variant_builds_offline_on_the_baseline_dataset(variant: Variant) -> None:
    for seed in SEEDS:
        config = variant.config(DEFAULT_CONFIG, seed)
        plan = config.feature_plan()
        check_limits(config, plan)
        model = build_model(config.model, plan, config.sampler.fanouts[0])
        assert sum(p.numel() for p in model.parameters()) > 0
        # The seed is the one setting config() adds; the source requests the model's inputs.
        assert config.training.seed == seed
        requested = tuple(g for g in config.features if g not in CLIENT_GROUPS)
        assert extraction_plan(plan).groups == requested
        # Every variant and seed trains on the baseline's dataset.
        assert dataset_id("source", config) == dataset_id("source", DEFAULT_CONFIG)
    # A variant changes results-relevant settings, never a seed, and says which.
    assert not set(variant.changes()) & set(SEED_SETTINGS)
    assert bool(variant.changes()) is (variant is not BASELINE)
    assert variant.question and variant.change_text()


def test_the_variants_name_distinct_runs_and_seed_42_of_the_baseline_is_mule_train() -> None:
    for seed in SEEDS:
        prints = [v.config(DEFAULT_CONFIG, seed).fingerprint() for v in VARIANTS.values()]
        assert len(set(prints)) == len(VARIANTS)
    assert BASELINE.config(DEFAULT_CONFIG, 42) == DEFAULT_CONFIG
    assert RunPaths.of(BASELINE.name, 42) == BASELINE_RUN and BASELINE.name == BASELINE_VARIANT


def test_a_drop_takes_the_groups_that_read_it_along() -> None:
    features = {v.name: set(v.config(DEFAULT_CONFIG, 42).features) for v in FEATURE_DROPS}
    removed = {name: set(BUILT_IN_GROUPS) - kept for name, kept in features.items()}
    assert removed == {
        "drop_entity_meta": {"entity_meta"},
        "drop_hub_indicator": {"hub_indicator"},
        "drop_time_encoding": {"time_encoding"},
        "drop_pair_history": {"pair_history", "pool_activity", "pool_internal_inflows"},
        "drop_flow_timing": {"flow_timing", "pool_activity"},
        "drop_pool_activity": {"pool_activity"},
        "drop_pool_internal_inflows": {"pool_internal_inflows"},
    }
    assert (
        "pool_activity and pool_internal_inflows read it" in VARIANTS["drop_pair_history"].question
    )
    assert "pool_activity reads it, so it goes too" in VARIANTS["drop_flow_timing"].question


def test_the_suites_start_with_the_baseline_and_all_holds_each_variant_once() -> None:
    assert list(SUITES) == ["controls", "feature_drops", "methods", "all"]
    assert all(suite[0] is BASELINE for suite in SUITES.values())
    assert [v.name for v in CONTROLS] == [
        "no_attention",
        "no_slot_sum",
        "no_pool_counts",
        "prior_weight",
        "no_weight_average",
        "drop_time_encoding",
    ]
    assert len(SUITES["all"]) == 18 == len(VARIANTS)
    # No variant reads a group outside training: there are no additions.
    assert all(
        set(v.config(DEFAULT_CONFIG, 42).features) <= set(BUILT_IN_GROUPS)
        for v in VARIANTS.values()
    )


def test_the_methods_suite_changes_the_selection_and_the_prior_on_the_baseline_dataset() -> None:
    assert [v.name for v in SUITES["methods"]] == [
        "baseline",
        "select_on_roc_auc",
        "select_on_pu_risk",
        "fixed_10_epochs",
        "prior_tenth",
        "prior_tenfold",
    ]
    assert {v.name: v.changes() for v in METHODS} == {
        "select_on_roc_auc": {"training.selection": "validation_roc_auc"},
        "select_on_pu_risk": {"training.selection": "validation_pu_risk"},
        "fixed_10_epochs": {"training.epochs": 10, "training.selection": "none"},
        "prior_tenth": {"loss.class_prior": pytest.approx(0.0001)},
        "prior_tenfold": {"loss.class_prior": pytest.approx(0.01)},
    }
    # One dataset for the suite and a run of its own for each variant and seed, as the
    # suite's offline check requires.
    for seed in SEEDS:
        configs = [v.config(DEFAULT_CONFIG, seed) for v in SUITES["methods"]]
        assert {dataset_id("source", config) for config in configs} == {
            dataset_id("source", DEFAULT_CONFIG)
        }
        assert len({config.fingerprint() for config in configs}) == len(configs)
    # The questions are asked for real use, and the prior's say what to expect.
    assert all("a handful" in v.question for v in METHODS[:3])
    assert all("near-null result is expected" in v.question for v in METHODS[3:])


def test_names_select_suites_and_variants_and_always_the_baseline() -> None:
    assert select() == (DEFAULT_SUITE, SUITES[DEFAULT_SUITE])
    name, chosen = select(["prior_weight", "no_attention", "prior_weight"])
    assert name == "prior_weight-no_attention"
    assert [v.name for v in chosen] == ["baseline", "prior_weight", "no_attention"]
    name, chosen = select(["feature_drops", "no_attention"])
    assert [v.name for v in chosen] == [v.name for v in SUITES["feature_drops"]] + ["no_attention"]
    with pytest.raises(ValueError, match=r"Unknown suites or variants \['no_graph'\]"):
        select(["no_graph"])


def test_ten_seeds_train_every_variant_and_the_help_names_them() -> None:
    assert SEEDS == tuple(range(42, 52)) and SEEDS[0] == DEFAULT_CONFIG.training.seed
    assert seeds_text(SEEDS) == "the 10 seeds 42 to 51"
    assert seeds_text((42,)) == "the seed 42"
    assert seeds_text((42, 43)) == "the seeds 42 and 43"
    assert seeds_text((44, 42, 43, 47)) == "the seeds 42, 43, 44 and 47"
    assert "variants (each trained with the 10 seeds 42 to 51):" in describe()


def test_the_help_lists_every_suite_and_variant_with_its_changes() -> None:
    text = describe()
    assert all(name in text for name in [*SUITES, *VARIANTS])
    assert "(model.architecture = summary; model.slot_sum = False)" in text
    assert "(loss.positive_weight = prior)" in text
    assert "(features without pool_activity, pool_internal_inflows)" in text
