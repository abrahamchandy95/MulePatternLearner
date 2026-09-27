"""Configurations saved before the typed configuration convert to the RunConfig they meant.

The built-in run as the old code validated it maps field by field onto DEFAULT_CONFIG, and
every old setting has its place in RunConfig, names nothing now or is refused.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import pytest

from mule_pattern_learner.config import BUILT_IN_SAMPLER, DEFAULT_CONFIG
from mule_pattern_learner.contract.sampler_plan import PoolPlan
from mule_pattern_learner.inference.saved_settings import (
    REQUIRED_SETTINGS,
    SAVED_SETTINGS,
    SAVED_VALUES,
    converted_run_config,
)

# The built-in run as the code before the typed configuration validated it
# (config.run_config()), key for key.
OLD_BUILT_IN: dict[str, Any] = {
    "scope_id": "strict_mule_v2",
    "reveal_per_split": 20,
    "dates": {"train": ["2024-07-01"], "validation": ["2024-10-01"], "test": ["2025-01-01"]},
    "seed_limits": {"train": 20000, "validation": 2000, "test": 2000},
    "seed": 42,
    "split_seed": 42,
    "evaluation_unlabeled_limit": 2000,
    "feature_groups": [
        "entity_meta",
        "hub_indicator",
        "message_core",
        "time_encoding",
        "pair_history",
        "flow_timing",
        "pool_activity",
        "pool_internal_inflows",
    ],
    "architecture": "split",
    "slot_sum": True,
    "fanouts": [16, 4],
    "sampler": {
        "recent": 8,
        "older": 4,
        "distinct": 4,
        "associations": 2,
        "max_history": 2048,
        "children": {
            "recent": 4,
            "older": 2,
            "distinct": 2,
            "associations": 0,
            "max_history": 2048,
        },
        "relation_fanouts": [8, 4],
        "association_fanout": 1,
        "association_slots": 2,
        "backend": "auto",
        "evaluation_seed": 0,
    },
    "hidden": 64,
    "heads": 4,
    "dropout": 0.15,
    "epochs": 30,
    "steps_per_epoch": 100,
    "patience": 6,
    "max_rejected_root_fraction": 0.0,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "weight_average_decay": 0.99,
    "class_prior": 0.001,
    "positive_weight": "balanced",
    "device": "auto",
    "threads": 4,
    "request_batch_size": 8,
    "query_concurrency": 16,
    "context_lru_capacity": 256,
    "encoding_check_every": 64,
    "max_query_attempts": 6,
    "max_outage_s": 900,
    "prepare_batch_size": 16,
    "create_scope": True,
    "scope_unowned": "linked",
    "deterministic": True,
    "prefetch_batches": 2,
    "checkpoint_every_steps": 0,
    "log_every_steps": 10,
}


def flat(
    table: dict[str, Any], prefix: str = "", *, nested: tuple[str, ...] = ()
) -> dict[str, Any]:
    """A table's values by dotted name; tables named in nested are flattened too."""
    values: dict[str, Any] = {}
    for key, value in table.items():
        name = prefix + key
        if isinstance(value, dict) and (not nested or name in nested):
            values.update(flat(value, name + ".", nested=nested))
        else:
            values[name] = value
    return values


def test_the_default_config_is_the_old_built_in_run_field_by_field() -> None:
    new = DEFAULT_CONFIG.to_dict()
    old = flat(OLD_BUILT_IN, nested=("sampler", "sampler.children"))
    assert set(old) <= SAVED_SETTINGS.keys()
    targets = set()
    for name, value in old.items():
        target = SAVED_SETTINGS[name]
        if target is None:
            # Only the batch size of the removed SQLite context storage names nothing now.
            assert name == "prepare_batch_size", name
            continue
        field: Any = new
        for part in target.split("."):
            field = field[part]
        # The graph model's architecture "split" is "tgat" now.
        renamed = SAVED_VALUES[name].get(value, value) if name in SAVED_VALUES else value
        assert field == renamed, (name, target)
        targets.add(target)
    # The reservoir seed and the reveal salt were the training seed.
    assert DEFAULT_CONFIG.dataset.seed == DEFAULT_CONFIG.scope.reveal_salt == OLD_BUILT_IN["seed"]
    targets |= {"dataset.seed", "scope.reveal_salt"}
    # Every field of RunConfig has its old counterpart.
    uncovered = [
        name
        for name in flat(new)
        if not any(name == t or name.startswith(t + ".") for t in targets)
    ]
    assert uncovered == []
    assert converted_run_config(OLD_BUILT_IN) == DEFAULT_CONFIG


# The fan-outs and sampler table every configuration the old code saved holds, here the
# built-in run's.
OLD_SAMPLING: dict[str, Any] = {
    "fanouts": [16, 4],
    "sampler": {**asdict(BUILT_IN_SAMPLER.roots), "children": asdict(BUILT_IN_SAMPLER.children)},
}


def old(**settings: Any) -> dict[str, Any]:
    """An old flat configuration: OLD_SAMPLING with settings."""
    return {**OLD_SAMPLING, **settings}


def test_old_configurations_convert_through_the_key_table() -> None:
    built_in = converted_run_config(old())
    assert built_in == DEFAULT_CONFIG
    # Settings of removed paths were saved with the one value that remains.
    retired = {
        "context_storage": "stream",
        "evaluation_protocol": "strict_inductive",
        "label_policy": "graph_observed",
        "observed_labels": None,
        "extraction_groups": ["entity_meta", "rolling_windows"],
        "fanouts": [16, 4],
        "sampler": {**asdict(BUILT_IN_SAMPLER.roots), "policy": "resample"},
        "dataset_id": "load_fixture",
        "prepared_id": None,
        "prepare_batch_size": 16,
    }
    converted = converted_run_config(retired)
    assert converted.sampler.roots == BUILT_IN_SAMPLER.roots
    assert converted.sampler.children == replace(BUILT_IN_SAMPLER.roots, associations=0)
    assert replace(converted, sampler=DEFAULT_CONFIG.sampler) == DEFAULT_CONFIG
    for key, value, message in (
        ("context_storage", "sqlite", "context_storage = 'sqlite' is no longer supported"),
        ("evaluation_protocol", "shared_history", "evaluation_protocol = 'shared_history'"),
        ("label_policy", "observed", "label_policy = 'observed' is no longer supported"),
        ("observed_labels", "x.parquet", "observed_labels = 'x.parquet' is no longer supported$"),
        ("variant", "wide", "variant = 'wide' is no longer supported"),
        ("stage", "offline", "Unknown saved configuration key"),
    ):
        with pytest.raises(ValueError, match=message):
            converted_run_config(old(**{key: value}))
    with pytest.raises(ValueError, match="sampler.policy = 'recent' is no longer supported"):
        converted_run_config(old(sampler={"policy": "recent"}))
    # The old code gave absent fan-outs and sampler pools values of its own, which
    # nothing else would catch, so a table without them is refused.
    for missing in REQUIRED_SETTINGS:
        partial = {k: v for k, v in old().items() if k != missing}
        with pytest.raises(ValueError, match=f"names no {missing}; the code that saved it"):
            converted_run_config(partial)
    with pytest.raises(ValueError, match="names no fanouts or sampler"):
        converted_run_config({"fanouts": None})
    # The model variants became settings.
    assert converted_run_config(old(variant="temporal")) == built_in
    tabular = converted_run_config(old(variant="tabular"))
    assert tabular == replace(built_in, model=replace(built_in.model, architecture="summary"))
    no_fourier = converted_run_config(old(variant="no_fourier")).features
    assert no_fourier == tuple(g for g in built_in.features if g != "time_encoding")
    # A [sampler] table's absent pool keys were those of PoolPlan(), and its children
    # pool the roots pool without associations, changed by [sampler.children].
    partial = converted_run_config(old(sampler={"recent": 4, "children": {"older": 1}})).sampler
    assert partial.roots == PoolPlan(recent=4)
    assert partial.children == PoolPlan(recent=4, older=1, associations=0)
    assert partial.relation_fanouts == BUILT_IN_SAMPLER.relation_fanouts
    # The reservoir seed and the reveal salt defaulted to the training seed.
    seeded = converted_run_config(old(seed=7, reveal_salt=None, reveal_per_split=None))
    assert (seeded.training.seed, seeded.dataset.seed, seeded.scope.reveal_salt) == (7, 7, 7)
    assert seeded.scope.reveal_per_split == built_in.scope.reveal_per_split
    pinned = converted_run_config(old(seed=7, cohort_seed=3, reveal_salt=5))
    assert (pinned.dataset.seed, pinned.scope.reveal_salt) == (3, 5)
    # A null limit saved on purpose stays null.
    unlimited = converted_run_config(old(steps_per_epoch=None, evaluation_unlabeled_limit=None))
    assert unlimited.training.steps_per_epoch is None
    assert unlimited.training.proxy_unlabeled_limit is None
