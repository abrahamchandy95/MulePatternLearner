"""The configuration sections: the built-in run, checked values, JSON and fingerprints."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from mule_pattern_learner.config import (
    BUILT_IN_SAMPLER,
    DEFAULT_CONFIG,
    LossConfig,
    ModelConfig,
    RunConfig,
    RuntimeConfig,
    ScopeConfig,
    SplitDates,
    TrainingConfig,
    TransportConfig,
    differing_settings,
)
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.inference.saved_model import (
    SAVED_SETTINGS,
    SAVED_VALUES,
    saved_run_config,
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
    assert saved_run_config(OLD_BUILT_IN) == DEFAULT_CONFIG


def test_sections_refuse_values_outside_their_ranges() -> None:
    bad = [
        (lambda: SamplerPlan(fanouts=(True, 4)), "Sampler fanouts"),
        (lambda: SamplerPlan(fanouts=(8,)), "fanouts must have one value per hop"),
        (lambda: TransportConfig(query_concurrency=17), "transport.query_concurrency"),
        (lambda: TransportConfig(request_batch_size=65), "transport.request_batch_size"),
        (lambda: TransportConfig(max_outage_s=-1), "transport.max_outage_s"),
        (lambda: TransportConfig(max_outage_s=1.5), "transport.max_outage_s"),  # pyright: ignore[reportArgumentType]
        (lambda: RuntimeConfig(deterministic="yes"), "runtime.deterministic"),
        (lambda: RuntimeConfig(deterministic=1), "runtime.deterministic"),  # pyright: ignore[reportArgumentType]
        (lambda: RuntimeConfig(max_rejected_root_fraction=1.5), "max_rejected_root_fraction"),
        (lambda: ScopeConfig(unowned="all"), "scope.unowned"),
        (lambda: ScopeConfig(id=""), "scope.id"),
        (lambda: ScopeConfig(reveal_per_split=1001), "scope.reveal_per_split"),
        (lambda: SplitDates(train=("not a date",)), "not an ISO date"),
        (lambda: SplitDates(train=("2024-11-01",)), "overlap or are out of order"),
        (lambda: SplitDates(test=()), "dataset.dates.test"),
        (lambda: ModelConfig(hidden=66), "divisible by model.heads"),
        (lambda: ModelConfig(slot_sum="yes"), "model.slot_sum"),  # pyright: ignore[reportArgumentType]
        (lambda: TrainingConfig(epochs=0), "training.epochs"),
        (lambda: TrainingConfig(batch_size=129), "training.batch_size"),
        (lambda: TrainingConfig(weight_average_decay=1.0), "training.weight_average_decay"),
        (lambda: replace(DEFAULT_CONFIG, features=("entity_meta", "no_such_group")), "no_such"),
        (lambda: replace(DEFAULT_CONFIG, features=("entity_meta", "entity_meta")), "twice"),
    ]
    for build, name in bad:
        with pytest.raises(ValueError, match=name):
            build()
    for policy in ("independent", "shared", "linked"):
        assert ScopeConfig(unowned=policy).unowned == policy
    assert RuntimeConfig(deterministic="strict").deterministic == "strict"
    for weight in ("prior", "balanced", 0.5):
        assert LossConfig(positive_weight=weight).positive_weight == weight
    for weight in ("equal", 1.0, 0.0):
        with pytest.raises(ValueError, match="loss.positive_weight"):
            LossConfig(positive_weight=weight)
    # Numbers are stored as floats, so equal settings fingerprint equally.
    assert TrainingConfig(learning_rate=1).learning_rate == 1.0
    assert isinstance(TrainingConfig(learning_rate=1).learning_rate, float)


def test_tables_map_to_configurations_and_back() -> None:
    assert RunConfig() == DEFAULT_CONFIG
    table = json.loads(json.dumps(DEFAULT_CONFIG.to_dict()))
    assert table == DEFAULT_CONFIG.to_dict() and RunConfig.from_dict(table) == DEFAULT_CONFIG
    changed = DEFAULT_CONFIG.with_changes(
        {
            "sampler": {"roots": {"recent": 4}},
            "training": {"epochs": 3},
            "features": ["entity_meta"],
        }
    )
    assert changed.sampler.roots == replace(BUILT_IN_SAMPLER.roots, recent=4)
    assert changed.sampler.children == BUILT_IN_SAMPLER.children
    assert changed.training == replace(DEFAULT_CONFIG.training, epochs=3)
    assert changed.features == ("entity_meta",)
    assert RunConfig.from_dict(json.loads(json.dumps(changed.to_dict()))) == changed
    torch_only = DEFAULT_CONFIG.with_changes({"sampler": {"backend": "torch"}})
    assert torch_only.sampler == replace(BUILT_IN_SAMPLER, backend="torch")
    dates = DEFAULT_CONFIG.with_changes({"dataset": {"dates": {"train": ["2024-05-01"]}}})
    assert dates.dataset.dates == replace(DEFAULT_CONFIG.dataset.dates, train=("2024-05-01",))
    with pytest.raises(ValueError, match=r"Unknown configuration key\(s\): training.learnig_rate"):
        DEFAULT_CONFIG.with_changes({"training": {"learnig_rate": 0.1}})
    with pytest.raises(ValueError, match=r"Unknown configuration key\(s\): sampler.roots.recnt"):
        RunConfig.from_dict({"sampler": {"roots": {"recnt": 2}}})
    with pytest.raises(ValueError, match="Pool older"):
        RunConfig.from_dict({"sampler": {"children": {"older": 99}}})
    with pytest.raises(ValueError, match="scope must be a table"):
        RunConfig.from_dict({"scope": "strict_mule_v2"})


def test_the_fingerprint_covers_what_can_change_results() -> None:
    base = DEFAULT_CONFIG.fingerprint()
    changes: list[dict[str, Any]] = [
        {"scope": {"reveal_salt": 7}},
        {"dataset": {"seed": 7}},
        {"sampler": {"fanouts": [8, 4]}},
        {"features": ["entity_meta", "message_core"]},
        {"model": {"slot_sum": False}},
        {"loss": {"positive_weight": "prior"}},
        {"training": {"epochs": 3}},
    ]
    for change in changes:
        assert DEFAULT_CONFIG.with_changes(change).fingerprint() != base, change
    # Transport, runtime and the sampler backend never change a run's numbers.
    for change in (
        {"transport": {"query_concurrency": 4}},
        {"runtime": {"device": "cpu", "threads": 1, "max_rejected_root_fraction": 0.5}},
        {"sampler": {"backend": "torch"}},
    ):
        assert DEFAULT_CONFIG.with_changes(change).fingerprint() == base, change
    other = DEFAULT_CONFIG.with_changes({"dataset": {"seed": 7}, "runtime": {"threads": 1}})
    assert differing_settings(other.results_view(), DEFAULT_CONFIG.results_view()) == [
        "dataset.seed"
    ]
