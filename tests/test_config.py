"""The configuration sections: the built-in run, checked values, JSON and fingerprints."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import json
import re
from typing import Any

import pytest

from mule_pattern_learner.config import (
    BUILT_IN_SAMPLER,
    BUILT_IN_SELECTION,
    DEFAULT_CONFIG,
    SELECTION_RULES,
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
from mule_pattern_learner.contract.fingerprints import fingerprint
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.paths import REPOSITORY_ROOT

# A row of the configuration reference: the setting and its built-in value, as code.
REFERENCE_ROW = re.compile(r"^\| `([a-z_.]+)` \| `([^`]*)` \|", re.M)


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
        (lambda: TrainingConfig(selection="test_ap"), "training.selection"),
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


def test_the_built_in_run_keeps_its_settings() -> None:
    # Every setting that can change results is in the fingerprint, so no default changes
    # without this literal; the configuration reference is edited with the code, so it
    # cannot pin them. The settings the golden run overrides are spelled out.
    assert DEFAULT_CONFIG.fingerprint() == (
        "39ea88acbe405b5200c6eee0f7145564b17148cbe7cc6fcfcc0cde0bd561e521"
    )
    training = DEFAULT_CONFIG.training
    assert (training.epochs, training.steps_per_epoch, training.patience) == (30, 100, 6)
    assert training.batch_size == 64 and DEFAULT_CONFIG.model.dropout == 0.15
    limits = DEFAULT_CONFIG.dataset.seed_limits
    assert (limits.train, limits.validation, limits.test) == (20_000, 2_000, 2_000)


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


def test_the_built_in_selection_rule_leaves_every_fingerprint_as_it_was() -> None:
    base = DEFAULT_CONFIG.fingerprint()
    assert DEFAULT_CONFIG.training.selection == BUILT_IN_SELECTION == "validation_ap"
    # A config.json written before the setting existed has no selection: it reads as the
    # built-in rule, and the fingerprint is the one it recorded, which covered every
    # setting but transport, runtime and the sampler backend.
    for changes in ({}, {"training": {"epochs": 3}}, {"loss": {"positive_weight": "prior"}}):
        config = DEFAULT_CONFIG.with_changes(changes)
        earlier = config.to_dict()
        del earlier["training"]["selection"]
        assert RunConfig.from_dict(earlier) == config
        del earlier["transport"], earlier["runtime"], earlier["sampler"]["backend"]
        assert config.fingerprint() == fingerprint(earlier)
    # Every other rule is a setting that changes results, named as such.
    for rule in SELECTION_RULES[1:]:
        chosen = DEFAULT_CONFIG.with_changes({"training": {"selection": rule}})
        assert chosen.fingerprint() != base
        assert differing_settings(chosen.results_view(), DEFAULT_CONFIG.results_view()) == [
            "training.selection"
        ]
        assert chosen.to_dict()["training"]["selection"] == rule


def settings(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every setting of a to_dict table by its dotted name, with its value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from settings(item, f"{path}.{key}" if path else key)
    else:
        yield path, value


def test_the_configuration_reference_lists_every_setting_with_its_value() -> None:
    page = (REPOSITORY_ROOT / "docs/reference/configuration.md").read_text()
    rows = REFERENCE_ROW.findall(page)
    names = [name for name, _ in rows]
    assert len(names) == len(set(names))
    # Each value as config.json records it.
    expected = {name: json.dumps(value) for name, value in settings(DEFAULT_CONFIG.to_dict())}
    assert dict(rows) == expected
