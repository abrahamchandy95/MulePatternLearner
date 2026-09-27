"""Preparation keys fingerprint only the settings that change a preparation."""

from __future__ import annotations

from pathlib import Path

from mule_pattern_learner.config import validate_config
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS
from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.testing.builders import unit_config


def test_preparation_keys_fingerprint_only_preparation_settings(tmp_path: Path) -> None:
    base = unit_config(tmp_path)
    view = data_manifest.preparation_view(base)
    assert tuple(view) == data_manifest.PREPARATION_KEYS
    same = [
        {"learning_rate": 0.5, "epochs": 3, "hidden": 8, "request_batch_size": 4},
        {"scope_unowned": "linked", "max_outage_s": 60},
        {"sampler": {**base["sampler"], "association_slots": 1}},  # selection, not pools
        # A preparation stores no contexts, so feature groups and architecture do not count.
        {"feature_groups": list(DEFAULT_GROUPS), "architecture": "summary"},
    ]
    for change in same:
        assert data_manifest.preparation_fingerprint(
            {**base, **change}
        ) == data_manifest.preparation_fingerprint(base), change
    assert data_manifest.preparation_fingerprint(
        validate_config(base)
    ) == data_manifest.preparation_fingerprint(base)
    different = [
        {"split_seed": 1},
        {"seed": 1},
        {"prepared_id": "again"},
        {"dates": {**base["dates"], "test": ["2025-02-01"]}},
        {"scope_unowned": "independent"},
        {"scope_unowned": "shared"},
        {"sampler": {**base["sampler"], "children": {"recent": 2, "associations": 0}}},
    ]
    for change in different:
        assert data_manifest.preparation_fingerprint(
            {**base, **change}
        ) != data_manifest.preparation_fingerprint(base), change
    manifest = {"source": {"preparation": view}}
    assert data_manifest.preparation_mismatches({**base, "split_seed": 1}, manifest) == [
        "split_seed"
    ]
