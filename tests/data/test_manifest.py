"""The dataset manifest: its preparation keys and the GSQL sources it records."""

from __future__ import annotations

from pathlib import Path

from mule_pattern_learner.config import validate_config
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS
from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.testing.builders import (
    FrameObservedLabels,
    live_config,
    scoped_accounts,
    supplied_labels,
    unit_config,
)
from mule_pattern_learner.contract.server import QUERY_FILES
from mule_pattern_learner.testing.fake_graph import FakeExecutor


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


def test_a_dataset_records_its_query_files_by_path_and_passes_its_own_check(
    tmp_path: Path,
) -> None:
    population = scoped_accounts()
    executor = FakeExecutor({}, population=population)
    config = live_config(seed_limits={"train": 10, "validation": 10, "test": 10})
    dataset = tmp_path / "dataset"
    manifest = prepare(
        config,
        dataset,
        executor,
        {"Account": len(population)},
        FrameObservedLabels(supplied_labels()),
    )
    recorded = manifest["source"]["query_hashes"]
    assert set(recorded) == set(QUERY_FILES) == set(data_manifest.query_hashes())
    assert data_manifest.changed_query_files(manifest) == []
    data_manifest.load_prepared(dataset)


def test_query_files_are_compared_by_their_text_not_their_path() -> None:
    current = data_manifest.query_hashes()
    # Datasets prepared before the query files moved recorded the same texts elsewhere.
    moved = {f"gsql/temporal/{Path(name).name}": text for name, text in current.items()}
    assert data_manifest.changed_query_files({"source": {"query_hashes": moved}}) == []
    # A changed or missing text is reported under the file's current path.
    changed = dict(current)
    changed["gsql/queries/hub_accounts.gsql"] = "0" * 64
    assert data_manifest.changed_query_files({"source": {"query_hashes": changed}}) == [
        "gsql/queries/hub_accounts.gsql"
    ]
    missing = {k: v for k, v in current.items() if k != "gsql/queries/split_cutoffs.gsql"}
    assert data_manifest.changed_query_files({"source": {"query_hashes": missing}}) == [
        "gsql/queries/split_cutoffs.gsql"
    ]
    assert data_manifest.changed_query_files({}) == sorted(current)
