"""The dataset manifest: the dataset's settings and id, and the GSQL sources it records."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mule_pattern_learner.contract.feature_groups import CORE_GROUPS
from mule_pattern_learner.contract.server import QUERY_FILES
from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    scoped_accounts,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader


def test_the_dataset_id_covers_only_the_dataset_settings() -> None:
    base = unit_config()
    settings = data_manifest.dataset_settings("source", base)
    assert tuple(settings) == data_manifest.DATASET_SETTINGS
    identity = data_manifest.dataset_id("source", base)
    same: list[dict[str, Any]] = [
        {"training": {"learning_rate": 0.5, "epochs": 3, "seed": 1}, "model": {"hidden": 8}},
        {"transport": {"request_batch_size": 4, "max_outage_s": 60}},
        {"scope": {"unowned": "linked", "create": False, "reveal_per_split": 5, "reveal_salt": 7}},
        {"sampler": {"association_slots": 1, "fanouts": [8, 2]}},  # selection, not pools
        # A dataset stores no contexts, so feature groups and architecture do not count.
        {"features": list(CORE_GROUPS), "model": {"architecture": "summary"}},
    ]
    for change in same:
        assert data_manifest.dataset_id("source", base.with_changes(change)) == identity, change
    different: list[dict[str, Any]] = [
        {"dataset": {"split_seed": 1}},
        {"dataset": {"seed": 1}},
        {"dataset": {"dates": {"test": ["2025-02-01"]}}},
        {"dataset": {"seed_limits": {"train": 10}}},
        {"scope": {"id": "other_scope"}},
        {"scope": {"unowned": "independent"}},
        {"scope": {"unowned": "shared"}},
        {"sampler": {"children": {"recent": 2}}},
    ]
    for change in different:
        assert data_manifest.dataset_id("source", base.with_changes(change)) != identity, change
    assert data_manifest.dataset_id("other_source", base) != identity
    manifest = {"source": {"source_id": "source", "settings": settings}}
    changed = base.with_changes({"dataset": {"split_seed": 1}})
    assert data_manifest.dataset_mismatches(changed, manifest) == ["dataset.split_seed"]
    # A manifest without settings differs in all of them.
    assert data_manifest.dataset_mismatches(base, {"source": {}}) == list(
        data_manifest.DATASET_SETTINGS
    )


def test_a_dataset_records_its_query_files_by_path_and_passes_its_own_check(
    tmp_path: Path,
) -> None:
    population = scoped_accounts()
    executor = FakeTigerGraph({}, population=population)
    config = unit_config(dataset={"seed_limits": {"train": 10, "validation": 10, "test": 10}})
    dataset = DatasetPaths(tmp_path / "dataset")
    manifest = prepare(
        config,
        UNIT_SOURCE,
        dataset,
        {"Account": len(population)},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(executor),
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
    )
    recorded = manifest["source"]["query_hashes"]
    assert set(recorded) == set(QUERY_FILES) == set(data_manifest.query_hashes())
    assert data_manifest.changed_query_files(manifest) == []
    data_manifest.load_prepared(dataset)


def test_a_manifest_names_its_dataset_and_the_frozen_source_it_was_prepared_from(
    tmp_path: Path,
) -> None:
    population = scoped_accounts()
    executor = FakeTigerGraph({}, population=population)
    config = unit_config(dataset={"seed_limits": {"train": 10, "validation": 10, "test": 10}})
    manifest = prepare(
        config,
        UNIT_SOURCE,
        DatasetPaths(tmp_path / "dataset"),
        {"Account": len(population)},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(executor),
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
    )
    # The dataset id of its recorded settings is the one its directory is named by.
    assert data_manifest.recorded_dataset_id(manifest) == data_manifest.dataset_id(
        UNIT_SOURCE, config
    )
    # The frozen source is named by the counts and the scope the live graph is checked
    # against, so other counts name another source of the same dataset id.
    source = data_manifest.source_fingerprint(manifest)
    changed = deepcopy(manifest)
    changed["source"]["source_counts"]["Account"] += 1
    assert data_manifest.source_fingerprint(changed) != source
    assert data_manifest.recorded_dataset_id(changed) == data_manifest.recorded_dataset_id(manifest)


def test_query_files_are_compared_by_their_path_and_text() -> None:
    current = data_manifest.query_hashes()
    # The same texts recorded under other paths, as the datasets prepared before the
    # layered restructure recorded them, are not this code's files.
    moved = {f"temporal/{Path(name).name}": text for name, text in current.items()}
    assert data_manifest.changed_query_files({"source": {"query_hashes": moved}}) == sorted(current)
    # Two files' texts recorded under each other's paths do not match either.
    first, second = sorted(current)[:2]
    swapped = {**current, first: current[second], second: current[first]}
    assert data_manifest.changed_query_files({"source": {"query_hashes": swapped}}) == [
        first,
        second,
    ]
    assert data_manifest.changed_query_files({"source": {"query_hashes": current}}) == []
    # A changed or missing text is reported under the file's current path.
    changed = dict(current)
    changed["queries/hub_accounts.gsql"] = "0" * 64
    assert data_manifest.changed_query_files({"source": {"query_hashes": changed}}) == [
        "queries/hub_accounts.gsql"
    ]
    missing = {k: v for k, v in current.items() if k != "queries/split_cutoffs.gsql"}
    assert data_manifest.changed_query_files({"source": {"query_hashes": missing}}) == [
        "queries/split_cutoffs.gsql"
    ]
    assert data_manifest.changed_query_files({}) == sorted(current)
