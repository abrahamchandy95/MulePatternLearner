"""Preparation against the graph: reuse of a ready dataset, the source id, first-run steps."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from mule_pattern_learner.config import RunConfig, ScopeConfig, SplitDates, TransportConfig
from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.data.manifest import dataset_id, dataset_settings, query_hashes
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.pipeline import prepare as pipeline_prepare
from mule_pattern_learner.testing.builders import (
    SNAPSHOT_SOURCE,
    UNIT_SOURCE,
    example_config,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import ScopeServer


@pytest.fixture
def fixed_hashes(monkeypatch: pytest.MonkeyPatch):
    hashes = {"queries/training_context.gsql": "aaa", "queries/hub_accounts.gsql": "b"}
    monkeypatch.setattr(data_manifest, "query_hashes", lambda: dict(hashes))
    return hashes


def write_manifest(
    data: Path,
    config: RunConfig,
    hashes: dict[str, Any],
    status: str = "ready",
    source_id: str = SNAPSHOT_SOURCE,
) -> tuple[DatasetPaths, dict[str, Any]]:
    """A manifest in the directory of config's dataset of source_id under data."""
    manifest = {
        "status": status,
        "source": {
            "source_id": source_id,
            "query_hashes": dict(hashes),
            "settings": dataset_settings(source_id, config),
        },
    }
    dataset = DatasetPaths.of(dataset_id(source_id, config), data)
    dataset.root.mkdir(parents=True, exist_ok=True)
    dataset.manifest.write_text(json.dumps(manifest))
    return dataset, manifest


def test_prepare_dataset_checks_query_hashes_before_reusing_a_ready_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_hashes: dict[str, str]
) -> None:
    config = unit_config()
    data = tmp_path / "data"
    dataset, _ = write_manifest(data, config, fixed_hashes)

    def no_connection(transport: Any) -> None:
        raise AssertionError("prepare_dataset must not connect for a ready dataset")

    monkeypatch.setattr(pipeline_prepare, "connect", no_connection)
    assert pipeline_prepare.prepare_dataset(config, data) == dataset
    # A query file preparation no longer uses cannot change the dataset.
    write_manifest(data, config, {**fixed_hashes, "queries/retired_query.gsql": "old"})
    assert pipeline_prepare.prepare_dataset(config, data) == dataset
    # Model and transport settings are not dataset settings.
    changes = {"training": {"learning_rate": 0.1}, "transport": {"query_concurrency": 2}}
    assert pipeline_prepare.prepare_dataset(config.with_changes(changes), data) == dataset
    # Other dataset settings name another dataset, which is prepared beside this one.
    split_seed = config.with_changes({"dataset": {"split_seed": 7}})
    assert pipeline_prepare.find_datasets(split_seed, data) == []
    fixed_hashes["queries/hub_accounts.gsql"] = "changed"
    with pytest.raises(ValueError, match=r"hub_accounts\.gsql.*mule install.*aside"):
        pipeline_prepare.prepare_dataset(config, data)
    with pytest.raises(ValueError, match="different GSQL sources"):
        data_manifest.load_prepared(dataset)


def test_the_source_id_comes_from_the_scope_or_the_graph() -> None:
    counts = {"Account": 10, "Party": 4}
    header = {"ready": True, "source_id": SNAPSHOT_SOURCE, "split_seed": 42}
    scope_id = unit_config().scope.id
    server = ScopeServer(header, "linked")
    server.client.graphname = "G"
    resolved = pipeline_prepare.resolve_source_id(cast(Any, server), scope_id, counts)
    assert resolved == SNAPSHOT_SOURCE
    fresh = ScopeServer(None, "linked")
    fresh.client.graphname = "G"
    derived = pipeline_prepare.resolve_source_id(cast(Any, fresh), scope_id, counts)
    assert derived == pipeline_prepare.derived_source_id("G", counts) and derived.startswith("G_")
    assert derived != pipeline_prepare.derived_source_id("G", {**counts, "Account": 11})


def record_graph_steps(monkeypatch: pytest.MonkeyPatch, steps: list[str]) -> None:
    """Replace the graph steps of prepare_dataset by entries in steps."""

    def connect(transport: TransportConfig) -> Any:
        return SimpleNamespace()

    def install(executor: Any) -> None:
        steps.append("install")

    def source_counts(executor: Any) -> dict[str, int]:
        return {"Account": 10}

    def resolve_source_id(executor: Any, scope_id: str, counts: dict[str, int]) -> str:
        steps.append(f"resolve {scope_id}")
        return UNIT_SOURCE

    def ensure_scope(executor: Any, scope: ScopeConfig, *, source_id: str, split_seed: int) -> None:
        steps.append(f"scope {scope.id} {source_id} {split_seed}")

    def ensure_revealed_labels(executor: Any, scope: ScopeConfig, dates: SplitDates) -> None:
        steps.append("reveal")

    def prepare(
        config: RunConfig, source_id: str, dataset: DatasetPaths, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        steps.append(f"prepare {source_id} {dataset.root.name}")
        return {"status": "ready"}

    for stand_in in (
        connect,
        install,
        source_counts,
        resolve_source_id,
        ensure_scope,
        ensure_revealed_labels,
        prepare,
    ):
        monkeypatch.setattr(pipeline_prepare, stand_in.__name__, stand_in)


def test_first_preparation_creates_the_scope_and_reveals_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps: list[str] = []
    record_graph_steps(monkeypatch, steps)
    config = unit_config()
    data = tmp_path / "data"
    identity = dataset_id(UNIT_SOURCE, config)
    assert pipeline_prepare.prepare_dataset(config, data) == DatasetPaths.of(identity, data)
    # The reveal draws its splits from the scope partitions, so the scope comes first.
    assert steps == [
        "install",
        "resolve unit_scope",
        f"scope unit_scope {UNIT_SOURCE} 42",
        "reveal",
        f"prepare {UNIT_SOURCE} {identity}",
    ]


def test_a_dataset_being_prepared_keeps_its_source_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_hashes: dict[str, str]
) -> None:
    config = unit_config()
    data = tmp_path / "data"
    dataset, _ = write_manifest(data, config, fixed_hashes, status="preparing")
    steps: list[str] = []
    record_graph_steps(monkeypatch, steps)
    assert pipeline_prepare.prepare_dataset(config, data) == dataset
    assert steps == [
        "install",
        f"scope unit_scope {SNAPSHOT_SOURCE} 42",
        "reveal",
        f"prepare {SNAPSHOT_SOURCE} {dataset.root.name}",
    ]


def test_datasets_of_several_sources_read_the_source_id_from_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_hashes: dict[str, str]
) -> None:
    # After a reload the old source's dataset stays beside the new one's.
    config = unit_config()
    data = tmp_path / "data"
    write_manifest(data, config, fixed_hashes)
    current, _ = write_manifest(data, config, fixed_hashes, source_id=UNIT_SOURCE)
    assert len(pipeline_prepare.find_datasets(config, data)) == 2
    steps: list[str] = []
    record_graph_steps(monkeypatch, steps)
    assert pipeline_prepare.prepare_dataset(config, data) == current
    assert steps[:2] == ["install", "resolve unit_scope"]


def test_ready_pipeline_reuses_cache_without_connecting(tmp_path: Path) -> None:
    from mule_pattern_learner.pipeline.prepare import find_datasets, prepare_dataset

    c = example_config()
    dataset, _ = write_manifest(tmp_path, c, query_hashes(), source_id=UNIT_SOURCE)
    with patch("mule_pattern_learner.pipeline.prepare.connect") as client:
        assert prepare_dataset(c, tmp_path) == dataset
        # Model settings may change; dataset settings may not, and nothing connects.
        changes = {"model": {"hidden": 32}, "training": {"learning_rate": 0.01}}
        assert prepare_dataset(c.with_changes(changes), tmp_path) == dataset
        assert (
            find_datasets(c.with_changes({"dataset": {"seed_limits": {"test": 10}}}), tmp_path)
            == []
        )
        client.assert_not_called()
