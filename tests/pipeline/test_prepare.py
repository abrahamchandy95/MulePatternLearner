"""Preparation against the graph: reuse of a ready dataset, identity, first-run steps."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.data.manifest import preparation_view, query_hashes
from mule_pattern_learner.pipeline import prepare as pipeline_prepare
from mule_pattern_learner.pipeline import train as pipeline_train
from mule_pattern_learner.testing.builders import live_config, unit_config
from mule_pattern_learner.testing.fake_graph import ScopeServer


@pytest.fixture
def fixed_hashes(monkeypatch: pytest.MonkeyPatch):
    hashes = {"gsql/queries/training_context.gsql": "aaa", "gsql/queries/hub_accounts.gsql": "b"}
    monkeypatch.setattr(data_manifest, "query_hashes", lambda: dict(hashes))
    return hashes


def write_manifest(
    path: Path, config: dict[str, Any], hashes: dict[str, Any], status: str = "ready"
) -> dict[str, Any]:
    manifest = {
        "status": status,
        "config": config,
        "source": {
            "query_hashes": dict(hashes),
            "preparation": data_manifest.preparation_view(config),
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_prepare_live_checks_query_hashes_before_reusing_a_ready_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_hashes: dict[str, str]
) -> None:
    config = unit_config(tmp_path)
    out = tmp_path / "prepared"
    manifest = write_manifest(out, config, fixed_hashes)

    def no_connection(config: dict[str, Any]) -> None:
        raise AssertionError("prepare_live must not connect for a ready dataset")

    monkeypatch.setattr(pipeline_prepare, "live_executor", no_connection)
    assert pipeline_prepare.prepare_live(config, out) == manifest
    # A query file preparation no longer uses cannot change the cohort.
    retired = write_manifest(
        out, config, {**fixed_hashes, "gsql/queries/retired_query.gsql": "old"}
    )
    assert pipeline_prepare.prepare_live(config, out) == retired
    # Model and transport settings are not preparation settings.
    assert pipeline_prepare.prepare_live(
        {**config, "learning_rate": 0.1, "query_concurrency": 2}, out
    )
    with pytest.raises(ValueError, match=r"split_seed.*new output"):
        pipeline_prepare.prepare_live({**config, "split_seed": 7}, out)
    fixed_hashes["gsql/queries/hub_accounts.gsql"] = "changed"
    with pytest.raises(ValueError, match=r"hub_accounts\.gsql.*mule-temporal install.*new output"):
        pipeline_prepare.prepare_live(config, out)
    with pytest.raises(ValueError, match="different GSQL sources"):
        data_manifest.load_prepared(out)


def test_dataset_identity_comes_from_the_scope_or_the_graph(tmp_path: Path) -> None:
    counts = {"Account": 10, "Party": 4}
    header = {"ready": True, "source_id": "unit_snapshot", "split_seed": 42}
    config = {k: v for k, v in unit_config(tmp_path).items() if k != "dataset_id"}
    server = ScopeServer(header, "linked")
    server.client.conn.graphname = "G"
    assert pipeline_prepare.resolve_identity(cast(Any, server), config, counts)["dataset_id"] == (
        "unit_snapshot"
    )
    fresh = ScopeServer(None, "linked")
    fresh.client.conn.graphname = "G"
    derived = pipeline_prepare.resolve_identity(cast(Any, fresh), config, counts)["dataset_id"]
    assert derived == pipeline_prepare.derived_dataset_id("G", counts) and derived.startswith("G_")
    assert derived != pipeline_prepare.derived_dataset_id("G", {**counts, "Account": 11})
    explicit = {**config, "dataset_id": "pinned"}
    assert pipeline_prepare.resolve_identity(cast(Any, fresh), explicit, counts) == explicit
    # With an existing dataset the identity comes from its manifest, and a pin is kept.
    manifest = {"source": {"dataset_id": "unit_snapshot"}}
    assert pipeline_train.prepared_config(config, manifest)["dataset_id"] == "unit_snapshot"
    assert pipeline_train.prepared_config(explicit, manifest)["dataset_id"] == "pinned"


def test_first_preparation_creates_the_scope_and_reveals_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps: list[str] = []
    monkeypatch.setattr(pipeline_prepare, "live_executor", lambda config: SimpleNamespace())
    monkeypatch.setattr(pipeline_prepare, "install", lambda executor: steps.append("install"))
    monkeypatch.setattr(pipeline_prepare, "source_counts", lambda executor: {"Account": 10})
    monkeypatch.setattr(
        pipeline_prepare, "ensure_scope", lambda executor, config: steps.append("scope")
    )
    monkeypatch.setattr(
        pipeline_prepare, "ensure_revealed_labels", lambda executor, config: steps.append("reveal")
    )

    def prepare(config: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        steps.append("prepare")
        return {"status": "ready"}

    monkeypatch.setattr(pipeline_prepare, "prepare", prepare)
    assert pipeline_prepare.prepare_live(unit_config(tmp_path), tmp_path / "run") == {
        "status": "ready"
    }
    # The reveal draws its splits from the scope partitions, so the scope comes first.
    assert steps == ["install", "scope", "reveal", "prepare"]


def test_ready_pipeline_reuses_cache_without_connecting(tmp_path: Path) -> None:
    from mule_pattern_learner.pipeline.prepare import prepare_live

    c = live_config()
    manifest = {
        "status": "ready",
        "source": {"query_hashes": query_hashes(), "preparation": preparation_view(c)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with patch("mule_pattern_learner.pipeline.prepare.live_executor") as client:
        assert prepare_live(c, tmp_path) == manifest
        # Model settings may change; preparation settings may not, and nothing connects.
        assert prepare_live({**c, "hidden": 32, "learning_rate": 0.01}, tmp_path) == manifest
        with pytest.raises(ValueError, match="seed_limits"):
            prepare_live({**c, "seed_limits": {**c["seed_limits"], "test": 10}}, tmp_path)
        client.assert_not_called()
