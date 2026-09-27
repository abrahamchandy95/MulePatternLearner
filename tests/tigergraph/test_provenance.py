"""Live provenance checks ignore experiment scope vertices."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.server import GRAPH_NAME
from mule_pattern_learner.data.manifest import dataset_settings
from mule_pattern_learner.testing.builders import SNAPSHOT_SOURCE, unit_config
from mule_pattern_learner.testing.fake_connection import executor
from mule_pattern_learner.testing.fake_graph import ScopeServer, policy_counts
from mule_pattern_learner.tigergraph import provenance


def test_source_counts_ignore_experiment_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"Account": 10, "Party": 4, "Temporal_Training_Scope": 1}
    header = {"ready": True, "source_id": "snap", "split_seed": 42}
    conn = SimpleNamespace(
        getVertexCount=lambda *a, **k: dict(counts),
        getVerticesById=lambda *a: [{"attributes": dict(header)}],
        runInstalledQuery=lambda name, params, **k: [{"status": "ok", **policy_counts("linked")}],
    )
    tg = executor(conn)
    assert provenance.source_counts(tg) == {"Account": 10, "Party": 4}
    monkeypatch.setattr(provenance, "verify_sources", lambda executor: [])
    config = DEFAULT_CONFIG.with_changes({"scope": {"id": "s"}, "dataset": {"split_seed": 42}})
    manifest: dict[str, Any] = {
        # Older manifests recorded the scope vertex count too.
        "source": {
            "source_counts": {"Account": 10, "Party": 4, "Temporal_Training_Scope": 1},
            "settings": dataset_settings("snap", config),
        },
    }
    provenance.verify_frozen_source(tg, manifest)
    counts["Temporal_Training_Scope"] = 3  # another experiment created scopes
    provenance.verify_frozen_source(tg, manifest)
    # The scope's unowned rule is rechecked on every streamed run.
    settings = manifest["source"]["settings"]
    settings["scope"]["unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope.unowned = 'linked'"):
        provenance.verify_frozen_source(tg, manifest)
    settings["scope"]["unowned"] = "linked"
    # So is the scope's source.
    settings["source_id"] = "another"
    with pytest.raises(ValueError, match="no longer valid.*different source"):
        provenance.verify_frozen_source(tg, manifest)
    settings["source_id"] = "snap"
    counts["Account"] = 11
    with pytest.raises(ValueError, match="counts changed"):
        provenance.verify_frozen_source(tg, manifest)


def test_the_source_id_comes_from_the_scope_or_the_graph() -> None:
    counts = {"Account": 10, "Party": 4}
    header = {"ready": True, "source_id": SNAPSHOT_SOURCE, "split_seed": 42}
    scope_id = unit_config().scope.id
    resolved = provenance.resolve_source_id(
        cast(Any, ScopeServer(header, "linked")), scope_id, counts
    )
    assert resolved == SNAPSHOT_SOURCE
    fresh = cast(Any, ScopeServer(None, "linked"))
    derived = provenance.resolve_source_id(fresh, scope_id, counts)
    assert derived == provenance.derived_source_id(counts)
    assert derived.startswith(GRAPH_NAME + "_")
    assert derived != provenance.derived_source_id({**counts, "Account": 11})
