"""Live provenance checks ignore experiment scope vertices."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.testing.fake_connection import executor
from mule_pattern_learner.testing.fake_graph import policy_counts
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
    manifest: dict[str, Any] = {
        "config": {"dataset_id": "snap", "scope_id": "s", "split_seed": 42},
        # Older manifests recorded the scope vertex count too.
        "source": {"source_counts": {"Account": 10, "Party": 4, "Temporal_Training_Scope": 1}},
    }
    provenance.verify_frozen_source(tg, manifest)
    counts["Temporal_Training_Scope"] = 3  # another experiment created scopes
    provenance.verify_frozen_source(tg, manifest)
    # The scope's unowned rule is rechecked on every streamed run.
    manifest["config"]["scope_unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope_unowned = 'linked'"):
        provenance.verify_frozen_source(tg, manifest)
    manifest["config"]["scope_unowned"] = "linked"
    counts["Account"] = 11
    with pytest.raises(ValueError, match="counts changed"):
        provenance.verify_frozen_source(tg, manifest)
