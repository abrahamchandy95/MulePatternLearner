"""Live provenance checks ignore experiment scope vertices."""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.server import CUTOFF_QUERY, GRAPH_NAME
from mule_pattern_learner.data.manifest import dataset_settings
from mule_pattern_learner.testing.builders import UNIT_SOURCE, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph import provenance


def test_source_counts_ignore_experiment_scopes() -> None:
    header = {"ready": True, "source_id": "snap", "split_seed": 42}
    graph = FakeTigerGraph(counts={"Account": 10, "Party": 4}, scopes={"s": header})
    assert provenance.source_counts(graph) == {"Account": 10, "Party": 4}
    config = DEFAULT_CONFIG.with_changes({"scope": {"id": "s"}, "dataset": {"split_seed": 42}})
    manifest: dict[str, Any] = {
        # Older manifests recorded the scope vertex count too.
        "source": {
            "source_counts": {"Account": 10, "Party": 4, "Temporal_Training_Scope": 1},
            "settings": dataset_settings("snap", config),
        },
    }
    provenance.verify_frozen_source(graph, manifest)
    graph.scopes["another"] = dict(header)  # another experiment created a scope
    provenance.verify_frozen_source(graph, manifest)
    # The scope's unowned rule is rechecked on every streamed run.
    settings = manifest["source"]["settings"]
    settings["scope"]["unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope.unowned = 'linked'"):
        provenance.verify_frozen_source(graph, manifest)
    settings["scope"]["unowned"] = "linked"
    # So is the scope's source.
    settings["source_id"] = "another"
    with pytest.raises(ValueError, match="no longer valid.*different source"):
        provenance.verify_frozen_source(graph, manifest)
    settings["source_id"] = "snap"
    graph.counts["Account"] = 11
    with pytest.raises(ValueError, match="counts changed"):
        provenance.verify_frozen_source(graph, manifest)
    graph.counts["Account"] = 10
    # And the installed queries: one whose text differs is refused.
    graph.stale = frozenset({CUTOFF_QUERY})
    with pytest.raises(ValueError, match=f"{CUTOFF_QUERY} differs.*mule install"):
        provenance.verify_frozen_source(graph, manifest)


def test_the_source_id_comes_from_the_scope_or_the_graph() -> None:
    counts = {"Account": 10, "Party": 4}
    header = {"ready": True, "source_id": UNIT_SOURCE, "split_seed": 42}
    scope_id = unit_config().scope.id
    scoped = FakeTigerGraph(scopes={scope_id: header})
    assert provenance.resolve_source_id(scoped, scope_id, counts) == UNIT_SOURCE
    derived = provenance.resolve_source_id(FakeTigerGraph(), scope_id, counts)
    assert derived == provenance.derived_source_id(counts)
    assert derived.startswith(GRAPH_NAME + "_")
    assert derived != provenance.derived_source_id({**counts, "Account": 11})
