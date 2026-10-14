"""Provenance checks ignore the split's vertex and check the split and the labels' reveal."""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.server import CUTOFF_QUERY, SCOPE_VERTEX
from mule_pattern_learner.data.manifest import dataset_settings, recorded_dataset_id
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, ready_scope
from mule_pattern_learner.tigergraph import provenance


def test_source_counts_ignore_the_split_and_its_settings_are_rechecked() -> None:
    graph = FakeTigerGraph(counts={"Account": 10, "Party": 4}, scope=ready_scope())
    counts = provenance.source_counts(graph)
    # Creating the split adds a scope vertex, which names no other source.
    assert counts == provenance.source_counts(FakeTigerGraph(counts=counts)) == graph.counts
    config = DEFAULT_CONFIG.with_changes({"dataset": {"split_seed": 42}})
    manifest: dict[str, Any] = {
        "source": {"source_counts": counts, "settings": dataset_settings(counts, config)},
    }
    dataset_id = recorded_dataset_id(manifest)
    provenance.verify_frozen_source(graph, manifest)
    # The split's seed and shares are rechecked on every streamed run.
    settings = manifest["source"]["settings"]
    settings["dataset"]["split_seed"] = 7
    with pytest.raises(ValueError, match="no longer valid.*split seed 42 .* not 7"):
        provenance.verify_frozen_source(graph, manifest)
    settings["dataset"]["split_seed"] = 42
    settings["scope"]["shares"] = [0.7, 0.15, 0.15]
    with pytest.raises(ValueError, match=r"no longer valid.*not 42 and \[0.7, 0.15, 0.15\]"):
        provenance.verify_frozen_source(graph, manifest)
    settings["scope"]["shares"] = list(config.scope.shares)
    # And the split itself: one its owner removed, or one never finished, is refused. A
    # ready dataset is reused without creating a split, so the refusal says to move it.
    split, graph.scope = graph.scope, None
    moved = rf"no longer valid: The graph has no split.*move this dataset \(data/{dataset_id}/\)"
    with pytest.raises(ValueError, match=moved):
        provenance.verify_frozen_source(graph, manifest)
    graph.scope = {**ready_scope(), "ready": False}
    with pytest.raises(ValueError, match=f"no longer valid.*incomplete.*{SCOPE_VERTEX}"):
        provenance.verify_frozen_source(graph, manifest)
    graph.scope = split
    graph.counts["Account"] = 11
    with pytest.raises(ValueError, match="counts changed"):
        provenance.verify_frozen_source(graph, manifest)
    graph.counts["Account"] = 10
    # And the installed queries: one whose text differs is refused.
    graph.stale = frozenset({CUTOFF_QUERY})
    with pytest.raises(ValueError, match=f"{CUTOFF_QUERY} differs.*mule install"):
        provenance.verify_frozen_source(graph, manifest)
    graph.stale = frozenset()
    # And the labels: those another reveal wrote are not the ones the dataset read.
    graph.other_reveal = 3
    with pytest.raises(ValueError, match="3 mules have another reveal's label"):
        provenance.verify_frozen_source(graph, manifest)
    graph.other_reveal = 0
    provenance.verify_frozen_source(graph, manifest)
    # A dataset of earlier code, which recorded no reveal settings, is refused.
    del settings["scope"]["reveal_salt"]
    with pytest.raises(ValueError, match="no reveal settings"):
        provenance.verify_frozen_source(graph, manifest)
