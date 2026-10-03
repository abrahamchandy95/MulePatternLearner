"""Preparation from the graph's scope, and the visibility phase each split is read in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.server import CONTEXT_QUERY
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import DatasetPaths, RunPaths
from mule_pattern_learner.testing.builders import UNIT_SOURCE, scoped_accounts, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader
from mule_pattern_learner.training.trainer import train


def test_strict_preparation_and_nnpu_use_the_correct_phase_end_to_end(tmp_path: Path) -> None:
    cfg = unit_config(
        scope={"id": "unit_strict"},
        dataset={"seed_limits": {"train": 10, "validation": 10, "test": 10}},
    )
    rows = scoped_accounts()
    run = RunPaths(tmp_path / "run")
    phases = []

    def strict(name: str, params: dict[str, Any]) -> None:
        if name == CONTEXT_QUERY:
            assert params["scope_id"] == "unit_strict"
            phases.append(params["visibility_phase"])
            if params["visibility_phase"] == 3:
                assert run.model.exists(), "Test evaluation happened before the model froze"

    executor = FakeTigerGraph(last_visible=lambda index, ms: 100, population=rows, before=strict)
    dataset = DatasetPaths(tmp_path / "dataset")
    manifest = prepare(
        cfg,
        UNIT_SOURCE,
        dataset,
        {"Account": len(rows)},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(executor),
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
    )
    assert manifest["status"] == "ready" and not executor.requested
    source = ContextSource(
        TigerGraphContextFetcher(executor),
        plan=extraction_plan(cfg.feature_plan()),
        sampler=cfg.sampler,
    )
    result = train(cfg, dataset, run, contexts=source)
    assert set(phases) == {1, 2, 3}
    assert result["known_mules"] == {"train": 20, "validation": 20, "test": 20}
    assert result["evaluation_protocol"] == "strict_inductive"
