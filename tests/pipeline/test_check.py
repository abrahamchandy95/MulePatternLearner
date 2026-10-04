"""mule check: read-only readiness, then one batch and one training step on the fakes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.artifacts import read_json
from mule_pattern_learner.config import RunConfig
from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.fingerprints import fingerprint
from mule_pattern_learner.contract.server import (
    CONTEXT_QUERY,
    CUTOFF_QUERY,
    GRAPH_NAME,
    RETIRED_QUERIES,
    TRAINING_QUERY_FILES,
)
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import DatasetPaths, check_report
from mule_pattern_learner.pipeline import check as pipeline_check
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    neighbourhood,
    scoped_accounts,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import EARLIER_SCOPE_TYPES, FakeTigerGraph
from mule_pattern_learner.tigergraph import gsql_text
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader

CONFIG = unit_config(training={"batch_size": 32}, sampler={"fanouts": [8, 2]})


def prepared(data: Path, config: RunConfig) -> tuple[DatasetPaths, FakeTigerGraph]:
    """config's dataset of the fake graph, prepared in its directory under data."""
    fake = FakeTigerGraph(factory=neighbourhood, hubs=[("N3", 101)], population=scoped_accounts())
    dataset = DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    prepare(
        config,
        UNIT_SOURCE,
        dataset,
        {"Account": 1000},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(fake),
        cutoffs=TigerGraphCutoffReader(fake),
        hub_reader=TigerGraphHubReader(fake),
    )
    return dataset, fake


def test_a_ready_graph_gets_one_batch_and_one_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    dataset, fake = prepared(data, CONFIG)
    monkeypatch.setattr(pipeline_check, "connect", lambda transport: fake)
    opened: list[DatasetPaths] = []

    def open_source(
        path: DatasetPaths, manifest: dict[str, Any], config: RunConfig, *, cached: bool = True
    ) -> Any:
        # Without the context cache, so the batch runs the installed context query.
        assert config == CONFIG and not cached
        opened.append(path)
        return ContextSource(
            TigerGraphContextFetcher(fake),
            plan=extraction_plan(config.feature_plan()),
            sampler=config.sampler,
        )

    monkeypatch.setattr(pipeline_check, "open_context_source", open_source)
    results = tmp_path / "results"
    report = pipeline_check.check(CONFIG, data, results)
    assert report["status"] == "ready" and report["problems"] == [] and opened == [dataset]
    # The whole report is the one results/check.json holds.
    assert read_json(check_report(results)) == report
    assert report["graph"] == GRAPH_NAME and report["scope_schema"] == "present"
    queries = gsql_text.repository_queries(TRAINING_QUERY_FILES)
    assert report["queries"] == {"up_to_date": list(queries), "stale": {}, "retired": []}
    assert report["dataset"] == dataset.root.name and report["graph_writes"] == 0
    step = report["first_step"]
    assert step["status"] == "passed" and step["roots"] == step["accepted_roots"] == 32
    assert step["batch"]["sampler_backend"] == "torch"
    assert step["batch"]["stub_children"] > 0 and step["batch"]["first_edges"] > 0
    assert step["context_requests"] > 0 and step["rest_calls"] == 0  # fakes count none
    assert step["loss"] > 0 and step["train_step_seconds"] > 0
    assert math.isfinite(step["objective"])
    # Digests of every batch tensor (test_golden_run pins their values).
    digests = step["tensor_digests"]
    assert digests["root_positions"]["shape"] == [32] and len(digests["x"]["sha256"]) == 64
    # And one digest of their bytes.
    shas = {name: digest["sha256"] for name, digest in digests.items()}
    assert step["batch_digest"] == fingerprint(shas)


def test_a_graph_that_is_not_ready_is_reported_without_a_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = gsql_text.repository_queries(TRAINING_QUERY_FILES)
    installed = {name: text for name, (_, text) in queries.items()}
    server = FakeTigerGraph(
        queries={**installed, RETIRED_QUERIES[0]: installed[CONTEXT_QUERY]},
        stale=[CUTOFF_QUERY],
        scope_types=None,
    )
    monkeypatch.setattr(pipeline_check, "connect", lambda transport: server)

    def refuse(*_: object) -> None:
        pytest.fail("opened a source")

    monkeypatch.setattr(pipeline_check, "open_context_source", refuse)
    report = pipeline_check.check(CONFIG, tmp_path / "data", tmp_path / "results")
    assert report["status"] == "not_ready" and "first_step" not in report
    assert read_json(check_report(tmp_path / "results")) == report
    assert report["scope_schema"] == "missing" and report["dataset"] is None
    assert report["queries"]["stale"] == {CUTOFF_QUERY: ["differs from repository source"]}
    # A retired query still installed is reported, and dropping it is left to mule install.
    assert report["queries"]["retired"] == [RETIRED_QUERIES[0]] and server.writes == []
    # Nothing but reads: SHOW QUERY, the endpoint listing and the schema.
    assert server.calls == []
    assert CUTOFF_QUERY not in report["queries"]["up_to_date"]
    assert len(report["problems"]) == 3
    assert any("mule install" in problem for problem in report["problems"])
    assert any("mule train" in problem for problem in report["problems"])


def test_an_outdated_scope_type_is_reported_with_what_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Scope types that predate the split shares: `mule install` replaces them while no
    # scope vertex uses them, and refuses while one does.
    graphs = [
        FakeTigerGraph(scope_types=EARLIER_SCOPE_TYPES),
        FakeTigerGraph(scope_types=EARLIER_SCOPE_TYPES, scopes={"strict_mule_v2": {}}),
    ]

    def checked(graph: FakeTigerGraph) -> dict[str, Any]:
        def connect(transport: object) -> FakeTigerGraph:
            return graph

        monkeypatch.setattr(pipeline_check, "connect", connect)
        report = pipeline_check.check(CONFIG, tmp_path / "data", tmp_path / "results")
        assert graph.writes == [] and graph.calls == []
        return report

    reports = [checked(graph) for graph in graphs]
    differences = ["Temporal_Training_Scope lacks train_share, validation_share, test_share"]
    for report, scopes in zip(reports, (0, 1), strict=True):
        assert report["status"] == "not_ready" and report["scope_schema"] == "outdated"
        assert report["scope_outdated"] == {"differences": differences, "scopes": scopes}
    empty, used = (report["problems"][0] for report in reports)
    assert empty == "the scope vertex type is outdated; `mule install` replaces it"
    assert used.startswith(
        "the scope vertex type is outdated and the graph holds 1 scope vertex, which "
        "`mule install` refuses to delete; clear the graph's data and load it again"
    )
