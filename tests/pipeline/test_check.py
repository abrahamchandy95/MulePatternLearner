"""mule check: read-only readiness, then one batch and one training step on the fakes."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.config import RunConfig, TransportConfig
from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.server import (
    CUTOFF_QUERY,
    GRAPH_NAME,
    SCOPE_VERTEX,
    TRAINING_QUERY_FILES,
)
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.pipeline import check as pipeline_check
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    FrameObservedLabels,
    example_config,
    neighbourhood,
    scoped_accounts,
    supplied_labels,
)
from mule_pattern_learner.testing.fake_connection import RecordingExecutor
from mule_pattern_learner.testing.fake_connection import executor as recording_executor
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph import gsql_text
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs
from mule_pattern_learner.tigergraph.scope import TigerGraphScope

CONFIG = example_config(training={"batch_size": 32}, sampler={"fanouts": [8, 2]})


def graph_server(*, stale: str | None = None, scope_vertex: bool = True) -> SimpleNamespace:
    """The schema and query listing of a graph whose training queries are installed.

    ``stale`` names a query whose installed text differs from the repository.
    """
    queries = gsql_text.repository_queries(TRAINING_QUERY_FILES)
    shown = {name: text for name, (_, text) in queries.items()}
    if stale is not None:
        shown[stale] = shown[stale].replace("{", "{ INT stale_marker = 0;", 1)
    endpoints: dict[str, dict[str, Any]] = {
        f"GET /query/{GRAPH_NAME}/{name}": {
            "enabled": True,
            "parameters": {
                name: {} for name in gsql_text.parameter_names(text) | {"query", "read_committed"}
            },
        }
        for name, (_, text) in queries.items()
    }
    vertices = [{"Name": "Account"}, *([{"Name": SCOPE_VERTEX}] if scope_vertex else [])]

    def gsql(text: str) -> str:
        assert "SHOW QUERY" in text, "the check must not run GSQL that writes"
        return shown.get(text.rsplit(" ", 1)[1], "Query not found")

    def schema(force: bool) -> dict[str, Any]:
        return {"VertexTypes": vertices}

    return SimpleNamespace(gsql=gsql, getInstalledQueries=lambda: endpoints, getSchema=schema)


def connecting(server: SimpleNamespace) -> Callable[[TransportConfig], RecordingExecutor]:
    """A connect() that reaches the fake server."""

    def connect(transport: TransportConfig) -> RecordingExecutor:
        return recording_executor(server)

    return connect


def prepared(data: Path, config: RunConfig) -> tuple[DatasetPaths, FakeTigerGraph]:
    """config's dataset of the fake graph, prepared in its directory under data."""
    fake = FakeTigerGraph(factory=neighbourhood, hubs=[("N3", 101)], population=scoped_accounts())
    dataset = DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    prepare(
        config,
        UNIT_SOURCE,
        dataset,
        {"Account": 1000},
        FrameObservedLabels(supplied_labels()),
        scope=TigerGraphScope(fake),
        cutoffs=TigerGraphCutoffs(fake),
        hub_reader=TigerGraphHubs(fake),
    )
    return dataset, fake


def test_a_ready_graph_gets_one_batch_and_one_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    dataset, fake = prepared(data, CONFIG)
    monkeypatch.setattr(pipeline_check, "connect", connecting(graph_server()))
    opened: list[DatasetPaths] = []

    def open_source(path: DatasetPaths, manifest: dict[str, Any], config: RunConfig) -> Any:
        assert config == CONFIG
        opened.append(path)
        return ContextSource(
            TigerGraphContextFetcher(fake),
            plan=extraction_plan(config.feature_plan()),
            sampler=config.sampler,
        )

    monkeypatch.setattr(pipeline_check, "open_context_source", open_source)
    report = pipeline_check.check(CONFIG, data)
    assert report["status"] == "ready" and report["problems"] == [] and opened == [dataset]
    assert report["graph"] == GRAPH_NAME and report["scope_schema"] == "present"
    queries = gsql_text.repository_queries(TRAINING_QUERY_FILES)
    assert report["queries"] == {"up_to_date": list(queries), "stale": {}}
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


def test_a_graph_that_is_not_ready_is_reported_without_a_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = graph_server(stale=CUTOFF_QUERY, scope_vertex=False)
    monkeypatch.setattr(pipeline_check, "connect", connecting(server))

    def refuse(*_: object) -> None:
        pytest.fail("opened a source")

    monkeypatch.setattr(pipeline_check, "open_context_source", refuse)
    report = pipeline_check.check(CONFIG, tmp_path / "data")
    assert report["status"] == "not_ready" and "first_step" not in report
    assert report["scope_schema"] == "missing" and report["dataset"] is None
    assert report["queries"]["stale"] == {CUTOFF_QUERY: ["differs from repository source"]}
    assert CUTOFF_QUERY not in report["queries"]["up_to_date"]
    assert len(report["problems"]) == 3
    assert any("mule install" in problem for problem in report["problems"])
    assert any("mule train" in problem for problem in report["problems"])
