"""The graph reads of `mule diagnose`: the frozen source first, then the analytics install."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig, TransportConfig
from mule_pattern_learner.contract.server import ANALYTICS_CONTEXT_QUERY, TRUTH_QUERY
from mule_pattern_learner.diagnostics.study import ANALYSES
from mule_pattern_learner.paths import REPOSITORY_ROOT, DatasetPaths
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import diagnose as pipeline_diagnose
from mule_pattern_learner.pipeline.connect import Session
from mule_pattern_learner.pipeline.diagnose import TigerGraphStudyReader
from mule_pattern_learner.pipeline.train import BASELINE_RUN
from mule_pattern_learner.testing.builders import reveal_inputs, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, prepared_graph

PACKAGE = REPOSITORY_ROOT / "src" / "mule_pattern_learner"


def test_the_reader_checks_the_frozen_source_before_any_read_and_installs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config()
    graph, dataset = prepared_graph(tmp_path / "data", config, reveal=reveal_inputs())
    steps: list[str] = []

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        return graph

    monkeypatch.setattr(pipeline_connect, "connect", connect)

    def verify(executor: FakeTigerGraph, manifest: dict[str, Any]) -> None:
        steps.append("verified")

    def install(executor: FakeTigerGraph, *, analytics: bool) -> dict[str, Any]:
        assert analytics
        steps.append("installed")
        return {}

    monkeypatch.setattr(pipeline_diagnose, "verify_frozen_source", verify)
    monkeypatch.setattr(pipeline_diagnose, "install", install)
    reader = TigerGraphStudyReader(config, dataset, Session(config.transport))
    # The reveal's parameters need no graph.
    assert reader.reveal_parameters()["apply"] is False and steps == []
    assert reader.reveal_inputs() == reveal_inputs() and steps == ["verified"]
    truth = reader.oracle()
    assert truth.read() is truth.read() and graph.names().count(TRUTH_QUERY) == 1
    fetcher = reader.analytics()
    reader.analytics()
    assert steps == ["verified", "installed"] and fetcher.executor is graph
    # The context source has the dataset's disk tier and is the caller's to close.
    contexts = reader.contexts()
    assert contexts.disk is not None and contexts.disk.cache.directory == dataset.contexts
    contexts.close()
    assert ANALYTICS_CONTEXT_QUERY not in graph.names()


def installs(node: ast.AST) -> bool:
    """Whether a node calls install with the analytics queries."""
    if not isinstance(node, ast.Call):
        return False
    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
    return name == "install" and any(
        keyword.arg == "analytics"
        and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False)
        for keyword in node.keywords
    )


def test_only_the_diagnose_reader_installs_the_analytics_queries() -> None:
    # `mule install` and `mule train` (pipeline.prepare) install without them.
    callers = [
        path.relative_to(PACKAGE).as_posix()
        for path in sorted(PACKAGE.rglob("*.py"))
        if any(installs(node) for node in ast.walk(ast.parse(path.read_text())))
    ]
    assert callers == ["pipeline/diagnose.py"]


def test_diagnose_studies_the_built_in_run_on_its_dataset_with_one_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = DatasetPaths.of("id", tmp_path / "data")
    prepared: list[object] = []
    studied: list[dict[str, Any]] = []

    def prepare(c: RunConfig, *, session: object) -> DatasetPaths:
        assert c is DEFAULT_CONFIG
        prepared.append(session)
        return dataset

    def diagnose(names: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        studied.append({"names": names, **kwargs})
        return {"status": "complete"}

    monkeypatch.setattr(pipeline_diagnose, "prepare_dataset", prepare)
    monkeypatch.setattr(pipeline_diagnose, "diagnose", diagnose)
    assert pipeline_diagnose.diagnose_built_in("drift") == {"status": "complete"}
    (study,) = studied
    assert study["names"] == ("drift",) and study["run"] == BASELINE_RUN
    assert study["config"] is DEFAULT_CONFIG and study["dataset"] == dataset
    # The study's graph reads use the session the dataset was prepared on.
    (session,) = prepared
    assert study["graph"].session is session and study["graph"].dataset == dataset
    pipeline_diagnose.diagnose_built_in(None)
    assert studied[-1]["names"] == ANALYSES
