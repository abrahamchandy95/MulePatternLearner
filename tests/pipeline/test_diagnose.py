"""The graph reads of `mule diagnose`: the frozen source first, then the analytics install."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.config import TransportConfig
from mule_pattern_learner.contract.server import ANALYTICS_CONTEXT_QUERY, TRUTH_QUERY
from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import diagnose as pipeline_diagnose
from mule_pattern_learner.pipeline.connect import Session
from mule_pattern_learner.pipeline.diagnose import TigerGraphStudyReader
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
