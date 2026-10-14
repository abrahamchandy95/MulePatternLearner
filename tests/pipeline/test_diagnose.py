"""The graph reads of `mule diagnose`: the frozen source checked first, nothing installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig, TransportConfig
from mule_pattern_learner.contract.server import TRUTH_QUERY
from mule_pattern_learner.diagnostics.study import ANALYSES
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import diagnose as pipeline_diagnose
from mule_pattern_learner.pipeline.connect import Session
from mule_pattern_learner.pipeline.diagnose import TigerGraphStudyReader
from mule_pattern_learner.pipeline.train import BASELINE_RUN
from mule_pattern_learner.testing.builders import reveal_inputs, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, prepared_graph


def test_the_reader_checks_the_frozen_source_before_any_read(
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

    monkeypatch.setattr(pipeline_diagnose, "verify_frozen_source", verify)
    reader = TigerGraphStudyReader(config, dataset, Session(config.transport))
    # The reveal's parameters need no graph.
    assert reader.reveal_parameters()["apply"] is False and steps == []
    assert reader.reveal_inputs() == reveal_inputs() and steps == ["verified"]
    truth = reader.oracle()
    assert truth.read() is truth.read() and graph.names().count(TRUTH_QUERY) == 1
    # The context source has the dataset's disk tier and is the caller's to close.
    contexts = reader.contexts()
    assert contexts.disk is not None and contexts.disk.cache.directory == dataset.contexts
    contexts.close()
    # The reads verify the frozen source once and install nothing.
    assert steps == ["verified"] and graph.writes == []


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
