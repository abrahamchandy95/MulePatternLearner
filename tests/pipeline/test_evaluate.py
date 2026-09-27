"""The audit of a run connects only once its inputs passed their checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.artifacts import read_events
from mule_pattern_learner.config import TransportConfig
from mule_pattern_learner.evaluation.truth import ParquetTruth
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.runtime.progress import emit
from mule_pattern_learner.testing.builders import base_config, prepared_dataset, saved_model
from mule_pattern_learner.tigergraph.oracle import TigerGraphTruth


def test_evaluate_run_connects_after_its_checks_and_reads_truth_on_that_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    dataset, _, _ = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config, dataset)
    executor = SimpleNamespace()
    connected: list[TransportConfig] = []
    verified: list[Any] = []

    def connect(transport: TransportConfig) -> Any:
        connected.append(transport)
        return executor

    def audit(audited: RunPaths, truth: Any, **options: Any) -> dict[str, Any]:
        emit({"event": "audit"})
        return {"run": audited, "truth": truth, **options}

    monkeypatch.setattr(pipeline_evaluate, "connect", connect)
    monkeypatch.setattr(pipeline_evaluate, "verify_frozen_source", lambda e, m: verified.append(e))
    monkeypatch.setattr(pipeline_evaluate, "audit", audit)
    existing = run.audit_metrics("test")
    existing.parent.mkdir()
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        pipeline_evaluate.evaluate_run(run, data=tmp_path)
    assert connected == [] and verified == []
    existing.unlink()
    # The dataset is the model's own: its dataset id's directory in data.
    result = pipeline_evaluate.evaluate_run(run, data=tmp_path)
    # The model's retry budgets, the frozen source checked, the graph's truth on it.
    assert connected == [config.transport]
    assert verified == [executor]
    assert result["scope"].executor is executor and result["fetcher"].executor is executor
    truth = result["truth"]
    assert isinstance(truth, TigerGraphTruth) and truth.executor is executor
    assert result["dataset"] == dataset and result["run"] == run
    # The lines the audit prints go to the run's events.jsonl.
    assert read_events(run.events) == [{"event": "audit"}]
    reader = ParquetTruth(tmp_path / "t.parquet")
    assert pipeline_evaluate.evaluate_run(run, truth=reader, data=tmp_path)["truth"] is reader
