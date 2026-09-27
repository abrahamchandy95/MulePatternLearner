"""The evaluation use cases connect only once their inputs passed their checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.config import TransportConfig
from mule_pattern_learner.evaluation.truth import ParquetEvaluationTruth
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.testing.builders import base_config, checkpoint, prepared_dataset
from mule_pattern_learner.tigergraph.oracle import GraphEvaluationTruth


def test_final_audit_connects_after_its_checks_and_reads_truth_on_that_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    dataset, _, _ = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    checkpoint(run.model, config, dataset)
    executor = SimpleNamespace()
    connected: list[TransportConfig] = []
    verified: list[Any] = []

    def connect(transport: TransportConfig) -> Any:
        connected.append(transport)
        return executor

    def audit(audited: RunPaths, truth: Any, **options: Any) -> dict[str, Any]:
        return {"run": audited, "truth": truth, **options}

    monkeypatch.setattr(pipeline_evaluate, "connect", connect)
    monkeypatch.setattr(pipeline_evaluate, "verify_frozen_source", lambda e, m: verified.append(e))
    monkeypatch.setattr(pipeline_evaluate, "evaluate_final_population", audit)
    existing = run.audit_metrics("test")
    existing.parent.mkdir()
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        pipeline_evaluate.final_audit(run, None)
    assert connected == [] and verified == []
    existing.unlink()
    result = pipeline_evaluate.final_audit(run, None)
    # The checkpoint's retry budgets, the frozen source checked, the graph's truth on it.
    assert connected == [config.transport]
    assert verified == [executor]
    assert result["scope"].executor is executor and result["fetcher"].executor is executor
    truth = result["truth"]
    assert isinstance(truth, GraphEvaluationTruth) and truth.executor is executor
    assert result["dataset"] == dataset and result["run"] == run
    supplied = pipeline_evaluate.final_audit(run, tmp_path / "t.parquet")
    assert isinstance(supplied["truth"], ParquetEvaluationTruth)
