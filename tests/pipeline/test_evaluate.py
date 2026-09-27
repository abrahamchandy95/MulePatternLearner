"""The evaluation use cases connect only once their inputs passed their checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.evaluation.truth import ParquetEvaluationTruth
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.testing.builders import base_config, checkpoint, prepared_dataset
from mule_pattern_learner.tigergraph.oracle import GraphEvaluationTruth


def test_final_audit_connects_after_its_checks_and_reads_truth_on_that_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    dataset, _, _ = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = checkpoint(tmp_path / "model.pt", config, dataset / "manifest.json")
    executor = SimpleNamespace()
    connected: list[dict[str, Any]] = []
    verified: list[Any] = []

    def connect(settings: dict[str, Any]) -> Any:
        connected.append(settings)
        return executor

    def audit(saved: Any, truth: Any, output: Path, **options: Any) -> dict[str, Any]:
        return {"truth": truth, "output": output, **options}

    monkeypatch.setattr(pipeline_evaluate, "connect", connect)
    monkeypatch.setattr(pipeline_evaluate, "verify_frozen_source", lambda e, m: verified.append(e))
    monkeypatch.setattr(pipeline_evaluate, "evaluate_final_population", audit)
    existing = tmp_path / "final.json"
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        pipeline_evaluate.final_audit(model, None, existing)
    assert connected == [] and verified == []
    result = pipeline_evaluate.final_audit(model, None, tmp_path / "audit.json")
    # The checkpoint's retry budgets, the frozen source checked, the graph's truth on it.
    assert [settings["scope_id"] for settings in connected] == [config["scope_id"]]
    assert verified == [executor] and result["executor"] is executor
    assert (
        isinstance(result["truth"], GraphEvaluationTruth) and result["truth"].executor is executor
    )
    assert result["dataset"] == dataset
    supplied = pipeline_evaluate.final_audit(model, tmp_path / "t.parquet", tmp_path / "b.json")
    assert isinstance(supplied["truth"], ParquetEvaluationTruth)
