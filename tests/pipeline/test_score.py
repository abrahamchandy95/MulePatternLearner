"""Scoring new accounts refuses existing outputs before it connects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.artifacts import pending_path
from mule_pattern_learner.pipeline import score as pipeline_score
from mule_pattern_learner.testing.builders import base_config, checkpoint


def test_score_new_checks_outputs_then_verifies_the_installed_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    model = checkpoint(tmp_path / "model.pt", config)
    output = tmp_path / "scores.parquet"
    executor = SimpleNamespace()
    steps: list[str] = []

    def connect(settings: dict[str, Any]) -> Any:
        steps.append("connect")
        return executor

    def score(saved: Any, ids: Any, date: str, path: Path, **options: Any) -> dict[str, Any]:
        steps.append("score")
        return options

    monkeypatch.setattr(pipeline_score, "connect", connect)
    monkeypatch.setattr(pipeline_score, "verify_sources", lambda e: steps.append("verify"))
    monkeypatch.setattr(pipeline_score, "score_new_accounts", score)
    # Another scoring run is writing this output.
    pending_path(output).write_text("")
    with pytest.raises(FileExistsError):
        pipeline_score.score_new(model, iter(["A1"]), "2025-01-01", output)
    assert steps == []
    pending_path(output).unlink()
    result = pipeline_score.score_new(model, iter(["A1"]), "2025-01-01", output)
    assert steps == ["connect", "verify", "score"]
    # The cutoff clock, the hub registry and the contexts are read on that connection.
    assert {name: port.executor for name, port in result.items()} == {
        "cutoffs": executor,
        "hub_reader": executor,
        "fetcher": executor,
    }
