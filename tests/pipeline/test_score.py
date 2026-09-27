"""Scoring the accounts of a file refuses bad inputs and existing outputs before it connects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.artifacts import pending_path
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline import score as pipeline_score
from mule_pattern_learner.testing.builders import base_config, checkpoint


def test_scoring_checks_inputs_and_outputs_then_verifies_the_installed_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    run = RunPaths(tmp_path / "run")
    checkpoint(run.model, config)
    accounts = tmp_path / "new_accounts.txt"
    accounts.write_text("A1\nA2\n")
    executor = SimpleNamespace()
    steps: list[str] = []
    scored: list[tuple[str, list[str], Path]] = []

    def connect(transport: Any) -> Any:
        steps.append("connect")
        return executor

    def score(saved: SavedModel, ids: Any, date: str, path: Path, **options: Any) -> dict[str, Any]:
        steps.append("score")
        scored.append((date, list(ids), path))
        return options

    monkeypatch.setattr(pipeline_score, "connect", connect)
    monkeypatch.setattr(pipeline_score, "verify_sources", lambda e: steps.append("verify"))
    monkeypatch.setattr(pipeline_score, "score_new_accounts", score)
    with pytest.raises(FileNotFoundError):
        pipeline_score.score_accounts(run, tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="ISO date"):
        pipeline_score.score_accounts(run, accounts, "../elsewhere")
    # Another scoring run is writing this output: the run's scores of the file at the
    # model's test cutoff.
    output = run.scores("new_accounts", "2025-01-01")
    output.parent.mkdir(parents=True)
    pending_path(output).write_text("")
    with pytest.raises(FileExistsError):
        pipeline_score.score_accounts(run, accounts)
    assert steps == []
    pending_path(output).unlink()
    result = pipeline_score.score_accounts(run, accounts)
    assert steps == ["connect", "verify", "score"]
    assert scored == [("2025-01-01", ["A1", "A2"], output)]
    # The cutoff clock, the hub registry and the contexts are read on that connection.
    assert {name: port.executor for name, port in result.items()} == {
        "cutoffs": executor,
        "hub_reader": executor,
        "fetcher": executor,
    }
    pipeline_score.score_accounts(run, accounts, "2025-02-01")
    assert scored[-1][0] == "2025-02-01" and scored[-1][2].name == "new_accounts_2025-02-01.parquet"
