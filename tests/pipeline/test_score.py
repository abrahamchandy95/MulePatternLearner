"""Scoring the accounts of a file refuses bad inputs and existing outputs before it connects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.artifacts import pending_path, read_events
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline import score as pipeline_score
from mule_pattern_learner.runtime.progress import emit
from mule_pattern_learner.testing.builders import RUNTIME_CHANGES, saved_model, unit_config


def test_scoring_checks_inputs_and_outputs_then_verifies_the_installed_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config(RUNTIME_CHANGES)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config)
    accounts = tmp_path / "new_accounts.txt"
    accounts.write_text("A1\nA2\n")
    executor = SimpleNamespace()
    steps: list[str] = []
    scored: list[tuple[str, list[str], Path, Path]] = []

    def connect(transport: Any) -> Any:
        steps.append("connect")
        return executor

    def score(saved: SavedModel, ids: Any, date: str, path: Path, **options: Any) -> dict[str, Any]:
        steps.append("score")
        scored.append((date, list(ids), path, options.pop("rejected_output")))
        # The ids are counted first, for the progress a terminal shows.
        assert options.pop("total") == 2
        emit({"event": "score", "date": date})
        return options

    monkeypatch.setattr(pipeline_score, "connect", connect)
    monkeypatch.setattr(pipeline_score, "verify_sources", lambda e: steps.append("verify"))
    monkeypatch.setattr(pipeline_score, "score_new_accounts", score)
    with pytest.raises(FileNotFoundError):
        pipeline_score.score_accounts(run, tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="ISO date"):
        pipeline_score.score_accounts(run, accounts, "../elsewhere")
    # Another scoring run is writing this output, or its rejected ids exist: the run's
    # scores of the file at the model's test cutoff.
    output = run.scores("new_accounts", "2025-01-01")
    rejected = run.scores_rejected("new_accounts", "2025-01-01")
    output.parent.mkdir(parents=True)
    for path in (pending_path(output), rejected):
        path.write_text("")
        with pytest.raises(FileExistsError):
            pipeline_score.score_accounts(run, accounts)
        path.unlink()
    assert steps == []
    result = pipeline_score.score_accounts(run, accounts)
    assert steps == ["connect", "verify", "score"]
    assert scored == [("2025-01-01", ["A1", "A2"], output, rejected)]
    # The lines scoring prints go to the run's events.jsonl.
    assert read_events(run.events) == [{"event": "score", "date": "2025-01-01"}]
    # The cutoff clock, the hub registry and the contexts are read on that connection.
    contexts = result.pop("contexts")
    contexts.close()
    assert contexts.fetcher.executor is executor and contexts.sampler == config.sampler
    assert {name: port.executor for name, port in result.items()} == {
        "cutoffs": executor,
        "hub_reader": executor,
    }
    pipeline_score.score_accounts(run, accounts, "2025-02-01")["contexts"].close()
    assert scored[-1][0] == "2025-02-01" and scored[-1][2].name == "new_accounts_2025-02-01.parquet"
