"""One structured line per event, recorded in the current run's events.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mule_pattern_learner.artifacts import read_events
from mule_pattern_learner.runtime.progress import emit, recording, warn


def test_lines_are_printed_and_recorded_only_while_a_run_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, other = tmp_path / "run.jsonl", tmp_path / "other.jsonl"
    emit({"event": "prepare"})
    with recording(run):
        emit({"event": "start", "step": 0})
        with recording(other):
            emit({"event": "inner"})
        emit({"event": "complete"})
    emit({"event": "after"})
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["event"] for e in printed] == ["prepare", "start", "inner", "complete", "after"]
    assert read_events(run) == [{"event": "start", "step": 0}, {"event": "complete"}]
    assert read_events(other) == [{"event": "inner"}]


def test_a_line_that_is_not_json_or_names_no_event_is_refused(tmp_path: Path) -> None:
    with recording(tmp_path / "events.jsonl"):
        with pytest.raises(ValueError):
            emit({"event": "train", "loss": float("nan")})
        with pytest.raises(ValueError, match="needs an event name"):
            emit({"scope": "s", "created": True})
    assert not (tmp_path / "events.jsonl").exists()


def test_warnings_are_events(tmp_path: Path) -> None:
    with recording(tmp_path / "events.jsonl"):
        warn("hub_stubs", "hub children become stubs")
    assert read_events(tmp_path / "events.jsonl") == [
        {"event": "warning", "warning": "hub_stubs", "message": "hub children become stubs"}
    ]
