"""Every event's full record in the recorded events.jsonl, and its short line on stdout."""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner.artifacts import read_events
from mule_pattern_learner.runtime.progress import emit, recording, warn


def test_records_are_kept_whole_while_a_file_records_and_stdout_shows_their_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, other = tmp_path / "run.jsonl", tmp_path / "other.jsonl"
    scored = {"event": "score", "split": "validation", "accounts": 64, "total": 128}
    audited = {
        "event": "audit",
        "split": "test",
        "date": "2025-01-01",
        "accounts": 2012,
        "rejected_accounts": 0,
        "output": "audit/test.json",
    }
    warn("before", "nothing records this one")
    with recording(run):
        emit(scored)
        with recording(other):
            emit({"event": "an_event_without_a_line", "detail": [1, 2]})
        emit(audited)
    emit({"event": "after"})
    # The innermost file gets each record, whole; nothing records outside a block.
    assert read_events(run) == [scored, audited]
    assert read_events(other) == [{"event": "an_event_without_a_line", "detail": [1, 2]}]
    # Stdout has a line for the events a person follows, and none for the others.
    assert capsys.readouterr().out == (
        "Warning: nothing records this one\n"
        "Audited test at 2025-01-01: 2,012 accounts scored, 0 rejected\n"
    )


def test_a_line_that_is_not_json_or_names_no_event_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with recording(tmp_path / "events.jsonl"):
        with pytest.raises(ValueError):
            emit({"event": "warning", "warning": "nan", "message": float("nan")})
        with pytest.raises(ValueError, match="needs an event name"):
            emit({"scope": "s", "created": True})
    assert not (tmp_path / "events.jsonl").exists()
    # Nothing is shown of a refused record either.
    assert capsys.readouterr().out == ""


def test_warnings_are_events(tmp_path: Path) -> None:
    with recording(tmp_path / "events.jsonl"):
        warn("hub_stubs", "hub children become stubs")
    assert read_events(tmp_path / "events.jsonl") == [
        {"event": "warning", "warning": "hub_stubs", "message": "hub children become stubs"}
    ]
