"""Every event's full record in the recorded events.jsonl, and its short line on stdout."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mule_pattern_learner.artifacts import read_events
from mule_pattern_learner.runtime.progress import emit, raised_in, recording, warn
from mule_pattern_learner.testing.builders import recorded_events


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
    assert recorded_events(run) == [scored, audited]
    assert recorded_events(other) == [{"event": "an_event_without_a_line", "detail": [1, 2]}]
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
    assert recorded_events(tmp_path / "events.jsonl") == [
        {"event": "warning", "warning": "hub_stubs", "message": "hub children become stubs"}
    ]


def test_each_record_is_led_by_its_time_and_the_command_a_block_names(tmp_path: Path) -> None:
    commands, run = tmp_path / "events.jsonl", tmp_path / "run.jsonl"
    before = datetime.now(timezone.utc).replace(microsecond=0)
    with recording(commands, command="train"):
        emit({"event": "install", "stale": []})
        with recording(run):
            emit({"event": "epoch", "epoch": 1})
    after = datetime.now(timezone.utc)
    (install,), (epoch,) = read_events(commands), read_events(run)
    # The time comes first, in ISO 8601 UTC to the second; then the command, only in the
    # block that names one, as the command line's results/events.jsonl does.
    assert list(install) == ["time", "command", "event", "stale"]
    assert install["command"] == "train"
    assert list(epoch) == ["time", "event", "epoch"]
    for record in (install, epoch):
        time = datetime.fromisoformat(record["time"])
        assert time.utcoffset() == timedelta(0) and time.microsecond == 0
        assert before <= time <= after


def test_an_error_names_the_innermost_file_that_was_recording_when_it_was_raised(
    tmp_path: Path,
) -> None:
    commands, run = tmp_path / "events.jsonl", tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match="inside") as inner:
        with recording(commands), recording(run):
            raise ValueError("inside the run's block")
    assert raised_in(inner.value) == run
    with pytest.raises(KeyError) as outer:
        with recording(commands):
            raise KeyError("outside it")
    assert raised_in(outer.value) == commands
    # An error no block saw names no file, and the blocks record nothing of their own.
    assert raised_in(ValueError("never recorded")) is None
    assert not commands.exists() and not run.exists()
