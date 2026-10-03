"""emit(): the one structured record of each event, and its short line on the console.

Every record is one JSON object whose "event" key names what happened; warnings are
events too (warn), and so are the executor's retries. While a command records its events
(``recording`` with the events.jsonl of a run, a prepared dataset, a suite or a study),
each record is appended to that file in full, led by the UTC time it was emitted
("time"). What stdout shows is the event's short line for a person (runtime.console), or
nothing: running totals, batch counts and the progress of scoring are in the files only.
The command line records each command in results/events.jsonl (paths.command_events),
naming the command on each record there ("command"), so a record emitted before a run, a
dataset or a study records, such as a connection's retries before preparation, is kept
there; only one emitted while nothing records at all, as when Python calls a use case,
is shown on the console alone.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
import contextlib
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any

from ..artifacts import append_event, event_line
from .console import show_event

_LOCK = threading.Lock()
# The events.jsonl files being recorded, the innermost last, each with the command its
# records name (None when they name none).
_RECORDING: list[tuple[Path, str | None]] = []


@contextlib.contextmanager
def recording(events: Path, command: str | None = None) -> Generator[None]:
    """Append the record of every event emitted during the block to events, an events.jsonl.

    Each record starts with the UTC time it was emitted ("time", ISO 8601 to the second)
    and, when the block is given a command, the command that emitted it ("command"), as
    the command line gives it for results/events.jsonl.
    """
    entry = (events, command)
    with _LOCK:
        _RECORDING.append(entry)
    try:
        yield
    finally:
        with _LOCK:
            _RECORDING.remove(entry)


def _stamped(record: Mapping[str, Any], command: str | None) -> dict[str, Any]:
    """record as events.jsonl keeps it: led by the UTC time, and the command if one is given."""
    named = {} if command is None else {"command": command}
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **named, **record}


def emit(record: Mapping[str, Any]) -> None:
    """Append record to the recorded events.jsonl as one JSON line, and show its console line.

    A record without an "event" key is refused, so every line says what it records, and
    so is one JSON cannot hold (NaN or infinity), before anything is written or shown.
    """
    if not isinstance(record.get("event"), str):
        raise ValueError(f"An event record needs an event name: {dict(record)}")
    with _LOCK:
        events, command = _RECORDING[-1] if _RECORDING else (None, None)
        line = event_line(_stamped(record, command))
        if events is not None:
            append_event(events, line)
        show_event(record)


def warn(warning: str, message: str) -> None:
    """Emit a warning: {"event": "warning", "warning": its name, "message": message}."""
    emit({"event": "warning", "warning": warning, "message": message})
