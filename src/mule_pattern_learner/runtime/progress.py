"""emit(): the one structured line a command prints for each event.

Every line is one JSON object on stdout whose "event" key names what happened; warnings
are events too (warn), and so are the executor's retries. While a run records its
events (``recording`` with the run's events.jsonl), each line is appended to that file
as well, so the file keeps what the commands printed for the run. Lines printed before
a run directory exists, such as those of preparation, go to stdout only.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
import contextlib
from pathlib import Path
import threading
from typing import Any

from ..artifacts import append_event, event_line

_LOCK = threading.Lock()
# The events.jsonl files being recorded, the innermost last.
_RECORDING: list[Path] = []


@contextlib.contextmanager
def recording(events: Path) -> Generator[None]:
    """Append every line emitted during the block to events, a run's events.jsonl."""
    with _LOCK:
        _RECORDING.append(events)
    try:
        yield
    finally:
        with _LOCK:
            _RECORDING.remove(events)


def emit(record: Mapping[str, Any]) -> None:
    """Print record as one JSON line, and append it to the recorded run's events.jsonl.

    A record without an "event" key is refused, so every line says what it records.
    """
    if not isinstance(record.get("event"), str):
        raise ValueError(f"An event record needs an event name: {dict(record)}")
    line = event_line(record)
    with _LOCK:
        print(line, flush=True)
        if _RECORDING:
            append_event(_RECORDING[-1], line)


def warn(warning: str, message: str) -> None:
    """Emit a warning: {"event": "warning", "warning": its name, "message": message}."""
    emit({"event": "warning", "warning": warning, "message": message})
