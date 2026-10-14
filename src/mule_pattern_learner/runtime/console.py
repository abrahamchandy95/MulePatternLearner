"""The console: a short line for each event a person needs to follow, and nothing more.

show prints a line. plural, duration and brief are how the lines write counts, durations
and messages, and retry_line is the line of a request the executor retries.
"""

from __future__ import annotations

import sys
import threading

_LOCK = threading.Lock()


def show(text: str) -> None:
    """Print text, one line or several, to stdout; threads never interleave their lines."""
    with _LOCK:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def plural(value: int, word: str, words: str | None = None) -> str:
    """A count and its noun: 1 query, 12 queries."""
    return f"{value:,} {word if value == 1 else words or word + 's'}"


def duration(seconds: float) -> str:
    """Seconds as a person reads them: 41 s, 3.1 min, 2.4 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def brief(text: str, length: int = 100) -> str:
    """Text on one line, cut to about ``length`` characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= length else flat[: length - 3].rstrip() + "..."


def _operation(operation: str) -> str:
    """An operation as a line names it: "fetch_subgraphs (512 keys)" is
    "fetch_subgraphs, 512 keys", so the reason is the line's only parenthesis."""
    name, _, detail = operation.partition(" (")
    return f"{name}, {detail.removesuffix(')')}" if detail else name


def retry_line(failure: str, operation: str, attempt: int, retry_in_s: float, reason: str) -> str:
    """The line of a retried request: what failed and why, the attempt and the wait.

    ``failure`` is the executor's failure class: "availability", "server_timeout" or
    "deterministic".
    """
    match failure:
        case "availability":
            what = "TigerGraph is not answering yet"
        case "server_timeout":
            what = f"TigerGraph timed out on {_operation(operation)}"
        case _:
            what = f"TigerGraph failed {_operation(operation)}"
    wait = duration(retry_in_s)
    return f"{what} ({reason or 'no answer'}): attempt {attempt}, retrying in {wait}"
