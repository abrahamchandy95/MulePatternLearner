"""Calendar dates as the millisecond clocks of the graph."""

from datetime import datetime, timezone


def timestamp(date: str) -> int:
    parsed = datetime.fromisoformat(date)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def cutoff_ms(date: str) -> int:
    """The millisecond before a calendar cutoff: the last one whose history is visible."""
    return timestamp(date) - 1
