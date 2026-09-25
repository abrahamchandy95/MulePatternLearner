"""Small shared utilities with no training, staging or supervision dependencies."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path


def timestamp(date: str) -> int:
    parsed = datetime.fromisoformat(date)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def cutoff_ms(date: str) -> int:
    """The millisecond before a calendar cutoff: the last one whose history is visible."""
    return timestamp(date) - 1


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash64(*parts: object) -> int:
    """First 8 bytes of sha256("part:part:..."), big-endian: a machine-independent hash."""
    value = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "big")


def stable_score(value: str, seed: int, purpose: str) -> float:
    return hash64(purpose, seed, value) / 2**64
