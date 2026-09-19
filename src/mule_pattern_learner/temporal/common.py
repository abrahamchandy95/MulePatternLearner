"""Small shared utilities with no training, staging or supervision dependencies."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path


def timestamp(date: str) -> int:
    parsed = datetime.fromisoformat(date)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_score(value: str, seed: int, purpose: str) -> float:
    value_hash = hashlib.sha256(f"{purpose}:{seed}:{value}".encode()).digest()
    return int.from_bytes(value_hash[:8], "big") / 2**64
