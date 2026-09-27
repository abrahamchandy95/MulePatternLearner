"""Stable hashes: of JSON values, of strings and of files.

There are several because each feeds a value that is already persisted: `fingerprint`
names configurations, plans and preparations, `hash64` seeds the per-step draws and
the reservoir ranks (`stable_score`), `stable_hash` keys the sampler's evaluation
draws, and `digest` names files. Changing any of them changes recorded values or
seeded draws.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash64(*parts: object) -> int:
    """First 8 bytes of sha256("part:part:..."), big-endian: a machine-independent hash."""
    value = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "big")


def stable_score(value: str, seed: int, purpose: str) -> float:
    return hash64(purpose, seed, value) / 2**64


def stable_hash(text: str) -> int:
    """64-bit hash that is identical across processes, machines and Python versions."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
