"""Stable hashes of JSON values and of strings (files are hashed by artifacts.file_digest).

There are several because each feeds a value that is already persisted: `fingerprint`
names configurations, plans and preparations, `hash64` seeds the per-step draws and
the reservoir ranks (`stable_score`), and `stable_hash` keys the sampler's evaluation
draws. Changing any of them changes recorded values or seeded draws.
"""

from __future__ import annotations

import hashlib
import json


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hash64(*parts: object) -> int:
    """First 8 bytes of sha256("part:part:..."), big-endian: a machine-independent hash."""
    value = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "big")


def stable_score(value: str, seed: int, purpose: str) -> float:
    return hash64(purpose, seed, value) / 2**64


def stable_hash(text: str) -> int:
    """64-bit hash that is identical across processes, machines and Python versions."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
