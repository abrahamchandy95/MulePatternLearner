"""The stable hashes keep their values: each names a persisted value or seeds a draw."""

from __future__ import annotations

import hashlib

from mule_pattern_learner.contract.fingerprints import (
    fingerprint,
    hash64,
    stable_hash,
    stable_score,
)
from mule_pattern_learner.contract.salts import RESERVOIR_SALT, STEP_SALT


def test_fingerprints_ignore_key_order_and_keep_their_values() -> None:
    value = "721ef82f2d6c0997bffb7a8ab3f40f8fb45b0b52ce2af3afa6b0f05efbdc317f"
    assert fingerprint({"b": [1, 2], "a": "x"}) == fingerprint({"a": "x", "b": [1, 2]}) == value
    assert fingerprint({"a": "x", "b": [2, 1]}) != value


def test_the_draw_hashes_keep_their_values() -> None:
    # hash64 is the first 8 bytes of sha256 of the parts joined by colons.
    step = hash64(STEP_SALT, 7, 1, 2)
    assert step == int.from_bytes(hashlib.sha256(b"temporal_live_step:7:1:2").digest()[:8], "big")
    assert step == 11822582476958647396
    # The reservoir rank of an account, which decides whether a dataset selects it.
    assert stable_score("A0001", 42, RESERVOIR_SALT) == 0.8761228215410102
    assert 0 <= stable_score("A0002", 42, RESERVOIR_SALT) < 1
    # The sampler's evaluation keys.
    assert stable_hash("Account\x1fA0001") == 11936327364855318941
