"""The salts keep the values the datasets and runs before the restructure drew with."""

from __future__ import annotations

import hashlib

from mule_pattern_learner.contract.fingerprints import stable_score
from mule_pattern_learner.contract.salts import RESERVOIR_SALT, STEP_SALT


def test_the_salts_keep_their_persisted_values() -> None:
    assert RESERVOIR_SALT == "marginal_cohort"
    assert STEP_SALT == "temporal_live_step"
    # A reservoir rank is the hash of the salt, the seed and the account.
    expected = hashlib.sha256(b"marginal_cohort:42:A0001").digest()
    assert stable_score("A0001", 42, RESERVOIR_SALT) == int.from_bytes(expected[:8], "big") / 2**64
