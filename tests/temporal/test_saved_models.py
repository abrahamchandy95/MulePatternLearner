"""Models saved before the layered restructure load and score as they did.

The fixtures were written by commit 9925f69, whose training and checkpoint code is the
code the saved models on the CUDA host come from. They are the built-in run, the tabular
variant and the no_fourier variant, each trained for two steps with 8 hidden units on the
fake graph. Their configurations hold keys that the restructure retires, such as
context_storage, evaluation_protocol, label_policy and variant. The literals are the
scores that commit's TemporalPredictor gave eight test accounts; this code must give the
same scores. Floating point rounding differs between machines, hence the tolerance.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mule_pattern_learner.temporal.common import cutoff_ms
from mule_pattern_learner.temporal.live.checkpoint import ModelCheckpoint
from mule_pattern_learner.temporal.live.contract import ContextKey
from mule_pattern_learner.temporal.live.predictor import TemporalPredictor
from temporal_fakes import FakeExecutor, neighbourhood

FIXTURES = Path(__file__).parent / "fixtures" / "saved_models"
RELATIVE = 1e-5
# Test-partition accounts of the fake scope, at the fake graph's test cutoff sequence.
ACCOUNTS = ("S0004", "S0009", "S0014", "S0019", "S0024", "S0029", "S0034", "S0039")
SCORES = {
    "built_in": [
        0.4991507417307398,
        0.5007370076392473,
        0.5002217446308491,
        0.4998289155840587,
        0.500896578159351,
        0.5003809564941073,
        0.4991540060069826,
        0.49990310473488864,
    ],
    "tabular": [
        0.5340324079266534,
        0.5216267707009064,
        0.5216267707009064,
        0.5340324079266534,
        0.5216267707009064,
        0.5340324079266534,
        0.5340324079266534,
        0.5216267707009064,
    ],
    "no_fourier": [
        0.568559441682482,
        0.5614372077809549,
        0.5589734916980292,
        0.5617949737483596,
        0.5556401659101458,
        0.5661534213136434,
        0.5634052456284869,
        0.5619818620262174,
    ],
}


@pytest.mark.parametrize("name", sorted(SCORES))
def test_models_saved_before_the_restructure_score_as_they_did(name: str) -> None:
    saved = ModelCheckpoint.load(FIXTURES / f"{name}.pt")
    keys = [
        ContextKey("Account", account, 103, cutoff_ms("2025-01-01"), saved.config["scope_id"], 3)
        for account in ACCOUNTS
    ]
    predictor = TemporalPredictor(saved, executor=FakeExecutor(factory=neighbourhood))
    try:
        frame = predictor.predict(keys)
    finally:
        predictor.contexts.close()
    assert frame.account_id.tolist() == list(ACCOUNTS)
    for have, want in zip(frame.score.tolist(), SCORES[name], strict=True):
        assert math.isclose(have, want, rel_tol=RELATIVE), (name, have, want)
