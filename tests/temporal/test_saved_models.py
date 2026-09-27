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
from mule_pattern_learner.temporal.live.contract import (
    DEFAULT_GROUPS,
    ContextKey,
    FeaturePlan,
    contract_fingerprint,
    fingerprint,
)
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


def test_the_built_in_inputs_keep_their_recorded_fingerprints() -> None:
    assert contract_fingerprint() == (
        "530e46c91b07b254d38722e57117b917761d2b68176e7eb1cb5e2e9de32307da"
    )
    assert FeaturePlan().fingerprint() == (
        "9cf8c7606ce403da4e74757799aa77cfa6fdcd0736686b23c38bce592fd08238"
    )
    assert FeaturePlan(architecture="summary").fingerprint() == (
        "77c38ffb11af48f5485d4e1efc9cd92e826990d3650738864b4a18cfc478be58"
    )


def test_models_whose_columns_moved_are_refused(tmp_path: Path) -> None:
    # Columns follow registry order now; before, the window groups' columns sat elsewhere.
    plan = FeaturePlan((*DEFAULT_GROUPS, "rolling_windows"))
    recorded = fingerprint(
        {"contract": contract_fingerprint(), "groups": sorted(plan.groups), "architecture": "split"}
    )
    saved = ModelCheckpoint(tmp_path / "model.pt", {"input_fingerprint": recorded})
    with pytest.raises(ValueError, match="input groups"):
        saved.check_inputs(plan)
