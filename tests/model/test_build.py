"""Probabilities are float64, so the highest scores do not tie."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mule_pattern_learner.model.build import probabilities_from_logits


def test_float64_probabilities_separate_logits_that_tie_in_float32() -> None:
    logits = torch.tensor([11.0, 11.004, 17.0, 20.0, 23.0], requires_grad=True)
    # The former float32 sigmoid: the first two tie, and the last three are exactly 1.
    single = torch.sigmoid(logits).detach().numpy()
    assert single[0] == single[1] and (single[2:] == 1).all()
    scores = probabilities_from_logits(logits)
    assert scores.dtype == np.float64 and (np.diff(scores) > 0).all() and (scores < 1).all()
    # The sigmoid of each float32 logit, to float64 precision.
    exact = 1 / (1 + np.exp(-logits.detach().numpy().astype(np.float64)))
    assert scores == pytest.approx(exact, rel=1e-15, abs=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_float64_probabilities_from_mps_logits() -> None:
    # MPS has no float64, so the logits must reach the CPU before the cast.
    logits = torch.tensor([17.0, 20.0], device="mps")
    scores = probabilities_from_logits(logits)
    assert scores.dtype == np.float64 and scores[0] < scores[1] < 1
