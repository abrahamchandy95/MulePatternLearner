"""The weight average warms up and restores the raw weights."""

from __future__ import annotations

import pytest
import torch

from mule_pattern_learner.training import averaging


def test_weight_average_warms_up_and_restores_the_raw_weights() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    average = averaging.WeightAverage(model, 0.99)
    with torch.no_grad():
        model.weight.fill_(1.0)
    average.update(model)  # warm-up decay min(0.99, 2 / 11)
    expected = 1 - 2 / 11
    assert average.state["weight"].flatten().tolist() == pytest.approx([expected] * 2)
    with average.applied(model):
        assert model.weight.flatten().tolist() == pytest.approx([expected] * 2)
    assert model.weight.flatten().tolist() == [1.0, 1.0]
    restored = averaging.WeightAverage(model, 0.99)
    restored.load(average.saved())
    assert restored.updates == 1 and torch.equal(restored.state["weight"], average.state["weight"])
