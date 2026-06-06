import math
from typing import cast

import pytest
import torch
from torch import Tensor

from mule_pattern_learner.training.loss import NonNegativePULoss


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _call(loss: NonNegativePULoss, f: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    return cast("tuple[Tensor, Tensor]", loss(f, t))


def test_objective_matches_hand_computation() -> None:
    # logits f, targets t (1=positive, 0=unlabeled), prior pi
    f = torch.tensor([2.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    t = torch.tensor([1, 0, 0, 1], dtype=torch.long)  # positives 0,3; unlabeled 1,2
    pi = 0.3
    _, objective = _call(NonNegativePULoss(prior=pi), f, t)

    raw = [2.0, -1.0, 0.5, -0.5]
    l_pos = [_sigmoid(-x) for x in raw]  # l(+f) = sigmoid(-f)
    l_neg = [_sigmoid(x) for x in raw]  # l(-f) = sigmoid(+f)
    n_p, n_u = 2.0, 2.0
    pos_idx, unl_idx = (0, 3), (1, 2)
    positive_risk = pi * sum(l_pos[i] for i in pos_idx) / n_p
    negative_risk = sum(l_neg[i] for i in unl_idx) / n_u - pi * sum(l_neg[i] for i in pos_idx) / n_p
    expected = positive_risk + negative_risk
    assert objective.item() == pytest.approx(expected, abs=1e-5)


def test_train_equals_objective_when_negative_risk_nonnegative() -> None:
    f = torch.tensor([2.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    t = torch.tensor([1, 0, 0, 1], dtype=torch.long)
    train, objective = _call(NonNegativePULoss(prior=0.3), f, t)
    # here negative_risk > 0, so no correction: train == objective
    assert train.item() == pytest.approx(objective.item(), abs=1e-5)


def test_non_negative_correction_fires() -> None:
    # unlabeled look very negative, positives very positive -> negative_risk < 0
    f = torch.tensor([10.0, 10.0, -10.0, -10.0], dtype=torch.float32)
    t = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    train, objective = _call(NonNegativePULoss(prior=0.5), f, t)
    # correction replaces the objective with gamma * (-negative_risk) > 0
    assert train.item() != pytest.approx(objective.item(), abs=1e-3)
    assert train.item() > 0.0
    assert objective.item() < 0.0


def test_gradients_flow_and_are_finite() -> None:
    f = torch.tensor([0.5, -0.5, 0.2, -0.2], dtype=torch.float32, requires_grad=True)
    t = torch.tensor([1, 0, 0, 1], dtype=torch.long)
    train, _ = _call(NonNegativePULoss(prior=0.3), f, t)
    _ = train.backward()
    grad = f.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all().item())


def test_prior_outside_unit_interval_raises() -> None:
    with pytest.raises(ValueError):
        _ = NonNegativePULoss(prior=1.5)


def test_confident_positives_lower_risk() -> None:
    # higher logits on positives -> lower l(+f) -> lower positive_risk
    t = torch.tensor([1, 1], dtype=torch.long)
    f_low = torch.tensor([0.0, 0.0], dtype=torch.float32)
    f_high = torch.tensor([5.0, 5.0], dtype=torch.float32)
    _, obj_low = _call(NonNegativePULoss(prior=0.3), f_low, t)
    _, obj_high = _call(NonNegativePULoss(prior=0.3), f_high, t)
    assert obj_low.item() > obj_high.item()
