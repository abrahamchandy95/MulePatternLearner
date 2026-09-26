import math
from typing import cast

import pytest
import torch
from torch import Tensor

from mule_pattern_learner.temporal.loss import NonNegativePULoss


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


def test_grouped_ap_interval_bootstraps_whole_groups() -> None:
    import numpy as np

    from mule_pattern_learner.temporal.metrics import grouped_ap_interval

    y = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.4, 0.2, 0.7])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    low, high = grouped_ap_interval(y, scores, groups, seed=7, draws=100) or (None, None)
    assert low is not None and high is not None and 0.0 <= low <= high <= 1.0
    # Same seed, same interval; one group or no positive has no interval.
    assert grouped_ap_interval(y, scores, groups, seed=7, draws=100) == [low, high]
    assert grouped_ap_interval(y, scores, np.zeros(8, dtype=np.int64)) is None
    assert grouped_ap_interval(np.zeros(8, dtype=np.int64), scores, groups) is None


def test_textbook_weight_drives_every_score_down_and_balanced_does_not() -> None:
    # 16 revealed positives and 48 unlabeled accounts (a training batch), float64 so the
    # tiny gradients are exact. Logit -15.6 is a score of about 1.7e-7: the state the
    # reference run with prior 0.001 settled in.
    prior = 0.001
    targets = torch.tensor([1.0] * 16 + [0.0] * 48, dtype=torch.float64)
    textbook = NonNegativePULoss(prior=prior)
    balanced = NonNegativePULoss(prior=prior, positive_weight=1 - prior)
    collapsed = torch.full((64,), -15.6, dtype=torch.float64, requires_grad=True)
    slope = _sigmoid(-15.6) * (1 - _sigmoid(-15.6))
    # Scoring everything near zero costs only the prior under the textbook weight, and
    # the summed gradient still points down (the unlabeled push beats the positives'
    # pull); under the balanced weight the same state costs 1 - prior and the pushes
    # cancel, so nothing draws the scores there.
    loss, _ = _call(textbook, collapsed, targets)
    assert loss.item() == pytest.approx(prior, rel=1e-3)
    (grad,) = torch.autograd.grad(loss, collapsed)
    assert grad.sum().item() == pytest.approx((1 - 2 * prior) * slope, rel=1e-6)
    loss, _ = _call(balanced, collapsed, targets)
    assert loss.item() == pytest.approx(1 - prior, rel=1e-3)
    (grad,) = torch.autograd.grad(loss, collapsed)
    assert abs(grad.sum().item()) < 1e-6 * slope
    # From an untrained start (logits 0) the positives' pull is 1 / (2 * prior) = 500
    # times larger, because the -prior * R_p^- term pulls them up under both weights.
    start = torch.zeros(64, dtype=torch.float64, requires_grad=True)
    pulls = []
    for loss_fn in (textbook, balanced):
        (grad,) = torch.autograd.grad(_call(loss_fn, start, targets)[0], start)
        pulls.append(-grad[:16].sum().item())
    assert pulls[1] / pulls[0] == pytest.approx(1 / (2 * prior), rel=1e-6)


def test_badly_scored_positives_keep_their_gradient_in_float32() -> None:
    # sigmoid(-f) rounds to 1 in float32 below f of about -17, which zeroed the gradient.
    targets = torch.tensor([1.0, 0.0])
    logits = torch.tensor([-20.0, 0.0], requires_grad=True)
    loss, _ = _call(NonNegativePULoss(prior=0.001, positive_weight=0.999), logits, targets)
    (grad,) = torch.autograd.grad(loss, logits)
    assert grad[0].item() == pytest.approx(-(0.999 + 0.001) * _sigmoid(-20.0), rel=1e-4)
