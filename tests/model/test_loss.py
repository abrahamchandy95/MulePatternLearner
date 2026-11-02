"""The non-negative PU loss: values, gradients and the non-negative correction."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from mule_pattern_learner.model.loss import NonNegativePULoss


def reference_nnpu(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    prior: float,
    positive_weight: float,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The former NonNegativePULoss.forward, with its host-side branch.

    l_pos is 1 - sigmoid(f), as the loss now computes it (the float32 gradient fix).
    """
    positive = (targets == 1).to(logits.dtype)
    unlabeled = (targets == 0).to(logits.dtype)
    n_positive = torch.clamp(positive.sum(), min=1.0)
    n_unlabeled = torch.clamp(unlabeled.sum(), min=1.0)
    l_pos, l_neg = 1 - torch.sigmoid(logits), torch.sigmoid(logits)
    positive_risk = positive_weight * torch.sum(positive * l_pos) / n_positive
    negative_risk = (
        torch.sum(unlabeled * l_neg) / n_unlabeled
        - prior * torch.sum(positive * l_neg) / n_positive
    )
    objective = positive_risk + negative_risk
    if negative_risk.item() < -beta:
        return gamma * (-negative_risk), objective
    return objective, objective


def loss_cases() -> list[dict[str, Any]]:
    cases = []
    generator = torch.Generator().manual_seed(0)
    for index in range(48):
        n = int(torch.randint(2, 40, (1,), generator=generator))
        scale = (0.5, 3.0, 12.0)[index % 3]
        logits = torch.randn(n, generator=generator, dtype=torch.float64) * scale
        targets = (torch.rand(n, generator=generator) < 0.3).long()
        if index % 7 == 0:
            targets[:] = 0  # no positives in the batch
        if index % 5 == 0:
            # Confident separation drives the negative risk below zero.
            logits = torch.where(targets == 1, logits.abs() + 8, -logits.abs() - 8)
        cases.append(
            {
                "logits": logits,
                "targets": targets,
                "prior": (0.001, 0.05, 0.3, 0.5)[index % 4],
                "positive_weight": (None, 0.1, 0.5)[index % 3],
                "beta": (0.0, 0.0, 0.25)[index % 3],
                "gamma": (1.0, 1.0, 0.5, 2.0)[index % 4],
            }
        )
    return cases


def _loss_outputs(
    module: torch.nn.Module, case: dict[str, Any], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits = case["logits"].to(dtype).clone().requires_grad_(True)
    train_loss, objective = module(logits, case["targets"])
    train_loss.backward()
    return train_loss.detach(), objective.detach(), logits.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nnpu_values_and_gradients_equal_the_former_branch(dtype: torch.dtype) -> None:
    fired = set()
    for case in loss_cases():
        # positive_weight None exercises the constructor's default, the balanced 1 - prior.
        weight = case["positive_weight"] or 1 - case["prior"]
        new = NonNegativePULoss(
            case["prior"],
            beta=case["beta"],
            gamma=case["gamma"],
            positive_weight=case["positive_weight"],
        )
        logits = case["logits"].to(dtype).clone().requires_grad_(True)
        old_train, old_objective = reference_nnpu(
            logits,
            case["targets"],
            prior=case["prior"],
            positive_weight=weight,
            beta=case["beta"],
            gamma=case["gamma"],
        )
        old_train.backward()
        train_loss, objective, gradient = _loss_outputs(new, case, dtype)
        fired.add(bool(old_train.detach() != old_objective.detach()))
        torch.testing.assert_close(train_loss, old_train.detach(), rtol=0, atol=0)
        torch.testing.assert_close(objective, old_objective.detach(), rtol=0, atol=0)
        torch.testing.assert_close(gradient, logits.grad, rtol=0, atol=0)
    assert fired == {True, False}, "Both the corrected and the plain branch must be covered"


def test_nnpu_forward_never_reads_a_value_on_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object) -> None:
        raise AssertionError("host synchronization")

    logits = torch.tensor([9.0, 9.0, -9.0, -9.0], requires_grad=True)
    targets = torch.tensor([1, 1, 0, 0])
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    train_loss, objective = NonNegativePULoss(0.5)(logits, targets)
    train_loss.backward()
    monkeypatch.undo()
    assert train_loss.item() > 0 > objective.item()


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
    # The default weight on the positive risk is the balanced 1 - pi.
    positive_risk = (1 - pi) * sum(l_pos[i] for i in pos_idx) / n_p
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


def test_textbook_weight_drives_every_score_down_and_balanced_does_not() -> None:
    # 16 revealed positives and 48 unlabeled accounts (a training batch), float64 so the
    # tiny gradients are exact. Logit -15.6 is a score of about 1.7e-7: the state the
    # reference run with prior 0.001 settled in.
    prior = 0.001
    targets = torch.tensor([1.0] * 16 + [0.0] * 48, dtype=torch.float64)
    textbook = NonNegativePULoss(prior=prior, positive_weight=prior)
    balanced = NonNegativePULoss(prior=prior)
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


def _weighted(
    loss: NonNegativePULoss, f: Tensor, t: Tensor, w: Tensor | None
) -> tuple[Tensor, Tensor]:
    return cast("tuple[Tensor, Tensor]", loss(f, t, w))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unit_weights_are_the_unweighted_loss(dtype: torch.dtype) -> None:
    for case in loss_cases():
        loss = NonNegativePULoss(case["prior"], positive_weight=case["positive_weight"])
        logits = case["logits"].to(dtype)
        ones = torch.ones(len(logits), dtype=dtype)
        plain = _weighted(loss, logits, case["targets"], None)
        unit = _weighted(loss, logits, case["targets"], ones)
        for left, right in zip(plain, unit, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_each_group_is_its_weighted_sum_over_its_count_in_the_batch() -> None:
    """Weights scale each example's loss in the positive and the unlabeled risk, and each
    group's sum is divided by how many examples of the group the batch holds, not by what
    they weigh: a positive of small weight alone in its group adds that small weight, not the
    whole positive risk. The correction reads every positive at unit weight."""
    f = torch.tensor([2.0, -0.5, 0.5, -1.0, 0.0], dtype=torch.float64)
    t = torch.tensor([1, 1, 0, 0, 0])
    loss = NonNegativePULoss(prior=0.3)
    weights = torch.tensor([0.1, 3.0, 0.05, 0.15, 2.0], dtype=torch.float64)
    _, objective = _weighted(loss, f, t, weights)
    l_pos, l_neg = [1 - _sigmoid(x) for x in f.tolist()], [_sigmoid(x) for x in f.tolist()]
    r_pos = (0.1 * l_pos[0] + 3.0 * l_pos[1]) / 2
    r_pos_neg = (l_neg[0] + l_neg[1]) / 2
    r_unl = (0.05 * l_neg[2] + 0.15 * l_neg[3] + 2.0 * l_neg[4]) / 3
    # The balanced weight on the positive risk is 1 - 0.3 = 0.7.
    assert objective.item() == pytest.approx(0.7 * r_pos + r_unl - 0.3 * r_pos_neg)
    _, objective = _weighted(loss, f[:1], t[:1], weights[:1])
    assert objective.item() == pytest.approx(0.7 * 0.1 * l_pos[0] - 0.3 * l_neg[0])
    # A group the batch does not hold adds nothing.
    _, objective = _weighted(loss, f[2:], t[2:], weights[2:])
    assert objective.item() == pytest.approx(r_unl)


def test_the_default_weight_is_balanced() -> None:
    """docs/design.md ("Training and selection"): nnPU with balanced weighting, 1 - prior on
    the positive risk, unless a weight is given."""
    for case in loss_cases():
        default = NonNegativePULoss(case["prior"])
        balanced = NonNegativePULoss(case["prior"], positive_weight=1 - case["prior"])
        for left, right in zip(
            _call(default, case["logits"], case["targets"]),
            _call(balanced, case["logits"], case["targets"]),
            strict=True,
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_the_correction_reads_the_positives_at_unit_weight() -> None:
    """A positive's weight, 1 over its mule's active weeks, scales its positive risk only:
    the negative risk, and so the corrected loss when it fires, does not change with it."""
    f = torch.tensor([3.0, 1.0, -9.0, -9.0], dtype=torch.float64)
    t = torch.tensor([1, 1, 0, 0])
    loss = NonNegativePULoss(prior=0.4)
    unit = torch.ones(4, dtype=torch.float64)
    weighted = torch.tensor([0.25, 1.75, 1.0, 1.0], dtype=torch.float64)
    corrected_unit, objective_unit = _weighted(loss, f, t, unit)
    corrected, objective = _weighted(loss, f, t, weighted)
    # The negative risk is below zero, so the correction fires, -negative_risk alike.
    l_neg = [_sigmoid(x) for x in f.tolist()]
    negative_risk = (l_neg[2] + l_neg[3]) / 2 - 0.4 * (l_neg[0] + l_neg[1]) / 2
    assert negative_risk < 0
    assert corrected.item() == pytest.approx(-negative_risk)
    assert corrected_unit.item() == pytest.approx(-negative_risk)
    # The objective differs by the positive risk alone: 0.6 x the weighted mean of l(+f).
    l_pos = [1 - _sigmoid(x) for x in f.tolist()]
    difference = 0.6 * ((0.25 - 1) * l_pos[0] + (1.75 - 1) * l_pos[1]) / 2
    assert objective.item() - objective_unit.item() == pytest.approx(difference)
