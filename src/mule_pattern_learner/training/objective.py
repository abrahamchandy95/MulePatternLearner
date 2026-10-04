"""The nnPU objective a loss section names, one optimizer step on it, and its risk."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from ..config import LossConfig
from ..model.loss import NonNegativePULoss


def nnpu_objective(loss: LossConfig) -> tuple[float, float]:
    """The class prior and the positive-risk weight of a loss section.

    "prior" is textbook nnPU (the weight is the prior). "balanced" is imbalanced nnPU
    (Su, Chen and Xu, IJCAI 2021) with a balanced target prior of 0.5: its risk
    0.5 * R_p^+ + 0.5 / (1 - prior) * (R_u^- - prior * R_p^-) is this loss with weight
    1 - prior, scaled by a constant (exactly so for the loss's beta = 0, gamma = 1).
    """
    prior, weight = loss.class_prior, loss.positive_weight
    match weight:
        case "prior":
            return prior, prior
        case "balanced":
            return prior, 1.0 - prior
        case float():
            return prior, weight
        case _:
            raise ValueError(f"Unknown positive weight {weight!r}")


def objective_name(prior: float, positive_weight: float) -> str:
    if positive_weight == prior:
        return "nnPU"
    if math.isclose(positive_weight, 1.0 - prior):
        return "imbalanced_nnPU"
    return "positive_reweighted_nnPU"


class StepLoss(NamedTuple):
    """One step's backpropagated nnPU loss and its unclamped risk estimate (detached).

    They differ exactly when the non-negative correction fired, which a model that
    memorises its few revealed positives makes frequent.
    """

    value: torch.Tensor
    objective: torch.Tensor


def nnpu_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: NonNegativePULoss,
    batch: dict[str, torch.Tensor],
    positives: int,
    seed: int,
) -> StepLoss:
    """One optimizer step; the batch's leading ``positives`` rows are observed positives.

    Dropout masks depend only on ``seed`` (the step seed), never on earlier history.
    """
    torch.manual_seed(seed)
    logits = model(batch)
    targets = torch.zeros_like(logits)
    targets[:positives] = 1
    value, objective = loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    value.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return StepLoss(value.detach(), objective.detach())


def pu_risk(labels: np.ndarray, scores: np.ndarray, prior: float, positive_weight: float) -> float:
    """The non-negative nnPU risk of scored accounts: the run's objective on a proxy sample.

    ``labels`` mark the revealed positives (1) and the unlabeled accounts (0), and
    ``scores`` are the model's probabilities, the sigmoid of its logits, so the surrogate
    losses of model.loss.NonNegativePULoss are 1 - score for an account taken as positive
    and the score for one taken as negative. The risk is

        positive_weight * R_p^+ + max(0, R_u^- - prior * R_p^-)

    with the run's prior and positive weight: the risk the loss estimates, whose negative
    part nnPU holds at zero (the loss's beta = 0), so a model that scores the unlabeled
    accounts below the positives' share of them gains nothing for it. Lower is better.
    Both classes are needed.
    """
    positive = labels == 1
    if positive.all() or not positive.any():
        raise ValueError("The nnPU risk needs revealed positives and unlabeled accounts")
    as_positive = float(np.mean(1.0 - scores[positive]))
    negative = float(np.mean(scores[~positive])) - prior * float(np.mean(scores[positive]))
    return positive_weight * as_positive + max(0.0, negative)
