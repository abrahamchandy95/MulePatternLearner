"""The nnPU objective a configuration names, and one optimizer step on it."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
from torch import nn

from ..model.loss import NonNegativePULoss


def nnpu_objective(config: dict[str, Any]) -> tuple[float, float]:
    """The class prior and the positive-risk weight.

    "prior" is textbook nnPU (the weight is the prior). "balanced" is imbalanced nnPU
    (Su, Chen and Xu, IJCAI 2021) with a balanced target prior of 0.5: its risk
    0.5 * R_p^+ + 0.5 / (1 - prior) * (R_u^- - prior * R_p^-) is this loss with weight
    1 - prior, scaled by a constant (exactly so for the loss's beta = 0, gamma = 1).
    """
    prior = float(config["class_prior"])
    weight = config["positive_weight"]
    if weight == "prior":
        return prior, prior
    if weight == "balanced":
        return prior, 1.0 - prior
    return prior, float(weight)


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
