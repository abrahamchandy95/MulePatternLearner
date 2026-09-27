"""The nnPU objective a configuration names: its class prior and positive weight."""

from __future__ import annotations

import math
from typing import Any


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
