"""The model a configuration describes, and float64 probabilities from its logits."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..config import fanouts
from ..contract.feature_groups import FeaturePlan
from .tgat import LiveTGAT


def probabilities_from_logits(logits: torch.Tensor) -> np.ndarray:
    """Float64 mule probabilities on the host from the model's float32 logits.

    A float32 probability near 1 resolves logits only to about 0.007 at a logit of 11
    (0.05 at 13) and rounds to 1 above about 17, so the highest scores tie and top-k
    rankings among them are arbitrary. In float64 the sigmoid tells apart adjacent
    float32 logits up to about 23 and rounds to 1 only above about 37. The logits move
    to the CPU first, since MPS has no float64.
    """
    return torch.sigmoid(logits.detach().cpu().double()).numpy()


def build_model(
    config: dict[str, Any], plan: FeaturePlan, *, dropout: float | None = None
) -> LiveTGAT:
    """The model a validated configuration describes (hidden, heads, dropout, slot_sum).

    ``dropout`` replaces the configured rate, for dropout-free determinism checks.
    The summary architecture has no hop-1 slots, so it ignores ``slot_sum`` as it
    ignores the fanouts.
    """
    return LiveTGAT(
        int(config["hidden"]),
        int(config["heads"]),
        float(config["dropout"] if dropout is None else dropout),
        plan=plan,
        slot_sum=config["slot_sum"] if plan.architecture != "summary" else False,
        first_fanout=fanouts(config)[0],
    )
