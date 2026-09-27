"""An exponential moving average of the model's weights, and the weights validation scores."""

from __future__ import annotations

from collections.abc import Generator
import contextlib
from typing import Any

import torch
from torch import nn


class WeightAverage:
    """Exponential moving average of a model's weights (Polyak averaging).

    With few revealed positives, resampled neighbourhoods and a constant learning
    rate, the raw weights keep moving around a region of similar loss; their average
    is a steadier model to validate, select and save. After n updates the decay is
    min(decay, (1 + n) / (10 + n)), as in TensorFlow's ExponentialMovingAverage, so
    early averages are not dominated by the initial weights. Training never reads it.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.updates = 0
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for key, value in model.state_dict().items():
            average = self.state[key]
            if average.is_floating_point():
                average.lerp_(value, 1 - decay)
            else:
                average.copy_(value)

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Generator[None]:
        """Load the average into ``model`` for the block, then restore its own weights."""
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state)
        try:
            yield
        finally:
            model.load_state_dict(raw)

    def saved(self) -> dict[str, Any]:
        return {
            "state": {k: v.detach().cpu().clone() for k, v in self.state.items()},
            "updates": self.updates,
        }

    def load(self, saved: dict[str, Any]) -> None:
        if set(saved["state"]) != set(self.state):
            raise ValueError("Saved weight average does not match the model's parameters")
        for key, value in saved["state"].items():
            self.state[key].copy_(value)
        self.updates = int(saved["updates"])


def evaluated_weights(
    model: nn.Module, average: WeightAverage | None
) -> contextlib.AbstractContextManager[None]:
    """The weights validation scores and selection keeps: the average, if any."""
    if average is None:
        return contextlib.nullcontext()
    return average.applied(model)
