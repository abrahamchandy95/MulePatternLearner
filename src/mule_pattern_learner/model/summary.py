"""The controls without attention: an MLP over the root's own node and summary columns.

It reads no neighbours, so its batches hold only the roots (batching.assemble).
"""

from __future__ import annotations

import torch
from torch import nn

from ..contract.feature_groups import FeaturePlan
from .tgat import check_width, projection


class SummaryMLP(nn.Module):
    """A projection of every root column, then the head TGAT puts on its embedding.

    The projection is created before the head, so a seed gives the same initial weights
    as the summary model of the same settings saved before the layered restructure.
    """

    def __init__(self, hidden: int, dropout: float, *, plan: FeaturePlan) -> None:
        super().__init__()
        check_width(hidden, dropout)
        if plan.architecture != "summary":
            raise ValueError(f"SummaryMLP needs a summary feature plan, not {plan.architecture!r}")
        self.plan = plan
        self.hidden = hidden
        # A summary plan always has node or summary columns (FeaturePlan).
        self.node = projection(len(plan.node_names), hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.node is not None
        return self.node(batch["x"])[batch["root_positions"]]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)
