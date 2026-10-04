"""The linear controls: one linear layer of the root's own inputs, alone or beside TGAT.

The root's own inputs are its node and summary columns, the inputs SummaryMLP reads
(FeaturePlan.node_names): the entity flags, the hub flag and the pool counts. The
"linear" architecture scores a root with one linear layer of them and reads no
neighbours, so its batches hold only the roots (batching.assemble). The
"wide_and_deep" architecture is the graph model with that linear layer's output added
to its logit, so the same inputs reach the score both through attention and directly.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from ..contract.feature_groups import FeaturePlan
from .tgat import TGAT


def root_inputs(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The node and summary columns of each root of a batch."""
    return batch["x"][batch["root_positions"]]


class LinearModel(nn.Module):
    """One linear layer of the root's own inputs: the logit is a weighted sum of them.

    Its encoding is the inputs themselves, so its embedding is what it reads, and its
    head is the layer.
    """

    def __init__(self, *, plan: FeaturePlan) -> None:
        super().__init__()
        if plan.architecture != "linear":
            raise ValueError(f"LinearModel needs a linear feature plan, not {plan.architecture!r}")
        self.plan = plan
        # A root-only plan always has node or summary columns (FeaturePlan).
        self.head = nn.Linear(len(plan.node_names), 1)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return root_inputs(batch)

    def logits(self, batch: dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        """The logits of roots whose encoding is ``hidden``."""
        return self.head(hidden).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logits(batch, self.encode(batch))


class WideAndDeep(TGAT):
    """The graph model with a linear layer of the root's own inputs added to its logit.

    TGAT's modules are created first, in its order, so a seed gives them the built-in
    model's initial weights; the wide path, one linear layer of the root's node and
    summary columns, comes after them. The encoding is TGAT's, so the embeddings a
    predictor keeps are the graph model's.
    """

    ARCHITECTURE: ClassVar[str] = "wide_and_deep"

    def __init__(
        self,
        hidden: int,
        heads: int,
        dropout: float,
        *,
        plan: FeaturePlan,
        slot_sum: bool,
        first_fanout: int,
    ) -> None:
        super().__init__(
            hidden, heads, dropout, plan=plan, slot_sum=slot_sum, first_fanout=first_fanout
        )
        self.wide = nn.Linear(len(plan.node_names), 1)

    def logits(self, batch: dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
        """TGAT's logits of roots encoded as ``hidden``, plus the wide path's."""
        return super().logits(batch, hidden) + self.wide(root_inputs(batch)).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logits(batch, self.encode(batch))
