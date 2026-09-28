"""TGAT-style event-time attention with fixed GSQL Fourier features.

Historical payment neighbors are represented at their own event cutoffs.
Valid-time associations retain the current cutoff and consume another layer.
The root's summary columns feed their own branch, and a sum of a per-slot MLP over
the root's hop-1 slots is switchable (attention averages linear projections of the
slots). The event path can run with zero node features; no recurrent memory or
account embedding table is used.
"""

from __future__ import annotations

import torch
from torch import nn

from ..contract.bounds import FANOUT, HEADS, HIDDEN
from ..contract.feature_groups import FeaturePlan
from ..contract.graph_schema import RAILS, RELATIONS


class AttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(
        self, root: torch.Tensor, neighbors: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # An unmasked self entry makes isolated nodes and padding well defined.
        messages = torch.cat((root[:, None], neighbors + edge), dim=1)
        padding = torch.cat(
            (torch.zeros((len(root), 1), dtype=torch.bool, device=root.device), ~mask), dim=1
        )
        pooled, _ = self.attention(
            root[:, None], messages, messages, key_padding_mask=padding, need_weights=False
        )
        hidden = self.norm(root + pooled[:, 0])
        return self.output_norm(hidden + self.feedforward(hidden))


def projection(width: int, hidden: int) -> nn.Module | None:
    """A linear projection of width input columns, or None when there are none."""
    return (
        nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden)) if width else None
    )


def check_width(hidden: int, dropout: float) -> None:
    """The hidden size and dropout rate every model of the package accepts."""
    if not HIDDEN.holds(hidden):
        raise ValueError(f"Hidden size must be {HIDDEN.low}..{HIDDEN.high}")
    if not 0 <= dropout < 1:
        raise ValueError("Dropout must be in [0,1)")


class TGAT(nn.Module):
    """The graph model: attention over the root's sampled hop-1 and hop-2 slots.

    Beside attention, a summary branch reads the root's summary columns, and the slot
    sum (when on) adds a per-slot MLP summed over the hop-1 slots. Its submodules are
    created in a fixed order, so a seed gives the same initial weights as the model of
    the same settings saved before the layered restructure.
    """

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
        super().__init__()
        check_width(hidden, dropout)
        if not HEADS.holds(heads) or hidden % heads:
            raise ValueError(f"Hidden size must be divisible by {HEADS.low}..{HEADS.high} heads")
        if not isinstance(slot_sum, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"slot_sum must be true or false, got {slot_sum!r}")
        # Batches hold one column per hop-1 slot, at most FANOUT.high (build_batch).
        if (
            isinstance(first_fanout, bool)
            or not isinstance(first_fanout, int)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not FANOUT.holds(first_fanout)
        ):
            raise ValueError(
                f"Hop-1 fan-out must be an integer in [{FANOUT.low},{FANOUT.high}], "
                f"got {first_fanout!r}"
            )
        if plan.architecture != "tgat":
            raise ValueError(f"TGAT needs a tgat feature plan, not {plan.architecture!r}")
        self.plan = plan
        self.hidden = hidden
        # The root's node columns go to attention, its summary columns to their own branch.
        node_names = plan.names("node")
        self.node_indices = tuple(plan.node_names.index(n) for n in node_names)
        self.summary_indices = tuple(plan.node_names.index(n) for n in plan.names("summary"))
        self.node = projection(len(node_names), hidden)
        self.summary = (
            projection(len(self.summary_indices), hidden) if self.summary_indices else None
        )
        self.base = projection(len(plan.names("node")), hidden)
        self.relation = nn.Embedding(len(RELATIONS), hidden)
        self.rail = nn.Embedding(len(RAILS), hidden)
        self.edge = nn.Linear(len(plan.edge_names), hidden)
        self.layers = nn.ModuleList([AttentionBlock(hidden, heads, dropout) for _ in range(2)])
        # Built only when on, so a model without it keeps its parameters and initial weights.
        self.first_fanout = first_fanout
        self.slot_sum = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            if slot_sum
            else None
        )
        width = hidden * (1 + (self.slot_sum is not None) + (self.summary is not None))
        self.head = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def project(self, module: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
        # A model without node features has no input projection parameters.
        return module(x) if module is not None else x.new_zeros((*x.shape[:-1], self.hidden))

    def edge_embedding(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        return (
            self.edge(batch[prefix + "edge"])
            + self.relation(batch[prefix + "relation"])
            + self.rail(batch[prefix + "rail"])
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.project(self.node, batch["x"][..., list(self.node_indices)])
        positions = batch["root_positions"]
        first = self.layers[0](
            x,
            self.project(self.base, batch["second_x"]),
            self.edge_embedding(batch, "second_"),
            batch["second_mask"],
        )
        neighbors = first[batch["neighbor_positions"]]
        edge = self.edge_embedding(batch, "first_")
        event = self.layers[1](first[positions], neighbors, edge, batch["first_mask"])
        parts = [event]
        if self.slot_sum is not None:
            # The tokens the block above attends over, besides the root itself.
            parts.append(self.slot_total(self.slot_sum, neighbors + edge, batch["first_mask"]))
        if self.summary is not None:
            parts.append(self.summary(batch["x"][positions][:, list(self.summary_indices)]))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else event

    def slot_total(self, mlp: nn.Module, slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """A per-slot MLP summed over the valid slots (mask true) over the configured fan-out.

        Attention averages linear projections of the slots, so a condition that combines
        several slot inputs (an internal, first-time and large inflow, say) cannot be
        separated before the slots are pooled. The MLP applies such a condition to each
        slot first, and the sum counts the slots that meet it (Xu, Hu, Leskovec and
        Jegelka, ICLR 2019, on sum against mean and max pooling). The divisor is the
        constant fan-out, not the number of valid slots: when every slot is filled, as
        for most roots under the built-in sampler, the result is the share of slots that
        meet the condition; with fewer slots it also carries their number, and extra
        padded columns change nothing.
        """
        if mask.shape[1] > self.first_fanout:
            raise ValueError(
                f"Batch has {mask.shape[1]} hop-1 slots, more than the model's fan-out "
                f"{self.first_fanout}"
            )
        return mlp(slots).masked_fill(~mask[..., None], 0.0).sum(dim=1) / self.first_fanout

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)
