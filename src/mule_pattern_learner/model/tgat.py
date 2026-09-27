"""TGAT-style event-time attention with fixed GSQL Fourier features.

Historical payment neighbors are represented at their own event cutoffs.
Valid-time associations retain the current cutoff and consume another layer.
Optional root summaries are independently switchable, and so is a sum of a per-slot
MLP over the root's hop-1 slots (attention averages linear projections of the slots).
The event path can run with zero node features; no recurrent memory or account
embedding table is used.
"""

from __future__ import annotations

import torch
from torch import nn

from ..contract.feature_groups import FeaturePlan
from ..contract.graph_schema import CHANNELS, RAILS, RELATIONS, STRATA


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


class LiveTGAT(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        dropout: float,
        *,
        plan: FeaturePlan | None = None,
        slot_sum: bool = False,
        first_fanout: int = 8,
    ) -> None:
        super().__init__()
        if not 8 <= hidden <= 512 or not 1 <= heads <= 16 or hidden % heads:
            raise ValueError("Hidden size must be 8..512 and divisible by 1..16 heads")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        if not isinstance(slot_sum, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"slot_sum must be true or false, got {slot_sum!r}")
        # Batches hold one column per hop-1 slot, at most 64 (make_live_batch).
        if (
            isinstance(first_fanout, bool)
            or not isinstance(first_fanout, int)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not 1 <= first_fanout <= 64
        ):
            raise ValueError(f"Hop-1 fan-out must be an integer in [1,64], got {first_fanout!r}")
        self.plan = plan = plan or FeaturePlan()
        if slot_sum and plan.architecture == "summary":
            raise ValueError("The summary architecture has no hop-1 slots to sum")
        self.hidden = hidden
        # A summary model reads every root column; the split model's summary columns go
        # to their own branch.
        node_names = plan.node_names if plan.architecture == "summary" else plan.names("node")
        self.node_indices = tuple(plan.node_names.index(n) for n in node_names)
        self.summary_indices = tuple(plan.node_names.index(n) for n in plan.names("summary"))
        self.node = self.projection(len(node_names), hidden)
        self.summary = (
            self.projection(len(self.summary_indices), hidden)
            if plan.architecture == "split" and self.summary_indices
            else None
        )
        if plan.architecture != "summary":
            self.base = self.projection(len(plan.names("node")), hidden)
            self.relation = nn.Embedding(len(RELATIONS), hidden)
            self.rail = nn.Embedding(len(RAILS), hidden)
            self.edge = nn.Linear(len(plan.edge_names), hidden)
            self.channel = (
                nn.Embedding(len(CHANNELS), hidden) if "event_channel" in plan.groups else None
            )
            self.stratum = (
                nn.Embedding(len(STRATA), hidden) if "sampler_meta" in plan.groups else None
            )
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

    @staticmethod
    def projection(width: int, hidden: int) -> nn.Module | None:
        return (
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden))
            if width
            else None
        )

    def project(self, module: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
        # A true zero-feature arm has no input projection parameters.
        return module(x) if module is not None else x.new_zeros((*x.shape[:-1], self.hidden))

    def edge_embedding(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        value = (
            self.edge(batch[prefix + "edge"])
            + self.relation(batch[prefix + "relation"])
            + self.rail(batch[prefix + "rail"])
        )
        if self.channel is not None:
            value = value + self.channel(batch[prefix + "channel"])
        if self.stratum is not None:
            value = value + self.stratum(batch[prefix + "stratum"])
        return value

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.project(self.node, batch["x"][..., list(self.node_indices)])
        positions = batch["root_positions"]
        if self.plan.architecture == "summary":
            return x[positions]
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
