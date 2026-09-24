"""TGAT-style event-time attention with fixed GSQL Fourier features.

Historical payment neighbors are represented at their own event cutoffs.
Valid-time associations retain the current cutoff and consume another layer.
Optional root summaries are independently switchable. The event path can run
with zero node features; no recurrent memory or account embedding table is used.
"""

from __future__ import annotations

import torch
from torch import nn

from .contract import FeaturePlan, RELATIONS, RAILS, CHANNELS, STRATA


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
        hidden: int = 64,
        heads: int = 4,
        dropout: float = 0.15,
        variant: str = "temporal",
        *,
        plan: FeaturePlan | None = None,
    ) -> None:
        super().__init__()
        if variant not in {"temporal", "no_fourier", "tabular"}:
            raise ValueError("Unknown model variant")
        if not 8 <= hidden <= 512 or not 1 <= heads <= 16 or hidden % heads:
            raise ValueError("Hidden size must be 8..512 and divisible by 1..16 heads")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        self.variant = variant
        self.plan = plan or FeaturePlan.from_config({"variant": variant})
        self.legacy_no_fourier = plan is None and variant == "no_fourier"
        self.hidden = hidden
        plan = self.plan
        node_names = (
            plan.node_names if plan.architecture in ("single", "summary") else plan.names("node")
        )
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
        width = hidden * (2 if self.summary is not None else 1)
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
        x = batch[prefix + "edge"]
        if self.legacy_no_fourier:
            x = torch.cat((x[..., :3], x[..., 4:7]), dim=-1)
        value = (
            self.edge(x)
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
        event = self.layers[1](
            first[positions],
            first[batch["neighbor_positions"]],
            self.edge_embedding(batch, "first_"),
            batch["first_mask"],
        )
        if self.summary is not None:
            summary = self.summary(batch["x"][positions][:, list(self.summary_indices)])
            return torch.cat((event, summary), dim=-1)
        return event

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)
