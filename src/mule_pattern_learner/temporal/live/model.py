"""TGAT-style event-time attention with fixed GSQL Fourier features.

Historical payment neighbors are represented at their own event cutoffs.
Valid-time associations retain the current cutoff and consume another layer.
This heterogeneous extension uses rolling GSQL summaries and a fixed basis,
not the original paper's learned time frequencies or recurrent TGN memory.
"""

from __future__ import annotations

import torch
from torch import nn

from .contract import FEATURE_NAMES, RELATIONS, RAILS


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
        self, hidden: int = 64, heads: int = 4, dropout: float = 0.15, variant: str = "temporal"
    ) -> None:
        super().__init__()
        if variant not in {"temporal", "no_fourier", "tabular"}:
            raise ValueError("Unknown model variant")
        if not 8 <= hidden <= 512 or not 1 <= heads <= 16 or hidden % heads:
            raise ValueError("Hidden size must be 8..512 and divisible by 1..16 heads")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        self.variant = variant
        self.node = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.base = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.relation = nn.Embedding(len(RELATIONS), hidden)
        self.rail = nn.Embedding(len(RAILS), hidden)
        self.edge = nn.Linear(135 if variant == "temporal" else 7, hidden)
        self.layers = nn.ModuleList([AttentionBlock(hidden, heads, dropout) for _ in range(2)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def edge_embedding(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        x = batch[prefix + "edge"]
        if self.variant != "temporal":
            x = x[..., :7]
        return (
            self.edge(x)
            + self.relation(batch[prefix + "relation"])
            + self.rail(batch[prefix + "rail"])
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.node(batch["x"])
        if self.variant == "tabular":
            return x[batch["root_positions"]]
        first = self.layers[0](
            x,
            self.base(batch["second_x"]),
            self.edge_embedding(batch, "second_"),
            batch["second_mask"],
        )
        return self.layers[1](
            first[batch["root_positions"]],
            first[batch["neighbor_positions"]],
            self.edge_embedding(batch, "first_"),
            batch["first_mask"],
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)
