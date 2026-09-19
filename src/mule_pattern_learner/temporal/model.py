"""Bounded two-hop attention for an entity's state at a scoring cutoff.

Payment vertices are projected for sampling through their observed roles;
each sampled message keeps its payment's rail, amount, age and directed pair gap.
This is snapshot temporal attention, not a recurrent TGN memory implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from .encoding import fourier64
from .snapshots import Snapshot


@dataclass
class Normalizer:
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]

    @classmethod
    def fit(cls, snapshot: Snapshot) -> Normalizer:
        # Only nodes visible in this training snapshot; padding/future nodes are zero.
        x = snapshot.x[np.any(snapshot.x != 0, axis=1)]
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), 1).astype(np.float32))

    def apply(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        result = np.clip((x - self.mean) / self.scale, -10, 10)
        result[~np.any(x != 0, axis=-1)] = 0
        return result.astype(np.float32)


class TemporalAttention(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(hidden)
        self.scale = hidden**-0.5

    def forward(
        self, roots: torch.Tensor, neighbors: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        messages = neighbors + edges
        scores = (self.query(roots).unsqueeze(1) * self.key(messages)).sum(-1) * self.scale
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) * mask
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-9)
        pooled = (weights.unsqueeze(-1) * self.value(messages)).sum(1)
        return self.norm(roots + self.output(torch.cat([roots, pooled], dim=-1)))


class TemporalModel(nn.Module):
    def __init__(
        self, feature_dim: int, hidden: int = 32, dropout: float = 0.15, variant: str = "temporal"
    ) -> None:
        super().__init__()
        if variant not in {"temporal", "no_fourier", "tabular"}:
            raise ValueError("Unknown model variant")
        self.variant = variant
        self.node = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.relation = nn.Embedding(24, hidden)
        self.rail = nn.Embedding(7, hidden)
        self.edge = nn.Linear(132 if variant == "temporal" else 4, hidden)
        self.layers = nn.ModuleList([TemporalAttention(hidden, dropout) for _ in range(2)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def edge_embedding(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        return (
            self.edge(batch[prefix + "edge"])
            + self.relation(batch[prefix + "relation"])
            + self.rail(batch[prefix + "rail"])
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.variant == "tabular":
            return self.node(batch["root_x"])
        first = self.layers[0](
            self.node(batch["unique_x"]),
            self.node(batch["second_x"]),
            self.edge_embedding(batch, "second_"),
            batch["second_mask"],
        )
        roots = first[batch["root_positions"]]
        neighbors = first[batch["neighbor_positions"]]
        hidden = self.layers[1](
            roots, neighbors, self.edge_embedding(batch, "first_"), batch["first_mask"]
        )
        return hidden

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)


def make_batch(
    snapshot: Snapshot,
    roots: NDArray[np.int64],
    normalizer: Normalizer,
    *,
    variant: str,
    fanouts: tuple[int, int] = (8, 4),
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    if min(fanouts) < 1 or max(fanouts) > snapshot.neighbors.shape[1]:
        raise ValueError("Fanout outside staged neighborhood capacity")
    arrays: dict[str, NDArray[Any]] = {"root_x": normalizer.apply(snapshot.x[roots])}
    if variant != "tabular":
        first_neighbors = snapshot.neighbors[roots, : fanouts[0]]
        joined = np.r_[roots, first_neighbors.reshape(-1)]
        unique, inverse = np.unique(joined, return_inverse=True)
        arrays["root_positions"] = inverse[: len(roots)]
        arrays["neighbor_positions"] = inverse[len(roots) :].reshape(first_neighbors.shape)
        arrays["unique_x"] = normalizer.apply(snapshot.x[unique])
        second_neighbors = snapshot.neighbors[unique, : fanouts[1]]
        arrays["second_x"] = normalizer.apply(snapshot.x[second_neighbors])
        for prefix, indices, count in (
            ("first_", roots, fanouts[0]),
            ("second_", unique, fanouts[1]),
        ):
            base = snapshot.edge[indices, :count].copy()
            if variant == "temporal":
                age = fourier64(snapshot.age_ms[indices, :count]) * base[..., 3:4]
                gap = fourier64(snapshot.gap_ms[indices, :count]) * base[..., 2:3]
                base = np.concatenate([base, age, gap], axis=-1)
            arrays[prefix + "edge"] = base
            arrays[prefix + "relation"] = snapshot.relation[indices, :count].astype(np.int64)
            arrays[prefix + "rail"] = snapshot.rail[indices, :count].astype(np.int64)
            arrays[prefix + "mask"] = snapshot.neighbors[indices, :count] != 0
    return {name: torch.from_numpy(np.asarray(value)).to(device) for name, value in arrays.items()}


def nnpu_loss(
    positive_logits: torch.Tensor,
    marginal_logits: torch.Tensor,
    prior: float,
    beta: float = 0,
    gamma: float = 1,
) -> torch.Tensor:
    """nnPU sigmoid risk. Marginal samples include both revealed and hidden P.

    Labels on hidden positives are not consulted; the class prior is an explicit
    assumption. Both independently drawn samples are mandatory in every step.
    """
    if not 0 < prior < 1 or not positive_logits.numel() or not marginal_logits.numel():
        raise ValueError("nnPU requires a prior in (0,1) and nonempty P and marginal samples")
    positive_risk = prior * torch.sigmoid(-positive_logits).mean()
    negative_risk = (
        torch.sigmoid(marginal_logits).mean() - prior * torch.sigmoid(positive_logits).mean()
    )
    if negative_risk.detach().item() < -beta:
        return -gamma * negative_risk
    return positive_risk + negative_risk
