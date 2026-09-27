"""The torch subset sampler: random keys plus a segmented rank, on any torch device."""

from __future__ import annotations

import numpy as np
import torch

from .candidates import NUM_RELATIONS, CandidateTable, group_ranks


class TorchGroupedSampler:
    """Uniform subset per (context, relation) from random keys; any torch device."""

    name = "torch"

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def subset(self, table: CandidateTable, quotas: np.ndarray, keys: torch.Tensor) -> torch.Tensor:
        context = torch.from_numpy(table.context).to(self.device)
        relation = torch.from_numpy(table.relation).to(self.device)
        group = context * NUM_RELATIONS + relation
        ranks = group_ranks(group, keys.to(self.device), table.num_contexts * NUM_RELATIONS)
        return ranks < torch.from_numpy(quotas).to(self.device)[relation]
