"""Explicit batch limits and disposable temporal-ID indexing.

No global ID table grows with the database or across batches. Limits bound this
pipeline's allocations; they cannot guarantee free memory in other processes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .contract import ContextKey


class BatchCapacityError(ValueError):
    """Reject an oversized request before allocating model tensors."""


@dataclass(frozen=True)
class BatchLimits:
    max_roots: int = 128
    max_contexts: int = 2048
    max_tensor_bytes: int = 64 * 1024 * 1024
    max_model_working_bytes: int = 512 * 1024 * 1024

    def validate(self, roots: int, fanouts: tuple[int, int]) -> None:
        first, second = fanouts
        contexts = roots * (first + 1)
        # Upper bound for all float32/int64/bool input arrays, before deduplication.
        edges = roots * first + contexts * second
        size = roots * 8 + contexts * 83 * 4 + roots * first * 8
        size += contexts * second * 9 * 4 + edges * (135 * 4 + 8 + 8 + 1)
        if roots > self.max_roots or contexts > self.max_contexts or size > self.max_tensor_bytes:
            raise BatchCapacityError(
                f"Batch exceeds memory budget: roots={roots}, contexts<={contexts}, "
                f"tensor_bytes<={size}. Reduce batch size or fanouts."
            )

    def validate_model(self, roots: int, fanouts: tuple[int, int], hidden: int) -> None:
        self.validate(roots, fanouts)
        contexts = roots * (fanouts[0] + 1)
        positions = roots * (fanouts[0] + 1) + contexts * (fanouts[1] + 1)
        # Conservative allowance for projections, attention, gradients and temporaries;
        # this is an admission estimate, not a guarantee about allocator/free memory.
        working = positions * hidden * 4 * 24
        if working > self.max_model_working_bytes:
            raise BatchCapacityError(
                f"Estimated model working memory {working} exceeds budget; "
                "reduce batch size, hidden size or fanouts"
            )


class BatchIndex:
    """Dense [0,N) IDs for this batch only; equal keys reuse the same slot.

    The stable key uses vertex type + public ID + clocks + scope/phase. TigerGraph
    internal IDs never become tensor offsets. Equal text IDs of different types,
    or an account at different historical times, remain distinct.
    """

    def __init__(self, keys: Iterable[ContextKey], capacity: int = 2048) -> None:
        self.keys: list[ContextKey] = []
        self.positions: dict[ContextKey, int] = {}
        for key in keys:
            if key in self.positions:
                continue
            if len(self.keys) >= capacity:
                raise BatchCapacityError("Batch-local ID capacity exceeded")
            self.positions[key] = len(self.keys)
            self.keys.append(key)

    def __getitem__(self, key: ContextKey) -> int:
        return self.positions[key]
