"""Explicit batch limits and disposable temporal-ID indexing.

No global ID table grows with the database or across batches. Limits bound this
pipeline's allocations; they cannot guarantee free memory in other processes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .contract import ContextKey, FeaturePlan, SamplerPlan


class BatchCapacityError(ValueError):
    """Reject an oversized request before allocating model tensors."""


@dataclass(frozen=True)
class BatchLimits:
    """Admission limits checked before any fetch.

    Contexts are bounded by roots * (K1 + 1): the roots plus one child per first-hop
    slot. Parsed candidate messages are bounded by the per-hop pools: every root
    returns at most `sampler.response_bound(1)` and every child at most
    `sampler.response_bound(2)` messages.
    """

    max_roots: int = 128
    max_contexts: int = 2048
    max_tensor_bytes: int = 64 * 1024 * 1024
    max_model_working_bytes: int = 512 * 1024 * 1024
    max_candidate_messages: int = 524_288

    def validate(
        self,
        roots: int,
        fanouts: tuple[int, int],
        plan: FeaturePlan = FeaturePlan(),
        sampler: SamplerPlan | None = None,
    ) -> None:
        first, second = fanouts
        contexts = roots if plan.architecture == "summary" else roots * (first + 1)
        # Upper bound for all float32/int64/bool input arrays, before deduplication.
        edges = roots * first + contexts * second
        size = roots * 8 + contexts * len(plan.node_names) * 4 + roots * first * 8
        size += contexts * second * len(plan.names("node")) * 4 + edges * (
            len(plan.edge_names) * 4 + 4 * 8 + 1
        )
        if roots > self.max_roots or contexts > self.max_contexts or size > self.max_tensor_bytes:
            raise BatchCapacityError(
                f"Batch exceeds memory budget: roots={roots}, contexts<={contexts}, "
                f"tensor_bytes<={size}. Reduce batch size or fanouts."
            )
        if sampler is not None:
            children = 0 if plan.architecture == "summary" else roots * first
            messages = roots * sampler.response_bound(1) + children * sampler.response_bound(2)
            if messages > self.max_candidate_messages:
                raise BatchCapacityError(
                    f"Candidate pools may return {messages} messages per batch "
                    f"(limit {self.max_candidate_messages}); reduce batch size, fanouts "
                    "or sampler pool sizes."
                )

    def validate_model(
        self,
        roots: int,
        fanouts: tuple[int, int],
        hidden: int,
        plan: FeaturePlan = FeaturePlan(),
        sampler: SamplerPlan | None = None,
    ) -> None:
        self.validate(roots, fanouts, plan, sampler)
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
