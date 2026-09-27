"""Candidate tables, selection keys and the merge of kept candidates into slots.

Rows are in canonical order, so keys drawn in row order do not depend on the order
TigerGraph printed them in. Evaluation keys are device-independent hashes; training
keys come from a CPU generator seeded by the step seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import functools
from typing import Any

import numpy as np
import torch

from ..contract.fingerprints import stable_hash
from ..contract.graph_schema import RELATION_INDEX, RELATIONS, ContextKey
from ..contract.sampler_plan import SamplerPlan

PAYMENT_RELATIONS = 4
NUM_RELATIONS = len(RELATIONS)
_MASK64 = (1 << 64) - 1


def context_hash(key: ContextKey) -> int:
    fields = (key.node_type, key.node_id, key.cutoff_seq, key.cutoff_ms)
    return stable_hash("\x1f".join(map(str, (*fields, key.scope_id, key.visibility_phase))))


def splitmix64(values: np.ndarray | np.integer[Any] | int) -> np.ndarray:
    """Vectorized SplitMix64 finalizer over uint64 (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hop_seed(step_seed: int, hop: int) -> int:
    """Hop 1 uses the step seed itself; hop 2 uses an independent derived stream."""
    seed = int(step_seed) & _MASK64
    return seed if hop == 1 else int(splitmix64(np.uint64(seed ^ hop)))


@dataclass(frozen=True)
class CandidateTable:
    """Candidate messages of C contexts, one row each, in canonical order.

    Rows are sorted by (context, relation, -event_seq, event_id, node_id), so keys
    drawn in row order do not depend on the order TigerGraph printed them in.
    """

    keys: tuple[ContextKey, ...]  # [C]
    context: np.ndarray  # int64 [N]
    relation: np.ndarray  # int64 [N], index into RELATIONS
    time_key: np.ndarray  # int64 [N]
    seed_time: np.ndarray  # int64 [C], 2 * cutoff_seq
    messages: tuple[dict[str, Any], ...]

    @property
    def num_contexts(self) -> int:
        return len(self.seed_time)

    def __len__(self) -> int:
        return len(self.context)

    @functools.cached_property
    def context_hash(self) -> np.ndarray:
        """uint64 [C] stable hash of every context key (evaluation keys only)."""
        return np.asarray([context_hash(k) for k in self.keys], dtype=np.uint64)

    @functools.cached_property
    def item_hash(self) -> np.ndarray:
        """uint64 [N] stable hash of relation:event_id (payments) or relation:node_id."""
        return np.asarray(
            [
                stable_hash(
                    m["relation"] + ":" + (m["event_id"] if r < PAYMENT_RELATIONS else m["node_id"])
                )
                for m, r in zip(self.messages, self.relation.tolist(), strict=True)
            ],
            dtype=np.uint64,
        )

    @classmethod
    def build(cls, keys: Sequence[ContextKey], rows: Sequence[dict[str, Any]]) -> CandidateTable:
        if len(keys) != len(rows):
            raise ValueError("One candidate row per context is required")
        context, relation, time_key, messages = [], [], [], []
        for c, (key, row) in enumerate(zip(keys, rows, strict=True)):
            ordered = sorted(
                row["messages"],
                key=lambda m: (_relation(m), -int(m["event_seq"]), m["event_id"], m["node_id"]),
            )
            for m in ordered:
                r = _relation(m)
                payment = r < PAYMENT_RELATIONS
                if payment and not m["event_id"]:
                    raise ValueError("Payment candidate without an event ID")
                context.append(c)
                relation.append(r)
                time_key.append(2 * int(m["event_seq"]) if payment else 2 * key.cutoff_seq - 1)
                messages.append(m)
        table = cls(
            tuple(keys),
            np.asarray(context, dtype=np.int64),
            np.asarray(relation, dtype=np.int64),
            np.asarray(time_key, dtype=np.int64),
            np.asarray([2 * key.cutoff_seq for key in keys], dtype=np.int64),
            tuple(messages),
        )
        if len(table) and np.any(table.time_key >= table.seed_time[table.context]):
            raise ValueError("Candidate event is not strictly before its context cutoff")
        return table


def _relation(message: dict[str, Any]) -> int:
    try:
        return RELATION_INDEX[message["relation"]]
    except KeyError:
        raise ValueError(f"Unknown relation {message['relation']!r}") from None


def relation_quotas(sampler: SamplerPlan, hop: int) -> np.ndarray:
    """Maximum sampled candidates per (context, relation); hop 2 is payments-only."""
    quotas = np.zeros(NUM_RELATIONS, dtype=np.int64)
    quotas[:PAYMENT_RELATIONS] = sampler.relation_fanouts[hop - 1]
    if hop == 1:
        quotas[PAYMENT_RELATIONS:] = sampler.association_fanout
    return quotas


def selection_keys(
    table: CandidateTable, *, mode: str, step_seed: int, evaluation_seed: int, hop: int
) -> torch.Tensor:
    """Nonnegative int64 key per candidate on the CPU; smaller keys are drawn first.

    Evaluation keys are SplitMix64(hop_seed(evaluation_seed, hop), context, item):
    hop 1 keys are those of SplitMix64(evaluation_seed, context, item) and hop 2 uses
    an independent derived stream, so a root selected at both hops does not see its
    hop-1 draw again as its hop-2 draw (SamplerPlan.fingerprint versions this).
    """
    if mode == "eval":
        # The same on every device and machine.
        state = splitmix64(np.uint64(hop_seed(evaluation_seed, hop)))
        state = splitmix64(state ^ table.context_hash[table.context])
        keys = splitmix64(state ^ table.item_hash) >> np.uint64(1)
        return torch.from_numpy(keys.astype(np.int64))
    generator = torch.Generator().manual_seed(hop_seed(step_seed, hop))
    return torch.randint(0, 2**62, (len(table),), generator=generator, dtype=torch.int64)


def group_ranks(group: torch.Tensor, keys: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Rank of each row within its group by ascending key; ties keep row order."""
    order = torch.argsort(keys, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    grouped = group[order]
    counts = torch.bincount(grouped, minlength=num_groups)
    starts = torch.cumsum(counts, 0) - counts
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), device=order.device) - starts[grouped]
    return ranks


def merge_slots(
    table: CandidateTable,
    keep: torch.Tensor,
    keys: torch.Tensor,
    *,
    hop: int,
    fanout: int,
    association_slots: int,
) -> np.ndarray:
    """Merge kept candidates into [C, fanout] row indices (-1 is padding).

    Payments interleave by position across RELATIONS[:4] and associations across
    RELATIONS[4:], positions ordered by the selection keys. With
    `reserve = min(association_slots, n_assoc, fanout // 4)` the slots hold
    `P[:K-reserve] + A[:reserve]`, then `P[K-reserve:] + A[reserve:]` up to K.
    Hop 2 keeps payments only: `P[:K]`.
    """
    device = keep.device
    rows = torch.nonzero(keep, as_tuple=True)[0]
    num = table.num_contexts
    out = torch.full((num, fanout), -1, dtype=torch.int64, device=device)
    context = torch.from_numpy(table.context).to(device)[rows]
    relation = torch.from_numpy(table.relation).to(device)[rows]
    payment = relation < PAYMENT_RELATIONS
    if hop == 2:
        rows, context, relation, payment = (
            rows[payment],
            context[payment],
            relation[payment],
            payment[payment],
        )
    if not len(rows):
        return out.cpu().numpy()
    position = group_ranks(
        context * NUM_RELATIONS + relation, keys.to(device)[rows], num * NUM_RELATIONS
    )
    associations = NUM_RELATIONS - PAYMENT_RELATIONS
    order = torch.where(
        payment,
        position * PAYMENT_RELATIONS + relation,
        position * associations + relation - PAYMENT_RELATIONS,
    )
    rank = group_ranks(context * 2 + (~payment).long(), order, 2 * num)
    n_pay = torch.bincount(context[payment], minlength=num)
    n_assoc = torch.bincount(context[~payment], minlength=num)
    limit = min(association_slots, fanout // 4) if hop == 1 else 0
    reserve = torch.clamp(n_assoc, max=limit)
    first_association = torch.minimum(n_pay, fanout - reserve)
    slot = torch.where(payment, rank, first_association[context] + rank)
    kept = torch.where(payment, rank < (fanout - reserve)[context], slot < fanout)
    out[context[kept], slot[kept]] = rows[kept]
    return out.cpu().numpy()


def group_counts(table: CandidateTable, rows: np.ndarray) -> np.ndarray:
    """[C, NUM_RELATIONS] count of `rows` (indices or a boolean mask) per group."""
    groups = table.context[rows] * NUM_RELATIONS + table.relation[rows]
    size = table.num_contexts * NUM_RELATIONS
    return np.bincount(groups, minlength=size).reshape(table.num_contexts, NUM_RELATIONS)


def expected_counts(table: CandidateTable, quotas: np.ndarray) -> np.ndarray:
    """min(visible candidates, fan-out) per (context, relation), what a draw must return.

    Only rows strictly before their context's seed time count, so a table that
    deliberately holds rows at or after the cutoff (the verify script's boundary
    probe) expects none of those.
    """
    visible = table.time_key < table.seed_time[table.context]
    return np.minimum(group_counts(table, visible), quotas[None, :])
