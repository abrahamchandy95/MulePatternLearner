"""What every subset sampler must do on a synthetic candidate table, as assertions.

tests/sampling/test_cugraph_sampler.py runs these checks on a mock pylibcugraph and
tests/integration/test_cugraph_sampler.py on a GPU: exactly min(candidates, fan-out)
per (context, relation) (hop 2 payments-only), strict temporal validity (candidates
exactly at the cutoff included), uniform inclusion by a chi-square test over many
seeds, and determinism for a fixed seed. The torch sampler is checked beside cuGraph;
the two draw different, equally distributed subsets for one seed, and only evaluation,
hash-keyed on the torch path, is identical across them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..contract.graph_schema import RELATIONS, ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..sampling.backend import select_resampled
from ..sampling.candidates import (
    NUM_RELATIONS,
    PAYMENT_COUNT,
    CandidateTable,
    group_counts,
    relation_quotas,
    selection_keys,
)
from ..sampling.cugraph_sampler import CuGraphSampler
from ..sampling.torch_sampler import TorchGroupedSampler

# 0.999 quantile of chi-square with 11 degrees of freedom.
CHI2_999 = {11: 31.26}
# (backend label, seed -> keep mask)
Runs = list[tuple[str, Callable[[int], np.ndarray]]]


def message(relation: str, seq: int, key: ContextKey, n: int) -> dict[str, Any]:
    payment = RELATIONS.index(relation) < PAYMENT_COUNT
    return {
        "relation": relation,
        "node_type": "Account" if payment else "Party",
        "node_id": f"{relation}-{n}",
        "event_id": f"{relation}:{seq}" if payment else "",
        "event_seq": seq if payment else key.cutoff_seq,
        "event_ts_ms": seq * 1000 if payment else key.cutoff_ms,
    }


def synthetic_table(
    contexts: int, rng: np.random.Generator, *, full: int | None = None
) -> CandidateTable:
    """Random payment and association candidates, or ``full`` payments per relation."""
    keys: list[ContextKey] = []
    rows: list[dict[str, Any]] = []
    for c in range(contexts):
        cutoff = int(rng.integers(50, 5000))
        key = ContextKey("Account", f"ctx{c}", cutoff, cutoff * 1000)
        messages: list[dict[str, Any]] = []
        for r, relation in enumerate(RELATIONS):
            limit = 20 if r < PAYMENT_COUNT else 3
            count = (
                full if full is not None and r < PAYMENT_COUNT else int(rng.integers(0, limit + 1))
            )
            if r < PAYMENT_COUNT:
                seqs = rng.choice(np.arange(1, cutoff), min(count, cutoff - 1), replace=False)
                messages += [message(relation, int(s), key, n) for n, s in enumerate(seqs)]
            elif full is None:
                messages += [message(relation, 0, key, n) for n in range(count)]
        keys.append(key)
        rows.append({"messages": messages})
    return CandidateTable.build(keys, rows)


def torch_subset(table: CandidateTable, quotas: np.ndarray, seed: int, device: str) -> np.ndarray:
    keys = selection_keys(table, mode="train", step_seed=seed, evaluation_seed=0, hop=1)
    return TorchGroupedSampler(device).subset(table, quotas, keys).cpu().numpy()


def check_subsets(
    engine: CuGraphSampler, table: CandidateTable, sampler: SamplerPlan, device: str
) -> None:
    """Exact counts, payments only at hop 2, no edge at or after the cutoff, determinism."""
    available = group_counts(table, np.ones(len(table), dtype=bool))
    for hop in (1, 2):
        quotas = relation_quotas(sampler, hop)
        expected = np.minimum(available, quotas[None, :])
        runs: Runs = [
            ("torch", lambda s, q=quotas: torch_subset(table, q, s, device)),
            ("cugraph", lambda s, q=quotas: engine.subset(table, q, random_state=s, device=device)),
        ]
        for name, run in runs:
            mask = run(11)
            assert np.array_equal(group_counts(table, mask), expected), (name, hop)
            if hop == 2:
                assert not mask[table.relation >= PAYMENT_COUNT].any(), name
            valid = table.time_key[mask] < table.seed_time[table.context[mask]]
            assert bool(valid.all()), f"{name} hop {hop} sampled an edge at its cutoff or later"
            assert np.array_equal(run(11), mask), f"{name} hop {hop}: same seed, other subset"
            assert not np.array_equal(run(12), mask), f"{name} hop {hop}: new seed, same subset"


def check_cutoff_boundary(engine: CuGraphSampler, device: str) -> None:
    """cuGraph's own strict filter keeps seq < cutoff and same-cutoff associations only.

    The table bypasses CandidateTable.build, whose validation would refuse it.
    """
    cutoff = 100
    key = ContextKey("Account", "boundary", cutoff, cutoff * 1000)
    times = np.asarray([2 * cutoff - 2, 2 * cutoff - 1, 2 * cutoff, 2 * cutoff + 1], dtype=np.int64)
    table = CandidateTable(
        keys=(key,),
        context=np.zeros(4, dtype=np.int64),
        relation=np.asarray([0, 4, 0, 0], dtype=np.int64),
        time_key=times,
        seed_time=np.asarray([2 * cutoff], dtype=np.int64),
        messages=tuple({"relation": RELATIONS[r]} for r in (0, 4, 0, 0)),
    )
    mask = engine.subset(table, np.full(NUM_RELATIONS, 8), random_state=3, device=device)
    assert mask.tolist() == [True, True, False, False]


def check_uniform_inclusion(engine: CuGraphSampler, seeds: int, device: str) -> None:
    """Every candidate of a relation is drawn equally often (chi-square at 0.999)."""
    n, quota, contexts = 12, 3, 64
    table = synthetic_table(contexts, np.random.default_rng(0), full=n)
    first = table.relation == 0
    quotas = np.zeros(NUM_RELATIONS, dtype=np.int64)
    quotas[0] = quota
    # Candidate rank inside its (context, relation) group, in canonical order.
    starts = np.searchsorted(table.context[first], np.arange(contexts))
    rank = np.arange(first.sum()) - starts[table.context[first]]
    runs: Runs = [
        ("torch", lambda s: torch_subset(table, quotas, s, device)),
        ("cugraph", lambda s: engine.subset(table, quotas, random_state=s, device=device)),
    ]
    for name, run in runs:
        counts = np.zeros(n)
        for seed in range(seeds):
            mask = run(seed)[first]
            counts += np.bincount(rank[mask], minlength=n)
        expected = seeds * contexts * quota / n
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        assert chi2 < CHI2_999[n - 1], f"{name} inclusion is not uniform: chi2={chi2:.1f}"


def check_merged_slots(engine: CuGraphSampler, sampler: SamplerPlan, device: str) -> None:
    """Slots stay in their context and never repeat; evaluation is the same on every backend."""
    table = synthetic_table(256, np.random.default_rng(2))
    for backend in ("torch", "cugraph"):
        slots = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=16,
            mode="train",
            step_seed=5,
            backend=backend,
            device=device,
            cugraph=engine,
        )
        rows = slots[slots >= 0]
        context = np.repeat(np.arange(len(slots)), (slots >= 0).sum(1))
        assert bool((table.context[rows] == context).all()), backend
        assert len(np.unique(rows)) == len(rows), backend
    evaluation = [
        select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=16,
            mode="eval",
            backend=b,
            device=d,
            cugraph=engine,
        )
        for b, d in (("torch", "cpu"), ("torch", device), ("cugraph", device))
    ]
    assert all(np.array_equal(evaluation[0], other) for other in evaluation[1:])
