"""Every numeric bound of the pipeline, defined once.

The configuration sections, the sampler plan, batch assembly and its limits, the model,
the dataset, the audit, the context source and the TigerGraph adapter read their ranges
here. Some are also checked by a query: the context query renders REQUEST_KEYS and POOL
into its text, and the hub and reveal queries refuse what HUB_CUTOFFS and
REVEAL_PER_SPLIT exclude.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator


@dataclass(frozen=True)
class Bound:
    """An inclusive range of integers."""

    low: int
    high: int

    def holds(self, value: int) -> bool:
        return self.low <= value <= self.high

    def check(self, name: str, value: object) -> int:
        """value as an int; a ValueError names it when it is no integer in the range."""
        try:
            number = operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            number = None
        if isinstance(value, bool) or number is None or not self.holds(number):
            raise ValueError(
                f"{name} must be an integer in [{self.low},{self.high}], got {value!r}"
            )
        return number


# Keys of one context request.
REQUEST_KEYS = Bound(1, 64)
# Context requests one source keeps in flight, and the contexts it keeps in memory.
QUERY_CONCURRENCY = Bound(1, 16)
CONTEXT_LRU_CAPACITY = Bound(0, 4096)
# Contexts the disk tier of one dataset keeps (data.context_cache). The baseline run's
# contexts_distinct should set it, so that the cache holds every context a run asked for
# and the next run of the dataset asks TigerGraph for none of them again. No run has
# measured that yet, so this is an estimate: a training step asks for at most 64 roots
# and 64 x 16 children, so the reference run's 11 epochs of 100 steps ask for at most
# 1.2 million contexts, and the proxy evaluations and the audits add about 70,000 each.
# Compressed, a full root pool takes about 14 KB and a child's about 6 KB, so the cap is
# roughly 10 GB of disk.
CONTEXT_CACHE_ENTRIES = 1_500_000
# Every n-th context request asks for the Fourier vectors, which are then checked.
ENCODING_CHECK_EVERY = Bound(1, 1_000_000)
# Attempts one query may count, and the seconds of unavailability it waits out.
QUERY_ATTEMPTS = Bound(1, 20)
OUTAGE_SECONDS = Bound(0, 86_400)
# Batches built ahead of the one in use.
PREFETCH_BATCHES = Bound(0, 8)
# Roots of one batch, and the contexts (roots and children) one batch holds; one
# fetch from a context source never asks for more contexts than a batch holds.
BATCH_ROOTS = Bound(1, 128)
BATCH_CONTEXTS = 2048
# Memory budgets of one batch (batching.limits.BatchLimits): its input tensors, the
# model's estimated working memory and the candidate messages its contexts may return.
TENSOR_BYTES = 64 * 1024 * 1024
MODEL_WORKING_BYTES = 512 * 1024 * 1024
CANDIDATE_MESSAGES = 524_288
# The model's hidden size and attention heads.
HIDDEN = Bound(8, 512)
HEADS = Bound(1, 16)
# Children sampled per context at each hop, and payment candidates per relation.
FANOUT = Bound(1, 64)
# The candidate pool of one context and hop (contract.sampler_plan.PoolPlan).
POOL = {
    "recent": Bound(1, 32),
    "older": Bound(0, 16),
    "distinct": Bound(0, 16),
    "associations": Bound(0, 8),
    "max_history": Bound(32, 4096),
}
# Candidates per association relation at hop 1, and the fanout slots they may fill.
ASSOCIATION_FANOUT = Bound(0, 8)
ASSOCIATION_SLOTS = Bound(0, 16)
# Seed of the evaluation sampler's draws: a signed 64-bit integer.
EVALUATION_SEED = Bound(0, 2**63 - 1)
# The seed reservoir of each split, the observed positives kept beside them, and the
# rows a prepared accounts or observed-label file may hold.
SEED_LIMIT = Bound(1, 20_000)
POSITIVE_POOL = 40_000
DATASET_ROWS = 100_000
# The ground-truth audit of a split: the accounts of the split's partition it holds, and
# the sample accounts it scores.
AUDIT_POPULATION = 1_000_000
AUDIT_SAMPLE = 100_000
# Known mules the reveal may reveal per split.
REVEAL_PER_SPLIT = Bound(0, 1000)
# Root cutoffs of one hub query.
HUB_CUTOFFS = Bound(1, 24)
# Bytes of an entity ID and of a scope ID in a request.
ID_BYTES = 1024
SCOPE_ID_BYTES = 256
