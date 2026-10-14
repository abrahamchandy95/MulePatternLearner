"""Bounded seed selection from server-assigned partitions, independent of truth."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
import heapq
from typing import Any

import pandas as pd

from ..config import DatasetConfig
from ..contract.bounds import ID_BYTES, POSITIVE_POOL
from ..contract.clock import timestamp
from ..contract.fingerprints import stable_score
from ..contract.graph_schema import PHASE_SPLIT, SPLITS
from ..contract.salts import RESERVOIR_SALT
from .observed_labels import ORACLE_COLUMNS
from .ports import ScopeReader


def scope_accounts(scope: ScopeReader) -> Iterator[dict[str, Any]]:
    """The split's accounts in account order, page by page from the scope reader.

    The one pager of the split's population: the seed reservoirs and the ground-truth
    audit both read it here. Rows carry the observed labels.
    """
    for page in scope.population_pages():
        yield from page


def select_accounts(
    scope: ScopeReader, dataset: DatasetConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep uniform hash reservoirs plus observed positives, never all account IDs.

    The reservoirs hold dataset.seed_limits accounts per split, ranked by a hash seeded
    with dataset.seed. in_marginal records membership in the label-blind reservoir.
    Positives retained outside it belong only to the separately sampled positive pool,
    not the nnPU marginal. Full graph neighborhoods are filtered server-side by the
    split, not by this statistical seed sample. The observed positives are the accounts
    revealed in the graph (pu_label), which the split's population reports.
    """
    limits, seed = dataset.seed_limits, dataset.seed
    heaps: dict[str, list[tuple[float, str, dict[str, Any]]]] = {s: [] for s in SPLITS}
    positives: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in scope_accounts(scope):
        if ORACLE_COLUMNS & set(row):
            raise ValueError("Oracle fields cannot enter population metadata")
        account = row["account_id"]
        if not isinstance(account, str) or len(account.encode()) > ID_BYTES:
            raise ValueError("Account pagination/ID violates the transport contract")
        # The server-assigned partitions are the visibility phases of the splits.
        if row["partition"] not in PHASE_SPLIT:
            raise ValueError("Unassigned account in the graph's split")
        split = PHASE_SPLIT[row.pop("partition")]
        row["split"] = split
        counts[split] += 1
        if row["first_seen_ts_ms"] >= min(timestamp(d) for d in dataset.dates[split]):
            continue
        row["in_marginal"] = True
        rank = stable_score(account, seed, RESERVOIR_SALT)
        entry = (-rank, account, row)
        heap = heaps[split]
        if len(heap) < limits[split]:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
        if row["observed_positive"]:
            if len(positives) >= POSITIVE_POOL:
                raise ValueError("Observed-positive pool exceeds the bounded dataset capacity")
            positives[account] = {**row, "in_marginal": False}
    selected = dict(positives)
    selected.update({row["account_id"]: row for heap in heaps.values() for _, _, row in heap})
    if not selected:
        raise ValueError("No eligible accounts in the split")
    return pd.DataFrame(selected.values()).sort_values("account_id").reset_index(drop=True), dict(
        counts
    )
