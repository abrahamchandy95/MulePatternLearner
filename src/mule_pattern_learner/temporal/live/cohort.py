"""Bounded seed selection from server-assigned partitions, independent of truth."""

from __future__ import annotations

from collections import Counter
import heapq
from typing import Any

import pandas as pd

from ..common import stable_score, timestamp
from .source import QueryExecutor, checked_rows
from .supervision import ObservedLabelSource

PARTITIONS = {1: "train", 2: "validation", 3: "test"}


def scoped_cohort(
    executor: QueryExecutor, config: dict[str, Any], labels: ObservedLabelSource | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep uniform hash reservoirs plus observed positives, never all account IDs.

    in_marginal records membership in the label-blind reservoir. Positives retained
    outside it belong only to the separately sampled positive pool, not the nnPU
    marginal. Full graph neighborhoods are filtered server-side by scope, not by
    this statistical seed sample.
    """
    limits = config.get("seed_limits", {"train": 20000, "validation": 2000, "test": 2000})
    if set(limits) != set(PARTITIONS.values()) or any(
        type(n) is not int or not 1 <= n <= 20000 for n in limits.values()
    ):
        raise ValueError("seed_limits needs three integer capacities in [1,20000]")
    known_ids = labels.positive_ids() if labels else set()
    heaps: dict[str, list[tuple[float, str, dict[str, Any]]]] = {s: [] for s in limits}
    positives: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    after = ""
    while True:
        result = checked_rows(
            executor.run(
                "temporal_scope_population",
                {
                    "scope_id": config["scope_id"],
                    "after_id": after,
                    "batch_size": 10000,
                    "include_observed": labels is None,
                },
            )
        )
        page = next(row["accounts"] for row in result if "accounts" in row)
        if not page:
            break
        if len(page) > 10000:
            raise ValueError("Population page exceeds transport contract")
        for item in page:
            row = dict(item.get("attributes", item))
            if {"is_mule", "true_label", "is_mule_masked", "ring_id"} & set(row):
                raise ValueError("Oracle fields cannot enter population metadata")
            account = row["account_id"]
            if not isinstance(account, str) or account <= after or len(account.encode()) > 1024:
                raise ValueError("Account pagination/ID violates the transport contract")
            after = account
            if row["partition"] not in PARTITIONS:
                raise ValueError("Unassigned account in frozen scope")
            split = PARTITIONS[row.pop("partition")]
            row["split"] = split
            counts[split] += 1
            if row["first_seen_ts_ms"] >= min(timestamp(d) for d in config["dates"][split]):
                continue
            row["in_marginal"] = True
            rank = stable_score(account, int(config.get("seed", 42)), "marginal_cohort")
            entry = (-rank, account, row)
            heap = heaps[split]
            if len(heap) < limits[split]:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
            if account in known_ids or (labels is None and row["observed_positive"]):
                if len(positives) >= 40000:
                    raise ValueError("Observed-positive pool exceeds bounded cohort capacity")
                positives[account] = {**row, "in_marginal": False}
        if len(page) < 10000:
            break
    selected = dict(positives)
    selected.update({row["account_id"]: row for heap in heaps.values() for _, _, row in heap})
    if not selected:
        raise ValueError("No eligible scoped accounts")
    if not known_ids <= set(selected):
        raise ValueError(
            "Observed positives are absent from the population or predate no split cutoff"
        )
    return pd.DataFrame(selected.values()).sort_values("account_id").reset_index(drop=True), dict(
        counts
    )
