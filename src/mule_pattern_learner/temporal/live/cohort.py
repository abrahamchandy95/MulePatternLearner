"""Bounded seed selection from server-assigned partitions, independent of truth."""

from __future__ import annotations

from collections import Counter
import heapq
from typing import Any

import pandas as pd

from ..common import stable_score, timestamp
from .source import QueryExecutor, checked_rows
from .supervision import ObservedLabelSource, reads_graph_labels

PARTITIONS = {1: "train", 2: "validation", 3: "test"}
DEFAULT_SEED_LIMITS = {"train": 20000, "validation": 2000, "test": 2000}


def cohort_seed(config: dict[str, Any]) -> int:
    """Seed of the label-blind seed reservoirs: `cohort_seed`, else the model `seed`.

    Pin `cohort_seed` to train several model seeds on one prepared cohort.
    """
    value = config.get("cohort_seed")
    return int(config.get("seed", 42) if value is None else value)


def _check_label_fields(row: dict[str, Any], graph_labels: bool) -> None:
    """Label fields must match what the query was asked for, checked while paging.

    Without include_observed the query must emit no label information at all.
    With it, only revealed positives carry a discovery time (see
    supervision.check_graph_label_rows, which checks the finished table too).
    """
    positive = bool(row.get("observed_positive") or False)
    known = int(row.get("known_from_ms") or 0)
    if not graph_labels and (positive or known):
        raise ValueError(
            "temporal_scope_population returned observed labels although include_observed "
            "is false; install the current query (mule-temporal install)"
        )
    if graph_labels and known > 0 and not positive:
        raise ValueError(
            f"Account {row.get('account_id')!r} has known_from_ms > 0 but observed_positive "
            "false: the installed temporal_scope_population predates the masked-label "
            "predicate. Run `mule-temporal install` and prepare again."
        )


def scoped_cohort(
    executor: QueryExecutor, config: dict[str, Any], labels: ObservedLabelSource | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep uniform hash reservoirs plus observed positives, never all account IDs.

    in_marginal records membership in the label-blind reservoir. Positives retained
    outside it belong only to the separately sampled positive pool, not the nnPU
    marginal. Full graph neighborhoods are filtered server-side by scope, not by
    this statistical seed sample. Graph labels are read only for an explicit
    GraphObservedLabels source.
    """
    if labels is None:
        raise ValueError("An explicit observed-label source is required")
    graph_labels = reads_graph_labels(labels)
    limits = config.get("seed_limits", DEFAULT_SEED_LIMITS)
    if set(limits) != set(PARTITIONS.values()) or any(
        type(n) is not int or not 1 <= n <= 20000 for n in limits.values()
    ):
        raise ValueError("seed_limits needs three integer capacities in [1,20000]")
    seed = cohort_seed(config)
    known_ids = labels.positive_ids()
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
                    "include_observed": graph_labels,
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
            _check_label_fields(row, graph_labels)
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
            rank = stable_score(account, seed, "marginal_cohort")
            entry = (-rank, account, row)
            heap = heaps[split]
            if len(heap) < limits[split]:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
            if account in known_ids or (graph_labels and row["observed_positive"]):
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
