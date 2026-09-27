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
from .observed_labels import ORACLE_COLUMNS
from .ports import ObservedLabelReader, ScopeReader


def scope_accounts(
    scope: ScopeReader, scope_id: str, *, include_observed: bool
) -> Iterator[dict[str, Any]]:
    """The scope's accounts in account order, page by page from the scope reader.

    The one pager of the scope population: the seed reservoirs and the final audit both
    read it here. Rows carry observed labels only with include_observed.
    """
    for page in scope.population_pages(scope_id, include_observed=include_observed):
        yield from page


def _check_label_fields(row: dict[str, Any], graph_labels: bool) -> None:
    """Label fields must match what the query was asked for, checked while paging.

    Without include_observed the query must emit no label information at all.
    With it, only revealed positives carry a discovery time (see
    tigergraph.labels.check_graph_label_rows, which checks the finished table too).
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
    scope: ScopeReader,
    scope_id: str,
    dataset: DatasetConfig,
    labels: ObservedLabelReader | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep uniform hash reservoirs plus observed positives, never all account IDs.

    The reservoirs hold dataset.seed_limits accounts per split, ranked by a hash seeded
    with dataset.seed. in_marginal records membership in the label-blind reservoir.
    Positives retained outside it belong only to the separately sampled positive pool,
    not the nnPU marginal. Full graph neighborhoods are filtered server-side by scope,
    not by this statistical seed sample. Graph labels are read only for a label source
    whose labels are the graph's (``from_graph``).
    """
    if labels is None:
        raise ValueError("An explicit observed-label source is required")
    graph_labels = labels.from_graph
    limits, seed = dataset.seed_limits, dataset.seed
    known_ids = labels.positive_ids()
    heaps: dict[str, list[tuple[float, str, dict[str, Any]]]] = {s: [] for s in SPLITS}
    positives: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in scope_accounts(scope, scope_id, include_observed=graph_labels):
        if ORACLE_COLUMNS & set(row):
            raise ValueError("Oracle fields cannot enter population metadata")
        _check_label_fields(row, graph_labels)
        account = row["account_id"]
        if not isinstance(account, str) or len(account.encode()) > ID_BYTES:
            raise ValueError("Account pagination/ID violates the transport contract")
        # Server-assigned scope partitions are the visibility phases of the splits.
        if row["partition"] not in PHASE_SPLIT:
            raise ValueError("Unassigned account in frozen scope")
        split = PHASE_SPLIT[row.pop("partition")]
        row["split"] = split
        counts[split] += 1
        if row["first_seen_ts_ms"] >= min(timestamp(d) for d in dataset.dates[split]):
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
            if len(positives) >= POSITIVE_POOL:
                raise ValueError("Observed-positive pool exceeds bounded cohort capacity")
            positives[account] = {**row, "in_marginal": False}
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
