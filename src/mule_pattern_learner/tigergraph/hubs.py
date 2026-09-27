"""The read-only hub query and the parsing of its rows into a HubRegistry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..contract.bounds import HUB_CUTOFFS
from ..contract.graph_schema import HUB_COLUMNS, HUB_REASONS
from ..data.hub_registry import HubRegistry, registry_phases
from .executor import CONVERSION_ERRORS, QueryExecutor, checked_rows

HUB_QUERY = "temporal_hub_registry"


def _parse_hubs(
    rows: list[dict[str, Any]], cutoffs: list[int], threshold: int, scope_id: str
) -> pd.DataFrame:
    checked_rows(rows)
    for row in rows:
        if "threshold" in row and int(row["threshold"]) != threshold:
            raise ValueError(
                f"Hub registry echoed threshold={row['threshold']}, expected {threshold}"
            )
        if "scope_id" in row and str(row["scope_id"]) != scope_id:
            raise ValueError(
                f"Hub registry echoed scope_id={row['scope_id']!r}, expected {scope_id!r}"
            )
        if "cutoff_seqs" in row and sorted(map(int, row["cutoff_seqs"])) != cutoffs:
            raise ValueError("Hub registry echoed different cutoffs")
    pages = [row["hubs"] for row in rows if "hubs" in row]
    if not pages:
        raise ValueError("Hub registry response has no hubs field")
    phases = registry_phases(scope_id)
    records: list[dict[str, Any]] = []
    for page in pages:
        for item in page:
            record = dict(item.get("attributes", item))
            try:
                hub: dict[str, Any] = {
                    name: kind(record[name]) for name, kind in HUB_COLUMNS.items()
                }
            except (KeyError, *CONVERSION_ERRORS):
                raise ValueError(f"Malformed hub registry row: {record}") from None
            if (
                not hub["account_id"]
                or hub["cutoff_seq"] not in cutoffs
                or hub["visibility_phase"] not in phases
                or hub["max_visible"] <= threshold
                or hub["max_degree"] < 0
                or hub["reason"] not in HUB_REASONS
            ):
                raise ValueError(f"Hub registry row violates the query contract: {record}")
            records.append(hub)
    return pd.DataFrame(records, columns=list(HUB_COLUMNS))


def query_hub_registry(
    executor: QueryExecutor,
    cutoff_seqs: Iterable[int],
    *,
    threshold: int,
    scope_id: str = "",
    timeout_s: float = 1800.0,
) -> HubRegistry:
    """Run the read-only temporal_hub_registry query for 1..24 root cutoffs.

    With a scope_id the scope must be ready, and rows cover phases 1, 2 and 3;
    without one the counts are unscoped and rows have phase 3.
    """
    cutoffs = sorted({int(value) for value in cutoff_seqs})
    if not HUB_CUTOFFS.holds(len(cutoffs)) or cutoffs[0] <= 0:
        raise ValueError(
            f"Hub registry needs {HUB_CUTOFFS.low}..{HUB_CUTOFFS.high} positive cutoff sequences"
        )
    if threshold < 1:
        raise ValueError("Hub threshold must be positive")
    rows = executor.run(
        HUB_QUERY,
        {"cutoff_seqs": cutoffs, "threshold": threshold, "scope_id": scope_id},
        timeout_s=timeout_s,
    )
    return HubRegistry(
        _parse_hubs(rows, cutoffs, threshold, scope_id),
        cutoff_seqs=cutoffs,
        threshold=threshold,
        scope_id=scope_id,
    )
