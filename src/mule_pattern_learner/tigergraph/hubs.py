"""The read-only hub query and the parsing of its rows into a HubRegistry (data.ports.HubReader)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..contract.bounds import HUB_CUTOFFS
from ..contract.graph_schema import HUB_COLUMNS
from ..contract.server import HUB_QUERY
from ..data.hub_registry import HubRegistry, registry_phases
from .executor import CONVERSION_ERRORS, QueryExecutor, checked_rows


def _parse_hubs(
    rows: list[dict[str, Any]], cutoffs: list[int], threshold: int, by_phase: bool
) -> pd.DataFrame:
    checked_rows(rows)
    pages = [row["hubs"] for row in rows if "hubs" in row]
    if not pages:
        raise ValueError("Hub registry response has no hubs field")
    phases = registry_phases(by_phase)
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
            ):
                raise ValueError(f"Hub registry row violates the query contract: {record}")
            records.append(hub)
    return pd.DataFrame(records, columns=list(HUB_COLUMNS))


class TigerGraphHubReader:
    """The HubReader of data.ports: the read-only hub query (contract.server.HUB_QUERY)."""

    def __init__(self, executor: QueryExecutor, *, timeout_s: float = 1800.0) -> None:
        self.executor, self.timeout_s = executor, timeout_s

    def hub_registry(
        self, cutoff_seqs: Iterable[int], *, threshold: int, by_phase: bool = False
    ) -> HubRegistry:
        """The hub registry of 1..24 root cutoffs.

        by_phase needs the graph's split ready, and rows cover its phases 1, 2 and 3;
        otherwise counts see every event and rows have phase 0 (no split).
        """
        cutoffs = sorted({int(value) for value in cutoff_seqs})
        if not HUB_CUTOFFS.holds(len(cutoffs)) or cutoffs[0] <= 0:
            raise ValueError(
                f"Hub registry needs {HUB_CUTOFFS.low}..{HUB_CUTOFFS.high} positive cutoff "
                "sequences"
            )
        if threshold < 1:
            raise ValueError("Hub threshold must be positive")
        rows = self.executor.run(
            HUB_QUERY,
            {"cutoff_seqs": cutoffs, "threshold": threshold, "by_phase": by_phase},
            timeout_s=self.timeout_s,
        )
        return HubRegistry(
            _parse_hubs(rows, cutoffs, threshold, by_phase),
            cutoff_seqs=cutoffs,
            threshold=threshold,
            by_phase=by_phase,
        )
