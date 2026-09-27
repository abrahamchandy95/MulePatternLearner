"""Oracle truth read from the graph, for evaluation only; training never imports it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..contract.server import TRUTH_QUERY
from .executor import QueryExecutor, account_pages


@dataclass
class TigerGraphTruth:
    """Oracle truth paged from the graph's label contract, for evaluation only.

    contract.server.TRUTH_QUERY is the oracle endpoint; training never calls it. An
    account whose label is not known (mule_label_known false) reports is_mule = -1,
    which the evaluators treat as unknown, never as a negative.
    """

    executor: QueryExecutor

    def read(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        pages = account_pages(self.executor, TRUTH_QUERY, {}, timeout_s=900.0)
        for page in pages:
            for row in page:
                known = bool(row["mule_label_known"])
                rows.append(
                    {
                        "account_id": str(row["account_id"]),
                        "is_mule": int(row["is_mule"]) if known else -1,
                    }
                )
        return pd.DataFrame(rows, columns=["account_id", "is_mule"])
