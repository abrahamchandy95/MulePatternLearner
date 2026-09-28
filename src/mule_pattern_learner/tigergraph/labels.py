"""The graph's label contract: revealed labels for training and the contract audit.

TigerGraphObservedLabels is the label source of every run: population queries report the
revealed positives and their discovery times, never a hidden label. validate_supervision
runs the contract audit, which returns only violation counts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..contract.server import LABEL_CONTRACT_QUERY
from ..data.observed_labels import align_observed_labels
from .executor import QueryExecutor, merged_rows


class TigerGraphObservedLabels:
    """Observed labels paged from the graph, the label source of every run.

    Preparation runs the population queries with include_observed = TRUE. They report
    the revealed positive of the account label contract (pu_label = 1: known,
    is_mule = 1, not masked) and its discovery time; every other account has
    observed_positive false and known_from_ms 0.
    """

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        labels = metadata[["account_id", "observed_positive", "known_from_ms"]].rename(
            columns={"observed_positive": "known_positive"}
        )
        return align_observed_labels(metadata, labels)


# Contract violations the label-contract query counts; all must be zero.
VIOLATIONS = ("invalid_mule", "invalid_pu", "invalid_unknown", "invalid_clocks", "invalid_ring")


def validate_supervision(executor: QueryExecutor) -> dict[str, Any]:
    """Label-contract audit counts; raises if any violation counter is nonzero."""
    counts = merged_rows(executor.run(LABEL_CONTRACT_QUERY, {}, timeout_s=900.0))
    bad = {name: int(counts.get(name, 0)) for name in VIOLATIONS if int(counts.get(name, 0))}
    if bad:
        raise ValueError(f"Account label contract violated after the reveal: {bad}")
    return counts
