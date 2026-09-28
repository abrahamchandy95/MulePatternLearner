"""The port of oracle truth, owned by evaluation; the graph's reader is tigergraph.oracle.

Only evaluation reads truth, so training never sees this port.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..contract.graph_schema import TRUTH_COLUMNS


class TruthReader(Protocol):
    """Ground truth per account, one row each with contract.graph_schema.TRUTH_COLUMNS."""

    def read(self) -> pd.DataFrame: ...


@dataclass
class ParquetTruth:
    path: Path

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


def checked_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """A reader's truth table, checked: its TRUTH_COLUMNS, one row per account.

    Account ids are strings, is_mule and ring_id integers and label_source a string.
    """
    missing = [name for name in TRUTH_COLUMNS if name not in truth.columns]
    if missing:
        raise ValueError(f"Evaluation truth lacks the columns {missing}")
    if truth.account_id.duplicated().any():
        raise ValueError("Evaluation truth must have unique account IDs")
    if not truth.is_mule.isin([-1, 0, 1]).all():
        raise ValueError("Evaluation truth requires integer is_mule (-1 unknown, 0 or 1)")
    return pd.DataFrame(
        {
            "account_id": truth.account_id.astype(str),
            "is_mule": truth.is_mule.astype("int64"),
            "ring_id": truth.ring_id.astype("int64"),
            "label_source": truth.label_source.astype(str),
        }
    ).reset_index(drop=True)
