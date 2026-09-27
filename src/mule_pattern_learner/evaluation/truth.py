"""The port of oracle truth, owned by evaluation; the graph's reader is tigergraph.oracle.

Only evaluation reads truth, so training never sees this port.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


class TruthReader(Protocol):
    """Ground truth per account: is_mule 1 or 0, and -1 where it is not known."""

    def read(self) -> pd.DataFrame: ...


@dataclass
class ParquetEvaluationTruth:
    path: Path

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)
