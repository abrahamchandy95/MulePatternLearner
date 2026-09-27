"""Oracle truth sources for evaluation; the graph's is tigergraph.oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


class EvaluationTruthSource(Protocol):
    def read(self) -> pd.DataFrame: ...


@dataclass
class ParquetEvaluationTruth:
    path: Path

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)
