"""Observed labels: the revealed positives and their discovery times, never truth.

Every run reads the labels revealed in the graph
(tigergraph.labels.TigerGraphObservedLabelReader, the ports.ObservedLabelReader that
prepare() is given).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..contract.bounds import DATASET_ROWS
from ..contract.clock import timestamp
from ..contract.graph_schema import SPLITS
from ..paths import DatasetPaths

LABEL_COLUMNS = ("account_id", "known_positive", "known_from_ms")
# Ground-truth fields that must never reach training metadata or observed labels.
ORACLE_COLUMNS = frozenset({"is_mule", "true_label", "is_mule_masked", "ring_id"})


def read_bounded_parquet(path: Path, message: str, limit: int = DATASET_ROWS) -> pd.DataFrame:
    """Read a parquet file, refusing (with message) one of more than limit rows before loading."""
    import pyarrow.parquet as pq

    if pq.ParquetFile(path).metadata.num_rows > limit:
        raise ValueError(message)
    return pd.read_parquet(path)


def validate_label_table(labels: pd.DataFrame) -> None:
    forbidden = set(labels.columns) & ORACLE_COLUMNS
    if forbidden:
        raise ValueError(f"Oracle fields cannot enter the observed-label interface: {forbidden}")
    if not set(LABEL_COLUMNS) <= set(labels.columns):
        raise ValueError("Observed labels need account_id, known_positive and known_from_ms")


def align_observed_labels(metadata: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    validate_label_table(labels)
    if labels.account_id.duplicated().any() or metadata.account_id.duplicated().any():
        raise ValueError("Duplicate account IDs in observed labels or metadata")
    if not labels.known_positive.isin([False, True, 0, 1]).all():
        raise ValueError("known_positive must be boolean or 0/1")
    if not set(labels.account_id) <= set(metadata.account_id):
        raise ValueError("Observed labels contain accounts outside this population")
    result = metadata[["account_id", "split"]].merge(
        labels[list(LABEL_COLUMNS)], on="account_id", how="left", validate="one_to_one"
    )
    result["known_positive"] = result.known_positive.fillna(False).astype(bool)
    result["known_from_ms"] = result.known_from_ms.fillna(0)
    clocks = result.known_from_ms.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(clocks).all()
        or (clocks < 0).any()
        or (clocks != np.floor(clocks)).any()
        or (result.known_positive & result.known_from_ms.le(0)).any()
    ):
        raise ValueError("Observed positives require an explicit positive discovery timestamp")
    result["known_from_ms"] = result.known_from_ms.astype("int64")
    result["pu_label"] = (result.known_positive & result.split.eq("train")).astype("int64")
    return result


def load_observed_labels(accounts: pd.DataFrame, dataset: DatasetPaths) -> pd.DataFrame:
    """The prepared observed labels, aligned with the prepared accounts."""
    labels = read_bounded_parquet(
        dataset.observed_labels,
        f"Observed-label source exceeds the {DATASET_ROWS}-row bounded pool",
    )
    return align_observed_labels(accounts, labels)


def visible_labels(labels: pd.DataFrame, date: str) -> np.ndarray:
    return labels.known_positive.to_numpy(bool) & (
        labels.known_from_ms.to_numpy() < timestamp(date)
    )


def label_summary(labels: pd.DataFrame) -> dict[str, int]:
    return {split: int((labels.known_positive & labels.split.eq(split)).sum()) for split in SPLITS}
