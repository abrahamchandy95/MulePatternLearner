"""Observed-label interfaces: the graph's revealed labels. This module never reads oracle truth.

Labels come from the graph (GraphObservedLabels). prepare() accepts any
ObservedLabelSource, so tests can supply a table of labels instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..common import timestamp
from .contract import SPLITS

LABEL_COLUMNS = ("account_id", "known_positive", "known_from_ms")
# Ground-truth fields that must never reach training metadata or observed labels.
ORACLE_COLUMNS = frozenset({"is_mule", "true_label", "is_mule_masked", "ring_id"})


class ObservedLabelSource(Protocol):
    """Provide known positives and discovery times; zero means unlabeled."""

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame: ...
    def positive_ids(self) -> set[str]: ...


class GraphObservedLabels:
    """Observed labels paged from the graph, the label source of every run.

    This is the only label source for which population queries run with
    include_observed = TRUE. The queries report the revealed positive of the
    account label contract (pu_label = 1: known, is_mule = 1, not masked) and its
    discovery time; every other account has observed_positive false and
    known_from_ms 0.
    """

    def positive_ids(self) -> set[str]:
        return set()  # Graph-provided positives are discovered while paging.

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        check_graph_label_rows(metadata)
        labels = metadata[["account_id", "observed_positive", "known_from_ms"]].rename(
            columns={"observed_positive": "known_positive"}
        )
        return align_observed_labels(metadata, labels)


def check_graph_label_rows(rows: pd.DataFrame) -> None:
    """Fail fast when a population query emits a discovery time for a non-positive.

    The current queries emit known_from_ms only for revealed positives. A
    nonzero clock on any other account means an older query is installed, one
    that reads hidden (masked) labels or reveals which accounts are labeled.
    """
    clocks = pd.to_numeric(rows["known_from_ms"], errors="coerce").fillna(0)
    positive = rows["observed_positive"].eq(True)
    stale = rows.loc[(clocks > 0) & ~positive, "account_id"]
    if len(stale):
        raise ValueError(
            f"{len(stale)} account(s) have known_from_ms > 0 but observed_positive false "
            f"(for example {stale.iloc[0]!r}): the installed population query predates the "
            "masked-label predicate. Run `mule-temporal install` and prepare again."
        )


def reads_graph_labels(labels: ObservedLabelSource) -> bool:
    """Whether population queries may read observed labels (include_observed)."""
    return isinstance(labels, GraphObservedLabels)


def read_bounded_parquet(path: Path, message: str, limit: int = 100_000) -> pd.DataFrame:
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


def load_observed_labels(
    accounts: pd.DataFrame, dataset: Path, manifest: dict[str, Any]
) -> pd.DataFrame:
    """The prepared observed labels, aligned with the prepared accounts."""
    if "observed_labels_sha256" not in manifest:
        raise ValueError("Legacy simulated-label cache: prepare with an observed-label provider")
    labels = read_bounded_parquet(
        dataset / "observed_labels.parquet",
        "Observed-label source exceeds the 100000-row bounded pool",
    )
    return align_observed_labels(accounts, labels)


def visible_labels(labels: pd.DataFrame, date: str) -> np.ndarray:
    return labels.known_positive.to_numpy(bool) & (
        labels.known_from_ms.to_numpy() < timestamp(date)
    )


def label_summary(labels: pd.DataFrame) -> dict[str, int]:
    return {split: int((labels.known_positive & labels.split.eq(split)).sum()) for split in SPLITS}
