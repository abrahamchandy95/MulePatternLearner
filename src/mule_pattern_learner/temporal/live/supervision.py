"""Production observed-label interfaces. This module never reads oracle truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..common import timestamp

SPLITS = ("train", "validation", "test")
LABEL_COLUMNS = ("account_id", "known_positive", "known_from_ms")


class ObservedLabelSource(Protocol):
    """Provide known positives and discovery times; zero means unlabeled."""

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame: ...
    def positive_ids(self) -> set[str]: ...


@dataclass
class FrameObservedLabels:
    """Adapter for externally supplied observed labels, including experiments."""

    labels: pd.DataFrame

    def positive_ids(self) -> set[str]:
        validate_label_table(self.labels)
        return set(self.labels.loc[self.labels.known_positive.astype(bool), "account_id"])

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return align_observed_labels(metadata, self.labels)


@dataclass
class ParquetObservedLabels:
    path: Path

    def _frame(self) -> pd.DataFrame:
        import pyarrow.parquet as pq

        if pq.ParquetFile(self.path).metadata.num_rows > 100000:
            raise ValueError("Observed-label source exceeds the 100000-row bounded pool")
        return pd.read_parquet(self.path)

    def positive_ids(self) -> set[str]:
        return FrameObservedLabels(self._frame()).positive_ids()

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return align_observed_labels(metadata, self._frame())


class GraphObservedLabels:
    def positive_ids(self) -> set[str]:
        return set()  # Graph-provided positives are discovered while paging.

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        labels = metadata[["account_id", "observed_positive", "known_from_ms"]].rename(
            columns={"observed_positive": "known_positive"}
        )
        return align_observed_labels(metadata, labels)


def validate_label_table(labels: pd.DataFrame) -> None:
    forbidden = {"is_mule", "true_label", "ring_id", "is_mule_masked"} & set(labels.columns)
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
    if "observed_labels_sha256" not in manifest:
        raise ValueError("Legacy simulated-label cache: prepare with an observed-label provider")
    return ParquetObservedLabels(dataset / "observed_labels.parquet").read(accounts)


def visible_labels(labels: pd.DataFrame, date: str) -> np.ndarray:
    return labels.known_positive.to_numpy(bool) & (
        labels.known_from_ms.to_numpy() < timestamp(date)
    )


def label_summary(labels: pd.DataFrame) -> dict[str, int]:
    return {split: int((labels.known_positive & labels.split.eq(split)).sum()) for split in SPLITS}
