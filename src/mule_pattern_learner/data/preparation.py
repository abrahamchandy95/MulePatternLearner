"""Resumable dataset preparation inside the frozen scope."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import atomic_write, file_digest
from ..config import RunConfig
from ..paths import DatasetPaths
from ..runtime.progress import emit
from .accounts import scoped_cohort
from .hub_registry import hub_manifest, hub_threshold
from .manifest import (
    dataset_id,
    dataset_settings,
    query_hashes,
    read_manifest,
    write_manifest,
)
from .observed_labels import ORACLE_COLUMNS, label_summary
from .ports import CutoffReader, HubReader, ObservedLabelReader, ScopeReader
from .splits import resolve_cutoffs


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a pending file, then rename it, so a crash never leaves a truncated file."""
    with atomic_write(path) as pending:
        frame.to_parquet(pending, index=False)


def _stage_population(
    config: RunConfig,
    dataset: DatasetPaths,
    manifest: dict[str, Any],
    scope: ScopeReader,
    labels: ObservedLabelReader,
) -> None:
    """Select and write accounts.parquet unless the manifest records it.

    The cohort is bounded seed reservoirs of the scope partitions plus the observed
    positives (accounts.scoped_cohort).
    """
    accounts_path = dataset.accounts
    if not accounts_path.exists() or "accounts_sha256" not in manifest:
        accounts, manifest["population_by_split"] = scoped_cohort(
            scope, config.scope.id, config.dataset, labels
        )
        if ORACLE_COLUMNS & set(accounts.columns):
            raise ValueError(
                "Population query returned oracle fields; install the observed-only query"
            )
        manifest["population_accounts"] = len(accounts)
        accounts = accounts.sort_values("account_id").reset_index(drop=True)
        _write_parquet(accounts, accounts_path)
        manifest["accounts_sha256"] = file_digest(accounts_path)
        manifest["cohort"] = "bounded_internal_deposit_seeds"
        write_manifest(dataset, manifest)


def _stage_labels(
    dataset: DatasetPaths,
    manifest: dict[str, Any],
    labels: ObservedLabelReader,
    accounts: pd.DataFrame,
) -> None:
    """Resolve observed labels without exposing any oracle columns to the trainer."""
    labels_path = dataset.observed_labels
    if "observed_labels_sha256" not in manifest:
        observed = labels.read(accounts)
        _write_parquet(observed, labels_path)
        manifest["observed_labels_sha256"] = file_digest(labels_path)
        manifest["known_mules"] = label_summary(observed)
        write_manifest(dataset, manifest)
    elif file_digest(labels_path) != manifest["observed_labels_sha256"]:
        raise ValueError("Observed label artifact changed")


def _stage_cutoffs(
    config: RunConfig, dataset: DatasetPaths, manifest: dict[str, Any], cutoffs: CutoffReader
) -> None:
    """Resolve the cutoff sequence of every configured date."""
    if "cutoff_seqs" not in manifest:
        manifest["cutoff_seqs"] = resolve_cutoffs(cutoffs, config.dataset.dates.all())
        write_manifest(dataset, manifest)


def _stage_hubs(
    config: RunConfig, dataset: DatasetPaths, manifest: dict[str, Any], hubs: HubReader
) -> None:
    """Query and save the hub registry of the prepared cutoffs and scope."""
    hubs_path = dataset.hubs
    if "hubs_sha256" not in manifest:
        registry = hubs.hub_registry(
            manifest["cutoff_seqs"].values(),
            threshold=hub_threshold(config.sampler),
            scope_id=config.scope.id,
        )
        registry.save(hubs_path)
        manifest.update(hub_manifest(registry, hubs_path))
        write_manifest(dataset, manifest)
        emit({"hub_counts": manifest["hub_counts"]})


def prepare(
    config: RunConfig,
    source_id: str,
    dataset: DatasetPaths,
    source_counts: dict[str, int],
    labels: ObservedLabelReader,
    *,
    scope: ScopeReader,
    cutoffs: CutoffReader,
    hubs: HubReader,
) -> dict[str, Any]:
    """Resumable preparation: the accounts, observed labels, cutoffs and hub registry.

    Contexts are not stored: training requests them from TigerGraph. It reads the
    graph only through its ports: the scope population, the cutoff clocks and the hub
    registry. `source_id` names the data loaded into the graph, and `labels` is the
    pipeline's label source, the labels revealed in the graph for every run. The
    manifest records the dataset's settings and its id (manifest.dataset_settings).
    Each stage writes the manifest when it is done, and a resumed preparation skips
    the stages the manifest records.
    """
    if not source_id:
        raise ValueError("A new immutable source id is required after each graph reload/backfill")
    dataset.root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_id": source_id,
        "source_counts": source_counts,
        "query_hashes": query_hashes(),
        "dataset_id": dataset_id(source_id, config),
        "settings": dataset_settings(source_id, config),
        "scope_id": config.scope.id,
    }
    manifest: dict[str, Any]
    if dataset.manifest.exists():
        manifest = read_manifest(dataset)
        if manifest["source"] != metadata:
            raise ValueError(f"Preparation inputs changed; move {dataset.root} aside")
    else:
        manifest = {
            "source": metadata,
            "status": "preparing",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_manifest(dataset, manifest)
    _stage_population(config, dataset, manifest, scope, labels)
    accounts = pd.read_parquet(dataset.accounts)
    if file_digest(dataset.accounts) != manifest["accounts_sha256"]:
        raise ValueError("Prepared account file changed")
    _stage_labels(dataset, manifest, labels, accounts)
    _stage_cutoffs(config, dataset, manifest, cutoffs)
    _stage_hubs(config, dataset, manifest, hubs)
    manifest["status"] = "ready"
    write_manifest(dataset, manifest)
    return manifest
