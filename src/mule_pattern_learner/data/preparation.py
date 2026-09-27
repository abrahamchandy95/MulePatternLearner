"""Resumable dataset preparation inside the frozen scope."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import atomic_write, file_digest
from ..config import validate_config
from ..contract.fingerprints import fingerprint
from ..contract.graph_schema import context_scope
from ..contract.sampler_plan import SamplerPlan
from ..tigergraph.executor import QueryExecutor
from ..tigergraph.hubs import query_hub_registry
from ..tigergraph.labels import GraphObservedLabels
from .accounts import scoped_cohort
from .hub_registry import HUB_FILE, hub_manifest, hub_threshold
from .manifest import MANIFEST, preparation_view, query_hashes, read_manifest, write_manifest
from .observed_labels import ORACLE_COLUMNS, ObservedLabelSource, label_summary
from .splits import resolve_cutoffs, validate_dates


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a pending file, then rename it, so a crash never leaves a truncated file."""
    with atomic_write(path) as pending:
        frame.to_parquet(pending, index=False)


def _stage_population(
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    executor: QueryExecutor,
    labels: ObservedLabelSource,
) -> None:
    """Select and write accounts.parquet unless the manifest records it.

    The cohort is bounded seed reservoirs of the scope partitions plus the observed
    positives (accounts.scoped_cohort).
    """
    accounts_path = output / "accounts.parquet"
    if not accounts_path.exists() or "accounts_sha256" not in manifest:
        accounts, manifest["population_by_split"] = scoped_cohort(executor, config, labels)
        if ORACLE_COLUMNS & set(accounts.columns):
            raise ValueError(
                "Population query returned oracle fields; install the observed-only query"
            )
        manifest["population_accounts"] = len(accounts)
        accounts = accounts.sort_values("account_id").reset_index(drop=True)
        _write_parquet(accounts, accounts_path)
        manifest["accounts_sha256"] = file_digest(accounts_path)
        manifest["cohort"] = "bounded_internal_deposit_seeds"
        write_manifest(output, manifest)


def _stage_labels(
    output: Path, manifest: dict[str, Any], labels: ObservedLabelSource, accounts: pd.DataFrame
) -> None:
    """Resolve observed labels without exposing any oracle columns to the trainer."""
    labels_path = output / "observed_labels.parquet"
    if "observed_labels_sha256" not in manifest:
        observed = labels.read(accounts)
        _write_parquet(observed, labels_path)
        manifest["observed_labels_sha256"] = file_digest(labels_path)
        manifest["known_mules"] = label_summary(observed)
        write_manifest(output, manifest)
    elif file_digest(labels_path) != manifest["observed_labels_sha256"]:
        raise ValueError("Observed label artifact changed")


def _stage_cutoffs(
    config: dict[str, Any], output: Path, manifest: dict[str, Any], executor: QueryExecutor
) -> None:
    """Resolve the cutoff sequence of every configured date."""
    if "cutoff_seqs" not in manifest:
        dates = sorted({date for values in config["dates"].values() for date in values})
        manifest["cutoff_seqs"] = resolve_cutoffs(executor, dates)
        write_manifest(output, manifest)


def _stage_hubs(
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    executor: QueryExecutor,
    sampler: SamplerPlan,
) -> None:
    """Query and save the hub registry of the prepared cutoffs and scope."""
    hubs_path = output / HUB_FILE
    if "hubs_sha256" not in manifest:
        registry = query_hub_registry(
            executor,
            manifest["cutoff_seqs"].values(),
            threshold=hub_threshold(sampler),
            scope_id=context_scope(config),
        )
        registry.save(hubs_path)
        manifest.update(hub_manifest(registry, hubs_path))
        write_manifest(output, manifest)
        print(json.dumps({"hub_counts": manifest["hub_counts"]}), flush=True)


def prepare(
    config: dict[str, Any],
    output: Path,
    executor: QueryExecutor,
    source_counts: dict[str, int],
    labels: ObservedLabelSource | None = None,
) -> dict[str, Any]:
    """Resumable preparation: cohort, observed labels, cutoffs and hub registry.

    Contexts are not stored: training requests them from TigerGraph. `labels`
    defaults to the labels revealed in the graph. Each stage writes the manifest when
    it is done, and a resumed preparation skips the stages the manifest records.
    """
    config = validate_config(config)
    sampler = SamplerPlan.from_config(config)
    context_scope(config)
    validate_dates(config)
    if not config.get("dataset_id"):
        raise ValueError("A new immutable dataset_id is required after each graph reload/backfill")
    labels = GraphObservedLabels() if labels is None else labels
    output.mkdir(parents=True, exist_ok=True)
    preparation = preparation_view(config)
    metadata = {
        "dataset_id": config["dataset_id"],
        "prepared_id": config.get("prepared_id"),
        "source_counts": source_counts,
        "query_hashes": query_hashes(),
        "preparation_sha256": fingerprint(preparation),
        "preparation": preparation,
        "scope_id": config.get("scope_id", ""),
    }
    manifest: dict[str, Any]
    if (output / MANIFEST).exists():
        manifest = read_manifest(output)
        if manifest["source"] != metadata:
            raise ValueError("Preparation inputs changed; use a new output directory")
    else:
        manifest = {
            "source": metadata,
            "config": config,
            "status": "preparing",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_manifest(output, manifest)
    _stage_population(config, output, manifest, executor, labels)
    accounts_path = output / "accounts.parquet"
    accounts = pd.read_parquet(accounts_path)
    if file_digest(accounts_path) != manifest["accounts_sha256"]:
        raise ValueError("Prepared account file changed")
    _stage_labels(output, manifest, labels, accounts)
    _stage_cutoffs(config, output, manifest, executor)
    _stage_hubs(config, output, manifest, executor, sampler)
    manifest["status"] = "ready"
    write_manifest(output, manifest)
    return manifest
