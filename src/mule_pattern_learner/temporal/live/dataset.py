"""Resumable cohort preparation inside the frozen scope, and the prepared-data gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mule_pattern_learner.configuration import REPOSITORY_ROOT

from ..common import cutoff_ms, digest, timestamp
from .cohort import cohort_seed, scoped_cohort
from .config_schema import validate_config, without_retired_keys
from .contract import (
    SPLIT_PHASE,
    ContextKey,
    SamplerPlan,
    extraction_plan,
    fingerprint,
    sampler_pools,
)
from .executor import QueryExecutor, checked_rows, printed
from .hubs import HUB_FILE, hub_manifest, hub_threshold, query_hub_registry
from .installation import QUERY_FILES
from .policy import context_scope
from .supervision import (
    ORACLE_COLUMNS,
    GraphObservedLabels,
    ObservedLabelSource,
    label_summary,
    read_bounded_parquet,
)

ROOT = REPOSITORY_ROOT
MANIFEST = "manifest.json"
# Settings that change what preparation produces. Training compares exactly these;
# model, optimisation and transport settings may change between runs.
PREPARATION_KEYS = (
    "dataset_id",
    "prepared_id",
    "dates",
    "seed_limits",
    "scope_id",
    "split_seed",
    "cohort_seed",
    "sampler_pools",
    "extraction_groups",
    "scope_unowned",
)


def read_manifest(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / MANIFEST).read_text())


def write_manifest(dataset: Path, manifest: dict[str, Any]) -> None:
    """Replace the manifest atomically, so a crash never leaves a truncated file."""
    pending = (dataset / MANIFEST).with_suffix(".pending.json")
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(dataset / MANIFEST)


def manifest_digest(dataset: Path) -> str:
    """The manifest's sha256; a checkpoint records it to name its prepared dataset."""
    return digest(dataset / MANIFEST)


def query_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in QUERY_FILES}


def changed_query_files(manifest: dict[str, Any]) -> list[str]:
    """Query files preparation uses whose repository text differs from the one it used.

    A recorded file that preparation no longer uses cannot affect the cohort, so it is
    not compared: retiring a query file leaves existing cohorts usable.
    """
    recorded = manifest.get("source", {}).get("query_hashes", {})
    return sorted(name for name, current in query_hashes().items() if recorded.get(name) != current)


def check_query_hashes(manifest: dict[str, Any], dataset: Path) -> None:
    changed = changed_query_files(manifest)
    if changed:
        raise ValueError(
            f"Prepared dataset {dataset} was built from different GSQL sources "
            f"({', '.join(changed)}). Install the current queries (mule-temporal install), "
            f"then train into a new output, or move {dataset} aside."
        )


def preparation_view(config: dict[str, Any]) -> dict[str, Any]:
    """Normalized values of PREPARATION_KEYS of the validated configuration.

    sampler_pools is what TigerGraph returns per hop. extraction_groups are the groups
    TigerGraph is asked for; the source derives the hop-2 flags from each training
    model, so arms of any architecture can share one preparation.
    """
    config = validate_config(config)
    sampler = SamplerPlan.from_config(config)
    plan = extraction_plan(config)
    view = {
        "dataset_id": config.get("dataset_id"),
        "prepared_id": config.get("prepared_id"),
        "dates": config["dates"],
        "seed_limits": config["seed_limits"],
        "scope_id": config["scope_id"],
        "split_seed": config["split_seed"],
        "cohort_seed": cohort_seed(config),
        "sampler_pools": sampler_pools(sampler),
        "extraction_groups": sorted(plan.groups),
        "scope_unowned": config["scope_unowned"],
    }
    assert tuple(view) == PREPARATION_KEYS
    return json.loads(json.dumps(view))


def preparation_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint(preparation_view(config))


def preparation_mismatches(config: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """PREPARATION_KEYS whose value in config differs from the prepared manifest."""
    recorded = manifest.get("source", {}).get("preparation")
    if not isinstance(recorded, dict):
        return list(PREPARATION_KEYS)
    current = preparation_view(config)
    return [key for key in PREPARATION_KEYS if current[key] != recorded.get(key)]


def load_prepared(dataset: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Shared training/inference integrity gate for all prepared artifacts."""
    manifest = read_manifest(dataset)
    if manifest["status"] != "ready":
        raise ValueError(f"Dataset is incomplete ({manifest['status']}); run prepare again")
    check_query_hashes(manifest, dataset)
    required = [
        ("accounts.parquet", "accounts_sha256"),
        ("observed_labels.parquet", "observed_labels_sha256"),
        (HUB_FILE, "hubs_sha256"),
    ]
    for name, field in required:
        if digest(dataset / name) != manifest[field]:
            raise ValueError(f"Prepared artifact changed: {name}")
    accounts = read_bounded_parquet(
        dataset / "accounts.parquet",
        "Prepared seed metadata exceeds the bounded population contract",
    )
    if ORACLE_COLUMNS & set(accounts.columns):
        raise ValueError("Oracle columns are forbidden in prepared training metadata")
    # A cohort prepared on a removed path is refused, not misread.
    without_retired_keys(manifest["config"])
    context_scope(manifest["config"])
    return manifest, accounts


def validate_dates(config: dict[str, Any]) -> None:
    dates = config["dates"]
    if not all(dates.get(split) for split in ("train", "validation", "test")):
        raise ValueError("Three nonempty chronological splits are required")
    clocks = {split: [timestamp(date) for date in dates[split]] for split in dates}
    if (
        not max(clocks["train"])
        < min(clocks["validation"])
        <= max(clocks["validation"])
        < min(clocks["test"])
    ):
        raise ValueError("Train, validation and test cutoffs overlap or are out of order")


def eligible_mask(accounts: pd.DataFrame, split: str, date: str) -> np.ndarray:
    """Rows of split whose account existed before the cutoff date."""
    return (accounts["split"].to_numpy() == split) & (
        accounts["first_seen_ts_ms"].to_numpy() < timestamp(date)
    )


def marginal_mask(accounts: pd.DataFrame) -> np.ndarray:
    """Rows of the label-blind reservoir (the observed positives outside it are not)."""
    return accounts["in_marginal"].to_numpy(bool)


def sample_keys(accounts: pd.DataFrame, date: str, manifest: dict[str, Any]) -> list[ContextKey]:
    ms = cutoff_ms(date)
    seq = int(manifest["cutoff_seqs"][date])
    scope = context_scope(manifest["config"])
    return [
        ContextKey("Account", str(row.account_id), seq, ms, scope, SPLIT_PHASE[str(row.split)])
        for row in accounts.itertuples(index=False)
    ]


def resolve_cutoffs(executor: QueryExecutor, dates: list[str]) -> dict[str, int]:
    """ContextKey cutoff_seq per date: one past the last event visible at date - 1 ms."""
    result = checked_rows(
        executor.run(
            "temporal_training_cutoffs",
            {"cutoff_times": [cutoff_ms(date) for date in dates]},
            timeout_s=900.0,
        )
    )
    clocks = printed(result, "last_visible_seqs")
    cutoffs = {}
    for date in dates:
        last = int(clocks[str(cutoff_ms(date))])
        if last <= 0:
            raise ValueError(f"No events or entities are visible before {date}")
        cutoffs[date] = last + 1
    return cutoffs


def resolve_cutoff(executor: QueryExecutor, date: str) -> tuple[int, int]:
    """The (cutoff_seq, cutoff_ms) of an unprepared scoring date."""
    return resolve_cutoffs(executor, [date])[date], cutoff_ms(date)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a pending file, then rename it, so a crash never leaves a truncated file."""
    temporary = path.with_suffix(".pending.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _stage_population(
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    executor: QueryExecutor,
    labels: ObservedLabelSource,
) -> None:
    """Select and write accounts.parquet unless the manifest records it.

    The cohort is bounded seed reservoirs of the scope partitions plus the observed
    positives (cohort.scoped_cohort).
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
        manifest["accounts_sha256"] = digest(accounts_path)
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
        manifest["observed_labels_sha256"] = digest(labels_path)
        manifest["known_mules"] = label_summary(observed)
        write_manifest(output, manifest)
    elif digest(labels_path) != manifest["observed_labels_sha256"]:
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
    if digest(accounts_path) != manifest["accounts_sha256"]:
        raise ValueError("Prepared account file changed")
    _stage_labels(output, manifest, labels, accounts)
    _stage_cutoffs(config, output, manifest, executor)
    _stage_hubs(config, output, manifest, executor, sampler)
    manifest["status"] = "ready"
    write_manifest(output, manifest)
    return manifest
