"""Supervision export, ownership-isolated splits and resumable cache preparation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mule_pattern_learner.configuration import REPOSITORY_ROOT, resolve_path

from ..common import cutoff_ms, digest, stable_score, timestamp
from .batching import build_root_batch, child_key
from .cohort import cohort_seed, scoped_cohort
from .config_schema import DEFAULT_RUN, OPERATIONAL_DEFAULTS, fanouts, setting, split_seed
from .contract import (
    SPLIT_PHASE,
    ContextKey,
    FeaturePlan,
    SamplerPlan,
    extraction_plan,
    fingerprint,
    sampler_pools,
)
from .executor import QueryExecutor, account_pages, checked_rows, printed
from .hubs import (
    HUB_FILE,
    HubRegistry,
    hub_manifest,
    hub_threshold,
    load_hub_registry,
    query_hub_registry,
)
from .installation import QUERY_FILES
from .policy import context_scope, validate_protocol
from .source import MAX_FETCH_KEYS, ContextStore, context_store
from .supervision import (
    ORACLE_COLUMNS,
    ObservedLabelSource,
    label_source,
    label_summary,
    missing_label_source,
    read_bounded_parquet,
    reads_graph_labels,
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
    "evaluation_protocol",
    "scope_id",
    "split_seed",
    "cohort_seed",
    "label_policy",
    "observed_labels",
    "context_storage",
    "sampler_pools",
    "extraction_groups",
    "scope_unowned",
    "sqlite_selection",
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
    """Query files whose repository text differs from the one used for preparation."""
    recorded = manifest.get("source", {}).get("query_hashes", {})
    current = query_hashes()
    return sorted(
        name for name in set(recorded) | set(current) if recorded.get(name) != current.get(name)
    )


def check_query_hashes(manifest: dict[str, Any], dataset: Path) -> None:
    changed = changed_query_files(manifest)
    if changed:
        raise ValueError(
            f"Prepared dataset {dataset} was built from different GSQL sources "
            f"({', '.join(changed)}). Install the current queries (mule-temporal install), "
            f"then train into a new output, or move {dataset} aside."
        )


def preparation_view(config: dict[str, Any]) -> dict[str, Any]:
    """Normalized values of PREPARATION_KEYS, with defaults resolved.

    observed_labels is the label file's content hash; a configured file that is
    missing raises, because the hash check cannot be skipped. sampler_pools is what
    TigerGraph returns per hop. extraction_groups are the groups TigerGraph is
    asked for; streaming derives the hop-2 flags from each training model, so
    arms of any architecture can share one streamed preparation.
    sqlite_selection records the fanouts, sampler and extraction architecture
    that decide what a SQLite cache holds (None for streaming).
    """
    sampler = SamplerPlan.from_config(config)
    plan = extraction_plan(config)
    labels = config.get("observed_labels")
    storage = config.get("context_storage", OPERATIONAL_DEFAULTS["context_storage"])
    if labels:
        path = resolve_path(labels)
        if not path.is_file():
            raise ValueError(missing_label_source(path))
        label_hash = digest(path)
    else:
        label_hash = None
    view = {
        "dataset_id": config.get("dataset_id"),
        "prepared_id": config.get("prepared_id"),
        "dates": config.get("dates"),
        "seed_limits": config.get("seed_limits", DEFAULT_RUN["seed_limits"]),
        "evaluation_protocol": config.get("evaluation_protocol"),
        "scope_id": config.get("scope_id", ""),
        "split_seed": split_seed(config),
        "cohort_seed": cohort_seed(config),
        "label_policy": setting(config, "label_policy"),
        "observed_labels": label_hash,
        "context_storage": storage,
        "sampler_pools": sampler_pools(sampler),
        "extraction_groups": sorted(plan.groups),
        "scope_unowned": config.get("scope_unowned", OPERATIONAL_DEFAULTS["scope_unowned"]),
        "sqlite_selection": (
            {
                "fanouts": list(fanouts(config)),
                "sampler": sampler.fingerprint(),
                "architecture": plan.architecture,
            }
            if storage == "sqlite"
            else None
        ),
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
    if manifest["source"].get("context_storage") != "stream":
        required.append(("contexts.sqlite", "cache_sha256"))
    for name, field in required:
        if field not in manifest or digest(dataset / name) != manifest[field]:
            raise ValueError(f"Prepared artifact changed or legacy oracle cache: {name}")
    accounts = read_bounded_parquet(
        dataset / "accounts.parquet",
        "Prepared seed metadata exceeds the bounded population contract",
    )
    if ORACLE_COLUMNS & set(accounts.columns):
        raise ValueError("Oracle columns are forbidden in prepared training metadata")
    validate_protocol(manifest["config"])
    return manifest, accounts


def export_accounts(executor: QueryExecutor, *, include_observed: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for rows in account_pages(
        executor, "temporal_training_population", {"include_observed": include_observed}
    ):
        if len(records) + len(rows) > 100000:
            raise ValueError(
                "Unbounded legacy population export refused; use strict scoped seed reservoirs"
            )
        records.extend(rows)
        print(json.dumps({"exported_accounts": len(records)}), flush=True)
    if not records:
        raise ValueError("No internal deposit accounts found")
    return pd.DataFrame(records)


def assign_groups(accounts: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    """Union all co-owners, including repeated tenures, before selecting cohorts."""
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for account in accounts.to_dict("records"):
        account_key = "Account:" + account["account_id"]
        root(account_key)
        for owner in account["owner_ids"]:
            a, b = root(account_key), root("Party:" + owner)
            parent[max(a, b)] = min(a, b)
    result = accounts.copy()
    result["group_id"] = [root("Account:" + account) for account in accounts["account_id"]]
    scores = result["group_id"].map(lambda v: stable_score(str(v), split_seed, "split"))
    result["split"] = np.where(scores < 0.7, "train", np.where(scores < 0.85, "validation", "test"))
    return result


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
    """Rows of the label-blind reservoir; every row when the cohort records none."""
    if "in_marginal" in accounts:
        return accounts["in_marginal"].to_numpy(bool)
    return np.ones(len(accounts), dtype=bool)


def sample_keys(accounts: pd.DataFrame, date: str, manifest: dict[str, Any]) -> list[ContextKey]:
    ms = cutoff_ms(date)
    seq = int(manifest["cutoff_seqs"][date])
    scope = context_scope(manifest["config"])
    return [
        ContextKey(
            "Account",
            str(row.account_id),
            seq,
            ms,
            scope,
            SPLIT_PHASE[str(row.split)] if scope else 3,
        )
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


def cache_contexts(
    store: ContextStore,
    keys: list[ContextKey],
    *,
    fanouts: tuple[int, int],
    plan: FeaturePlan,
    sampler: SamplerPlan,
    hubs: HubRegistry,
) -> None:
    """Fill a SQLite cache with every context training and evaluation can request.

    Deterministic policies (recent, stratified) cache what their fixed selection
    reaches. The resample policy draws children anew each step, so every candidate
    child of every root is cached (hub stubs are never fetched). Rejected contexts
    are cached as status rows; training drops rejected roots and masks children.
    """
    if sampler.policy != "resample":
        build_root_batch(
            store,
            keys,
            fanouts=fanouts,
            device="cpu",
            plan=plan,
            sampler=sampler,
            hubs=hubs,
            mode="eval",
        )
        return
    rows = store.fetch(keys, hop=1)
    if plan.architecture == "summary":
        return
    children = list(
        dict.fromkeys(
            child_key(message, key)
            for key, row in zip(keys, rows, strict=True)
            if row is not None
            for message in row["messages"]
            if not hubs.is_stub(
                message["node_type"],
                message["node_id"],
                key.cutoff_seq,
                key.batch_phase,
            )
        )
    )
    for start in range(0, len(children), MAX_FETCH_KEYS):
        store.fetch(children[start : start + MAX_FETCH_KEYS], hop=2)


def select_population(
    config: dict[str, Any], executor: QueryExecutor, labels: ObservedLabelSource
) -> tuple[pd.DataFrame, str, dict[str, int] | None]:
    """The cohort of the configured protocol: accounts, cohort name and per-split counts.

    strict_inductive keeps bounded seed reservoirs of the scope partitions;
    shared_history exports every internal deposit account and splits ownership
    groups by hash (no per-split counts).
    """
    if config["evaluation_protocol"] == "strict_inductive":
        accounts, population_counts = scoped_cohort(executor, config, labels)
        return accounts, "bounded_internal_deposit_seeds", population_counts
    accounts = assign_groups(
        export_accounts(executor, include_observed=reads_graph_labels(labels)), split_seed(config)
    )
    return accounts, "internal_deposit_accounts", None


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
    """Select and write accounts.parquet unless the manifest records it."""
    accounts_path = output / "accounts.parquet"
    if not accounts_path.exists() or "accounts_sha256" not in manifest:
        accounts, cohort, population_counts = select_population(config, executor, labels)
        if population_counts is not None:
            manifest["population_by_split"] = population_counts
        if ORACLE_COLUMNS & set(accounts.columns):
            raise ValueError(
                "Population query returned oracle fields; install the observed-only query"
            )
        manifest["population_accounts"] = len(accounts)
        accounts = accounts.sort_values("account_id").reset_index(drop=True)
        _write_parquet(accounts, accounts_path)
        manifest["accounts_sha256"] = digest(accounts_path)
        manifest["cohort"] = cohort
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


def _stage_contexts(
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    executor: QueryExecutor,
    accounts: pd.DataFrame,
    plan: FeaturePlan,
    sampler: SamplerPlan,
) -> None:
    """Fill contexts.sqlite for every split and date; the cache skips what it holds."""
    hubs = load_hub_registry(output, manifest)
    step = int(config.get("prepare_batch_size", OPERATIONAL_DEFAULTS["prepare_batch_size"]))
    store = context_store(
        output / "contexts.sqlite",
        metadata,
        config,
        plan=extraction_plan(config),
        sampler=sampler,
        executor=executor,
    )
    try:
        for split, dates in config["dates"].items():
            for date in dates:
                keys = sample_keys(accounts[eligible_mask(accounts, split, date)], date, manifest)
                for start in range(0, len(keys), step):
                    cache_contexts(
                        store,
                        keys[start : start + step],
                        fanouts=fanouts(config),
                        plan=plan,
                        sampler=sampler,
                        hubs=hubs,
                    )
                    print(
                        json.dumps(
                            {
                                "preparing": split,
                                "date": date,
                                "accounts": min(start + step, len(keys)),
                                "total": len(keys),
                                "query_calls": store.query_calls,
                                "rejections": dict(store.rejections),
                            }
                        ),
                        flush=True,
                    )
        manifest["cached_contexts"] = store.conn.execute(
            "SELECT COUNT(*) FROM contexts"
        ).fetchone()[0]
    finally:
        store.close()


def prepare(
    config: dict[str, Any],
    output: Path,
    executor: QueryExecutor,
    source_counts: dict[str, int],
    labels: ObservedLabelSource | None = None,
) -> dict[str, Any]:
    """Resumable preparation: cohort, observed labels, cutoffs, hub registry, cache.

    `labels` defaults to the source configured by label_source(config); there is
    no implicit graph-label fallback. Each stage writes the manifest when it is
    done, and a resumed preparation skips the stages the manifest records.
    """
    plan = FeaturePlan.from_config(config)
    sampler = SamplerPlan.from_config(config)
    validate_protocol(config)
    validate_dates(config)
    if not config.get("dataset_id"):
        raise ValueError("A new immutable dataset_id is required after each graph reload/backfill")
    storage = config.get("context_storage", OPERATIONAL_DEFAULTS["context_storage"])
    if storage not in ("stream", "sqlite"):
        raise ValueError("context_storage must be stream or sqlite")
    labels = label_source(config) if labels is None else labels
    output.mkdir(parents=True, exist_ok=True)
    preparation = preparation_view(config)
    metadata = {
        "dataset_id": config["dataset_id"],
        "prepared_id": config.get("prepared_id"),
        "source_counts": source_counts,
        "query_hashes": query_hashes(),
        "preparation_sha256": fingerprint(preparation),
        "preparation": preparation,
        "context_storage": storage,
        "evaluation_protocol": config["evaluation_protocol"],
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
    if storage == "stream":
        manifest["status"] = "ready"
        manifest["cached_contexts"] = 0
        write_manifest(output, manifest)
        return manifest
    _stage_contexts(config, output, manifest, metadata, executor, accounts, plan, sampler)
    manifest["status"] = "ready"
    manifest["cache_sha256"] = digest(output / "contexts.sqlite")
    write_manifest(output, manifest)
    return manifest
