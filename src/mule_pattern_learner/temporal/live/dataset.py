"""Supervision export, ownership-isolated splits and resumable cache preparation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import digest
from ..common import stable_score
from ..common import timestamp
from .batching import child_key
from .config_schema import OPERATIONAL_DEFAULTS
from .contract import ContextKey, FeaturePlan, SamplerPlan, fingerprint
from .hubs import HUB_FILE, HubRegistry, hub_manifest, hub_threshold, query_hub_registry
from .source import (
    MAX_FETCH_KEYS,
    ContextStore,
    QueryExecutor,
    checked_rows,
    run_query,
    sampler_pools,
)
from .supervision import (
    ObservedLabelSource,
    label_source,
    label_summary,
    missing_label_source,
    reads_graph_labels,
)
from .policy import validate_protocol

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCOPE_UNOWNED: str = OPERATIONAL_DEFAULTS["scope_unowned"]
QUERY_FILES = (
    "gsql/features/temporal_fourier64.gsql",
    "gsql/temporal/training_context.gsql",
    "gsql/temporal/training_population.gsql",
    "gsql/temporal/training_scope.gsql",
    "gsql/temporal/training_cutoffs.gsql",
    "gsql/temporal/hub_registry.gsql",
)
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
            "then prepare into a new directory: set a new prepared_id, or move "
            f"{dataset} aside."
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
    from mule_pattern_learner.configuration import resolve_path

    from .cohort import DEFAULT_SEED_LIMITS, cohort_seed
    from .contract import SamplerPlan
    from .source import extraction_plan

    sampler = SamplerPlan.from_config(config)
    plan = extraction_plan(config)
    labels = config.get("observed_labels")
    storage = config.get("context_storage", "stream")
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
        "seed_limits": config.get("seed_limits", DEFAULT_SEED_LIMITS),
        "evaluation_protocol": config.get("evaluation_protocol"),
        "scope_id": config.get("scope_id", ""),
        "split_seed": int(config.get("split_seed", 42)),
        "cohort_seed": cohort_seed(config),
        "label_policy": config.get("label_policy", "observed"),
        "observed_labels": label_hash,
        "context_storage": storage,
        "sampler_pools": sampler_pools(sampler),
        "extraction_groups": sorted(plan.groups),
        "scope_unowned": config.get("scope_unowned", DEFAULT_SCOPE_UNOWNED),
        "sqlite_selection": (
            {
                "fanouts": list(config.get("fanouts", [8, 4])),
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
    manifest = json.loads((dataset / "manifest.json").read_text())
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
    import pyarrow.parquet as pq

    if pq.ParquetFile(dataset / "accounts.parquet").metadata.num_rows > 100000:
        raise ValueError("Prepared seed metadata exceeds the bounded population contract")
    accounts = pd.read_parquet(dataset / "accounts.parquet")
    if {"is_mule", "true_label", "is_mule_masked", "ring_id"} & set(accounts.columns):
        raise ValueError("Oracle columns are forbidden in prepared training metadata")
    validate_protocol(manifest["config"])
    return manifest, accounts


def export_accounts(executor: QueryExecutor, *, include_observed: bool = False) -> pd.DataFrame:
    after = ""
    records: list[dict[str, Any]] = []
    while True:
        result = checked_rows(
            executor.run(
                "temporal_training_population",
                {
                    "after_id": after,
                    "batch_size": 10000,
                    "include_observed": include_observed,
                },
            )
        )
        page = next(row["accounts"] for row in result if "accounts" in row)
        if not page:
            break
        rows = [item.get("attributes", item) for item in page]
        ids = [row["account_id"] for row in rows]
        if ids != sorted(set(ids)) or ids[0] <= after:
            raise ValueError("Account pagination is not strictly ordered")
        if len(records) + len(rows) > 100000:
            raise ValueError(
                "Unbounded legacy population export refused; use strict scoped seed reservoirs"
            )
        records.extend(rows)
        after = ids[-1]
        print(json.dumps({"exported_accounts": len(records)}), flush=True)
        if len(page) < 10000:
            break
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


def context_scope(config: dict[str, Any]) -> str:
    """The scope_id of every context of a preparation: the scope for strict_inductive."""
    if config.get("evaluation_protocol") != "strict_inductive":
        return ""
    return str(config.get("scope_id") or "")


def sample_keys(accounts: pd.DataFrame, date: str, manifest: dict[str, Any]) -> list[ContextKey]:
    ms = timestamp(date) - 1
    seq = int(manifest["cutoff_seqs"][date])
    scope = context_scope(manifest["config"])
    phase = {"train": 1, "validation": 2, "test": 3}
    return [
        ContextKey(
            "Account", str(row.account_id), seq, ms, scope, phase[str(row.split)] if scope else 3
        )
        for row in accounts.itertuples(index=False)
    ]


def _write(path: Path, manifest: dict[str, Any]) -> None:
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(path)


def resolve_cutoffs(executor: QueryExecutor, dates: list[str]) -> dict[str, int]:
    """ContextKey cutoff_seq per date: one past the last event visible at date - 1 ms."""
    result = checked_rows(
        run_query(
            executor,
            "temporal_training_cutoffs",
            {"cutoff_times": [timestamp(date) - 1 for date in dates]},
            timeout_s=900.0,
        )
    )
    clocks = next(row["last_visible_seqs"] for row in result if "last_visible_seqs" in row)
    cutoffs = {}
    for date in dates:
        last = int(clocks[str(timestamp(date) - 1)])
        if last <= 0:
            raise ValueError(f"No events or entities are visible before {date}")
        cutoffs[date] = last + 1
    return cutoffs


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
        from .predictor import build_root_batch

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
    # The phase rule of make_live_batch: a root's own phase when scoped, 3 otherwise.
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
                key.visibility_phase if key.scope_id else 3,
            )
        )
    )
    for start in range(0, len(children), MAX_FETCH_KEYS):
        store.fetch(children[start : start + MAX_FETCH_KEYS], hop=2)


def prepare(
    config: dict[str, Any],
    output: Path,
    executor: QueryExecutor,
    source_counts: dict[str, int],
    labels: ObservedLabelSource | None = None,
) -> dict[str, Any]:
    """Resumable preparation: cohort, observed labels, cutoffs, hub registry, cache.

    `labels` defaults to the source configured by label_source(config); there is
    no implicit graph-label fallback.
    """
    from .source import extraction_plan

    plan = FeaturePlan.from_config(config)
    sampler = SamplerPlan.from_config(config)
    validate_protocol(config)
    validate_dates(config)
    if not config.get("dataset_id"):
        raise ValueError("A new immutable dataset_id is required after each graph reload/backfill")
    storage = config.get("context_storage", "stream")
    if storage not in ("stream", "sqlite"):
        raise ValueError("context_storage must be stream or sqlite")
    labels = label_source(config) if labels is None else labels
    graph_labels = reads_graph_labels(labels)
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
    meta_path = output / "manifest.json"
    manifest: dict[str, Any]
    if meta_path.exists():
        manifest = json.loads(meta_path.read_text())
        if manifest["source"] != metadata:
            raise ValueError("Preparation inputs changed; use a new output directory")
    else:
        manifest = {
            "source": metadata,
            "config": config,
            "status": "preparing",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write(meta_path, manifest)
    accounts_path = output / "accounts.parquet"
    if not accounts_path.exists() or "accounts_sha256" not in manifest:
        if config["evaluation_protocol"] == "strict_inductive":
            from .cohort import scoped_cohort

            accounts, population_counts = scoped_cohort(executor, config, labels)
            manifest["population_by_split"] = population_counts
        else:
            accounts = assign_groups(
                export_accounts(executor, include_observed=graph_labels),
                int(config.get("split_seed", 42)),
            )
        if {"is_mule", "true_label", "is_mule_masked", "ring_id"} & set(accounts.columns):
            raise ValueError(
                "Population query returned oracle fields; install the observed-only query"
            )
        manifest["population_accounts"] = len(accounts)
        accounts = accounts.sort_values("account_id").reset_index(drop=True)
        temporary = accounts_path.with_suffix(".pending.parquet")
        accounts.to_parquet(temporary, index=False)
        temporary.replace(accounts_path)
        manifest["accounts_sha256"] = digest(accounts_path)
        manifest["cohort"] = (
            "bounded_internal_deposit_seeds"
            if config["evaluation_protocol"] == "strict_inductive"
            else "internal_deposit_accounts"
        )
        _write(meta_path, manifest)
    accounts = pd.read_parquet(accounts_path)
    if digest(accounts_path) != manifest["accounts_sha256"]:
        raise ValueError("Prepared account file changed")
    # Resolve observed labels without exposing any oracle columns to the trainer.
    labels_path = output / "observed_labels.parquet"
    if "observed_labels_sha256" not in manifest:
        observed = labels.read(accounts)
        temporary = labels_path.with_suffix(".pending.parquet")
        observed.to_parquet(temporary, index=False)
        temporary.replace(labels_path)
        manifest["observed_labels_sha256"] = digest(labels_path)
        manifest["known_mules"] = label_summary(observed)
        _write(meta_path, manifest)
    elif digest(labels_path) != manifest["observed_labels_sha256"]:
        raise ValueError("Observed label artifact changed")
    if "cutoff_seqs" not in manifest:
        dates = sorted({date for values in config["dates"].values() for date in values})
        manifest["cutoff_seqs"] = resolve_cutoffs(executor, dates)
        _write(meta_path, manifest)
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
        _write(meta_path, manifest)
        print(json.dumps({"hub_counts": manifest["hub_counts"]}), flush=True)
    if storage == "stream":
        manifest["status"] = "ready"
        manifest["cached_contexts"] = 0
        _write(meta_path, manifest)
        return manifest
    from .hubs import load_hub_registry

    hubs = load_hub_registry(output, manifest)
    step = int(config.get("prepare_batch_size", 16))
    store = ContextStore(
        output / "contexts.sqlite",
        metadata,
        executor,
        plan=extraction_plan(config),
        sampler=sampler,
        request_batch_size=int(config.get("request_batch_size", 16)),
        encoding_check_every=int(config.get("encoding_check_every", 64)),
    )
    try:
        for split, dates in config["dates"].items():
            for date in dates:
                eligible = accounts[
                    (accounts["split"] == split) & (accounts["first_seen_ts_ms"] < timestamp(date))
                ]
                keys = sample_keys(eligible, date, manifest)
                for start in range(0, len(keys), step):
                    cache_contexts(
                        store,
                        keys[start : start + step],
                        fanouts=tuple(config.get("fanouts", [8, 4])),  # type: ignore[arg-type]
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
    manifest["status"] = "ready"
    manifest["cache_sha256"] = digest(output / "contexts.sqlite")
    _write(meta_path, manifest)
    return manifest
