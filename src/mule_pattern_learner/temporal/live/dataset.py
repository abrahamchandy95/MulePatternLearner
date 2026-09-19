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
from .batching import make_live_batch
from .contract import ContextKey, fingerprint
from .source import ContextStore, QueryExecutor, checked_rows
from .supervision import ObservedLabelSource, GraphObservedLabels, label_summary
from .policy import validate_protocol

ROOT = Path(__file__).resolve().parents[4]
QUERY_FILES = (
    "gsql/features/temporal_fourier64.gsql",
    "gsql/temporal/training_context.gsql",
    "gsql/temporal/training_population.gsql",
    "gsql/temporal/training_scope.gsql",
    "gsql/temporal/training_cutoffs.gsql",
)


def query_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in QUERY_FILES}


def load_prepared(dataset: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Shared training/inference integrity gate for all prepared artifacts."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["status"] != "ready" or manifest["source"]["query_hashes"] != query_hashes():
        raise ValueError("Dataset is incomplete or query sources changed")
    required = [
        ("accounts.parquet", "accounts_sha256"),
        ("observed_labels.parquet", "observed_labels_sha256"),
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


def export_accounts(executor: QueryExecutor, *, include_observed: bool = True) -> pd.DataFrame:
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


def sample_keys(accounts: pd.DataFrame, date: str, manifest: dict[str, Any]) -> list[ContextKey]:
    ms = timestamp(date) - 1
    seq = int(manifest["cutoff_seqs"][date])
    scope = (
        manifest["config"].get("scope_id", "")
        if manifest["config"].get("evaluation_protocol") == "strict_inductive"
        else ""
    )
    phase = {"train": 1, "validation": 2, "test": 3}
    return [
        ContextKey(
            "Account", str(row.account_id), seq, ms, scope, phase[str(row.split)] if scope else 3
        )
        for row in accounts.itertuples(index=False)
    ]


def prepare(
    config: dict[str, Any],
    output: Path,
    executor: QueryExecutor,
    source_counts: dict[str, int],
    labels: ObservedLabelSource | None = None,
) -> dict[str, Any]:
    validate_protocol(config)
    validate_dates(config)
    if not config.get("dataset_id"):
        raise ValueError("A new immutable dataset_id is required after each graph reload/backfill")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_id": config["dataset_id"],
        "source_counts": source_counts,
        "query_hashes": query_hashes(),
        "config_sha256": fingerprint(config),
        "context_storage": config.get("context_storage", "stream"),
        "evaluation_protocol": config["evaluation_protocol"],
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
        meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
    accounts_path = output / "accounts.parquet"
    if not accounts_path.exists() or "accounts_sha256" not in manifest:
        if config["evaluation_protocol"] == "strict_inductive":
            from .cohort import scoped_cohort

            accounts, population_counts = scoped_cohort(executor, config, labels)
            manifest["population_by_split"] = population_counts
        else:
            accounts = assign_groups(
                export_accounts(executor, include_observed=labels is None),
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
        meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
    accounts = pd.read_parquet(accounts_path)
    if digest(accounts_path) != manifest["accounts_sha256"]:
        raise ValueError("Prepared account file changed")
    # Resolve observed labels without exposing any oracle columns to the trainer.
    labels_path = output / "observed_labels.parquet"
    if "observed_labels_sha256" not in manifest:
        observed = (labels or GraphObservedLabels()).read(accounts)
        temporary = labels_path.with_suffix(".pending.parquet")
        observed.to_parquet(temporary, index=False)
        temporary.replace(labels_path)
        manifest["observed_labels_sha256"] = digest(labels_path)
        manifest["known_mules"] = label_summary(observed)
        meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
    elif digest(labels_path) != manifest["observed_labels_sha256"]:
        raise ValueError("Observed label artifact changed")
    if "cutoff_seqs" not in manifest:
        dates = sorted({date for values in config["dates"].values() for date in values})
        result = checked_rows(
            executor.run(
                "temporal_training_cutoffs",
                {
                    "cutoff_times": [timestamp(date) - 1 for date in dates],
                },
            )
        )
        clocks = next(row["last_visible_seqs"] for row in result if "last_visible_seqs" in row)
        manifest["cutoff_seqs"] = {
            date: int(clocks[str(timestamp(date) - 1)]) + 1 for date in dates
        }
        meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if metadata["context_storage"] == "stream":
        manifest["status"] = "ready"
        manifest["cached_contexts"] = 0
        meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    if metadata["context_storage"] != "sqlite":
        raise ValueError("context_storage must be stream or sqlite")
    store = ContextStore(
        output / "contexts.sqlite",
        metadata,
        executor,
        per_relation=int(config.get("per_relation", 2)),
    )
    try:
        for split, dates in config["dates"].items():
            for date in dates:
                eligible = accounts[
                    (accounts["split"] == split) & (accounts["first_seen_ts_ms"] < timestamp(date))
                ]
                keys = sample_keys(eligible, date, manifest)
                for start in range(0, len(keys), int(config.get("prepare_batch_size", 16))):
                    make_live_batch(
                        store,
                        keys[start : start + int(config.get("prepare_batch_size", 16))],
                        fanouts=tuple(config.get("fanouts", [8, 4])),
                    )
                    print(
                        json.dumps(
                            {
                                "preparing": split,
                                "date": date,
                                "accounts": min(
                                    start + int(config.get("prepare_batch_size", 16)), len(keys)
                                ),
                                "total": len(keys),
                                "query_calls": store.query_calls,
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
    meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
