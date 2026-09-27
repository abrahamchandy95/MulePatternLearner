"""The prepared dataset's manifest, its preparation settings and the integrity gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import validate_config, without_retired_keys
from ..contract.fingerprints import digest, fingerprint
from ..contract.graph_schema import context_scope
from ..contract.sampler_plan import SamplerPlan, sampler_pools
from ..paths import REPOSITORY_ROOT
from ..tigergraph.installer import QUERY_FILES
from .accounts import cohort_seed
from .hub_registry import HUB_FILE
from .observed_labels import ORACLE_COLUMNS, read_bounded_parquet

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

    sampler_pools is what TigerGraph returns per hop. Feature groups are not a
    preparation setting: a preparation stores no contexts, and the source requests each
    training model's groups, so arms of any groups and architecture share one
    preparation.
    """
    config = validate_config(config)
    sampler = SamplerPlan.from_config(config)
    view = {
        "dataset_id": config.get("dataset_id"),
        "prepared_id": config.get("prepared_id"),
        "dates": config["dates"],
        "seed_limits": config["seed_limits"],
        "scope_id": config["scope_id"],
        "split_seed": config["split_seed"],
        "cohort_seed": cohort_seed(config),
        "sampler_pools": sampler_pools(sampler),
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
