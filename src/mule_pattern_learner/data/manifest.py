"""The prepared dataset's manifest, the settings and id of the dataset, the integrity gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import pandas as pd

from ..artifacts import atomic_write, file_digest
from ..config import RunConfig, as_json, differing_settings
from ..contract.fingerprints import fingerprint
from ..contract.graph_schema import context_scope
from ..contract.sampler_plan import sampler_pools
from ..contract.server import QUERY_FILES
from ..paths import GSQL_DIR, DatasetPaths
from .observed_labels import ORACLE_COLUMNS, read_bounded_parquet

# The settings a dataset is prepared from (dataset_settings). Training compares exactly
# these; model, optimisation, transport and runtime settings may change between runs.
DATASET_SETTINGS = ("source_id", "scope", "dataset", "sampler_pools")


def read_manifest(dataset: DatasetPaths) -> dict[str, Any]:
    return json.loads(dataset.manifest.read_text())


def write_manifest(dataset: DatasetPaths, manifest: dict[str, Any]) -> None:
    """Replace the manifest atomically, so a crash never leaves a truncated file."""
    with atomic_write(dataset.manifest) as pending:
        pending.write_text(json.dumps(manifest, indent=2) + "\n")


def manifest_digest(dataset: DatasetPaths) -> str:
    """The manifest's sha256; a checkpoint records it to name its prepared dataset."""
    return file_digest(dataset.manifest)


def query_hashes() -> dict[str, str]:
    return {name: file_digest(GSQL_DIR / name) for name in QUERY_FILES}


def changed_query_files(manifest: dict[str, Any]) -> list[str]:
    """Query files preparation uses whose text is none of the texts it used.

    The manifest records each file's digest by its path, and files are compared by
    digest alone: a file that moved with its text unchanged still matches, as every
    query file did in the layered restructure. A recorded file that preparation no
    longer uses cannot affect the dataset, so it is not compared: retiring a query file
    leaves existing datasets usable.
    """
    recorded = set(manifest.get("source", {}).get("query_hashes", {}).values())
    return sorted(name for name, current in query_hashes().items() if current not in recorded)


def check_query_hashes(manifest: dict[str, Any], dataset: DatasetPaths) -> None:
    changed = changed_query_files(manifest)
    if changed:
        raise ValueError(
            f"Prepared dataset {dataset.root} was built from different GSQL sources "
            f"({', '.join(changed)}). Install the current queries (mule install), "
            f"then move {dataset.root} aside to prepare it again."
        )


def dataset_settings(source_id: str, config: RunConfig) -> dict[str, Any]:
    """What preparation reads, as JSON values: the input of the dataset id.

    The source id names the data loaded into the graph. The scope's id and its rule
    for accounts no party owns decide the split partitions, the dataset section gives
    the cutoffs and seed reservoirs, and sampler_pools is what TigerGraph returns per
    hop. The scope's other settings (create, reveal_per_split, reveal_salt) act once on
    the graph, when a missing scope is created and in the one-time reveal, so they name
    no other dataset. Feature groups are not a dataset setting: a dataset stores no
    contexts, and the source requests each training model's groups, so variants of any
    groups and architecture share one dataset.
    """
    settings = {
        "source_id": source_id,
        "scope": {"id": config.scope.id, "unowned": config.scope.unowned},
        "dataset": as_json(config.dataset),
        "sampler_pools": sampler_pools(config.sampler),
    }
    assert tuple(settings) == DATASET_SETTINGS
    return settings


def dataset_id(source_id: str, config: RunConfig) -> str:
    """The fingerprint of the dataset's settings (dataset_settings)."""
    return fingerprint(dataset_settings(source_id, config))


def recorded_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    """The settings a dataset records (dataset_settings), or a ValueError without them."""
    settings = manifest.get("source", {}).get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The dataset records no dataset settings; prepare a new one")
    return settings


def recorded_dataset_id(manifest: dict[str, Any]) -> str:
    """The dataset id of the settings a dataset records: its directory's name."""
    return fingerprint(recorded_settings(manifest))


@dataclass(frozen=True)
class PreparedSource:
    """The graph a dataset was prepared from, as its manifest records it.

    The vertex counts and the scope: its id, its rule for accounts no party owns, the
    source id it names and the split seed of its partitions.
    """

    counts: dict[str, int]
    scope_id: str
    unowned: str
    source_id: str
    split_seed: int


def prepared_source(manifest: dict[str, Any]) -> PreparedSource:
    """The graph a dataset was prepared from; a ValueError without its settings."""
    settings = recorded_settings(manifest)
    return PreparedSource(
        counts=dict(manifest["source"]["source_counts"]),
        scope_id=settings["scope"]["id"],
        unowned=settings["scope"]["unowned"],
        source_id=settings["source_id"],
        split_seed=settings["dataset"]["split_seed"],
    )


def source_fingerprint(manifest: dict[str, Any]) -> str:
    """The fingerprint of the graph a dataset was prepared from (prepared_source).

    tigergraph.provenance.verify_frozen_source checks the live graph against these
    counts and this scope before a run reads it, so the fingerprint names the frozen
    source whose contexts the dataset's disk tier keeps.
    """
    return fingerprint(asdict(prepared_source(manifest)))


def dataset_mismatches(config: RunConfig, manifest: dict[str, Any]) -> list[str]:
    """The dataset settings in which config differs from those the dataset records.

    The source id is the dataset's own. A manifest that records no settings differs in
    all of them.
    """
    source = manifest.get("source", {})
    recorded = source.get("settings")
    if not isinstance(recorded, dict) or not source.get("source_id"):
        return list(DATASET_SETTINGS)
    return differing_settings(dataset_settings(source["source_id"], config), recorded)


def load_prepared(dataset: DatasetPaths) -> tuple[dict[str, Any], pd.DataFrame]:
    """Shared training/inference integrity gate for all prepared artifacts."""
    manifest = read_manifest(dataset)
    if manifest["status"] != "ready":
        raise ValueError(
            f"Dataset is incomplete ({manifest['status']}); run `mule train` to finish preparing it"
        )
    check_query_hashes(manifest, dataset)
    required = [
        (dataset.accounts, "accounts_sha256"),
        (dataset.observed_labels, "observed_labels_sha256"),
        (dataset.hubs, "hubs_sha256"),
    ]
    for path, field in required:
        if file_digest(path) != manifest[field]:
            raise ValueError(f"Prepared artifact changed: {path.name}")
    accounts = read_bounded_parquet(
        dataset.accounts,
        "Prepared seed metadata exceeds the bounded population contract",
    )
    if ORACLE_COLUMNS & set(accounts.columns):
        raise ValueError("Oracle columns are forbidden in prepared training metadata")
    context_scope(manifest["source"].get("scope_id"))
    return manifest, accounts
