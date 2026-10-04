"""The prepared dataset's manifest, the settings and id of the dataset, the integrity gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import pandas as pd

from ..artifacts import atomic_write, file_digest
from ..config import RunConfig, ScopeConfig, SplitDates, as_json, differing_settings
from ..contract.fingerprints import fingerprint
from ..contract.graph_schema import SPLITS, context_scope
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
    """Query files preparation uses whose text is not the one the dataset recorded for them.

    The manifest records each file's digest by its path under the gsql folder, and each
    file is compared with the digest recorded for its own path: a file recorded under
    another path, or not at all, has changed. A recorded file that preparation no longer
    uses cannot affect the dataset, so it is not compared: retiring a query file leaves
    existing datasets usable.
    """
    recorded = manifest.get("source", {}).get("query_hashes", {})
    return sorted(name for name, current in query_hashes().items() if recorded.get(name) != current)


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

    The source id names the data loaded into the graph. The scope's id, its rule for
    accounts no party owns and its split shares decide the split partitions, its reveal
    settings (reveal_per_split, reveal_salt) the labels the dataset reads, the dataset
    section gives the cutoffs and seed reservoirs, and sampler_pools is what TigerGraph
    returns per hop. scope.create only allows the first run to create a missing scope,
    so it names no other dataset. Feature groups are not a dataset setting: a dataset
    stores no contexts, and the source requests each training model's groups, so
    variants of any groups and architecture share one dataset.
    """
    settings = {
        "source_id": source_id,
        "scope": {
            "id": config.scope.id,
            "unowned": config.scope.unowned,
            "shares": list(config.scope.shares),
            "reveal_per_split": config.scope.reveal_per_split,
            "reveal_salt": config.scope.reveal_salt,
        },
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
    source id it names, and the split seed and the train, validation and test shares of
    its partitions.
    """

    counts: dict[str, int]
    scope_id: str
    unowned: str
    source_id: str
    split_seed: int
    shares: tuple[float, float, float]


def prepared_source(manifest: dict[str, Any]) -> PreparedSource:
    """The graph a dataset was prepared from; a ValueError without its settings.

    A dataset of earlier code records no split shares in its scope settings, and is
    refused with a ValueError that names them: its scope was split 70, 15 and 15%.
    """
    settings = recorded_settings(manifest)
    if "shares" not in settings["scope"]:
        raise ValueError(
            "The dataset records no scope shares (scope.train_share, scope.validation_share "
            "and scope.test_share): it was prepared by earlier code, from a scope split 70, "
            "15 and 15%. Move it aside; `mule train` prepares a new one"
        )
    train, validation, test = settings["scope"]["shares"]
    return PreparedSource(
        counts=dict(manifest["source"]["source_counts"]),
        scope_id=settings["scope"]["id"],
        unowned=settings["scope"]["unowned"],
        source_id=settings["source_id"],
        split_seed=settings["dataset"]["split_seed"],
        shares=(float(train), float(validation), float(test)),
    )


def prepared_reveal(manifest: dict[str, Any]) -> tuple[ScopeConfig, SplitDates]:
    """The scope and split dates of the reveal whose labels a dataset read.

    A dataset of earlier code records no reveal settings, and is refused with a
    ValueError that names them: its labels may come from a reveal of 20 mules per split.
    """
    settings = recorded_settings(manifest)
    scope = settings["scope"]
    if "reveal_salt" not in scope:
        raise ValueError(
            "The dataset records no reveal settings (scope.reveal_per_split and "
            "scope.reveal_salt): it was prepared by earlier code, from labels revealed with "
            "at most 20 mules per split. Move it aside; `mule train` prepares a new one"
        )
    train, validation, test = scope["shares"]
    dates = settings["dataset"]["dates"]
    return (
        ScopeConfig(
            id=scope["id"],
            unowned=scope["unowned"],
            train_share=train,
            validation_share=validation,
            test_share=test,
            reveal_per_split=scope["reveal_per_split"],
            reveal_salt=scope["reveal_salt"],
        ),
        SplitDates(**{split: tuple(dates[split]) for split in SPLITS}),
    )


def source_fingerprint(manifest: dict[str, Any]) -> str:
    """The fingerprint of the graph a dataset was prepared from (prepared_source).

    tigergraph.provenance.verify_frozen_source checks the graph against these
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
