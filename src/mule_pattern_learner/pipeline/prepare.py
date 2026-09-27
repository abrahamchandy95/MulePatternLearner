"""Preparation against the graph, with built-in settings, and the query installation.

`mule train` needs only the TigerGraph credentials in .env. On a fresh graph
the first run installs the training queries, creates the frozen scope and reveals the
known mules through the label contract; later runs find all three in place. Settings
are a config.RunConfig (DEFAULT_CONFIG for the command line), and the source id is read
from the scope (or derived from the graph). A dataset is prepared in
data/<dataset id>/ (paths.DatasetPaths.of), where the dataset id is the fingerprint of
the source id and the dataset settings (data.manifest.dataset_id).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..data.manifest import check_query_hashes, dataset_id, read_manifest, write_manifest
from ..data.preparation import prepare
from ..paths import DATA_DIR, DatasetPaths, datasets
from ..tigergraph.cutoffs import TigerGraphCutoffs
from ..tigergraph.hubs import TigerGraphHubs
from ..tigergraph.installer import install, undefined_queries
from ..tigergraph.labels import TigerGraphObservedLabels
from ..tigergraph.provenance import resolve_source_id, source_counts
from ..tigergraph.reveal import ensure_revealed_labels
from ..tigergraph.scope import TigerGraphScope, ensure_scope
from .connect import connect


def install_queries(config: RunConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """What `mule install` does, on a connection with config's retry budgets.

    It adds the scope vertex type if it is missing, installs the training queries whose
    text differs (installer.install) and lists the installed queries that no repository
    file defines, which it leaves in place.
    """
    executor = connect(config.transport)
    result = install(executor)
    result["not_defined"] = undefined_queries(executor)
    return result


def find_datasets(config: RunConfig, data: Path = DATA_DIR) -> list[DatasetPaths]:
    """The datasets under data that config prepares, whatever their source id.

    A dataset directory is named by its dataset id, the fingerprint of its source id
    and its dataset settings; it belongs to config when that name is the dataset id
    config gives its recorded source id.
    """
    found: list[DatasetPaths] = []
    for dataset in datasets(data):
        source_id = read_manifest(dataset).get("source", {}).get("source_id")
        if source_id and dataset.root.name == dataset_id(source_id, config):
            found.append(dataset)
    return found


def prepare_dataset(config: RunConfig, data: Path = DATA_DIR) -> DatasetPaths:
    """The ready dataset of config under data, prepared (or resumed) as needed.

    A ready dataset of config is reused without connecting, but only after its GSQL
    hashes match the current ones. Otherwise the graph is brought to a trainable state
    first: stale queries are installed, the scope is created if missing, and known
    mules are revealed if the graph has none. A dataset being prepared keeps its
    source id; with none, or several (the graph was reloaded), the source id is read
    from the graph.
    """
    found = find_datasets(config, data)
    source_id: str | None = None
    if len(found) == 1:
        (dataset,) = found
        manifest = read_manifest(dataset)
        check_query_hashes(manifest, dataset)
        if manifest["status"] == "ready":
            # The trainer re-verifies artifacts before use. No database connection is needed.
            return dataset
        source_id = manifest["source"]["source_id"]
    executor = connect(config.transport)
    install(executor)
    counts = source_counts(executor)
    if source_id is None:
        source_id = resolve_source_id(executor, config.scope.id, counts)
    dataset = DatasetPaths.of(dataset_id(source_id, config), data)
    split_seed = config.dataset.split_seed
    ensure_scope(executor, config.scope, source_id=source_id, split_seed=split_seed)
    # The reveal draws its splits from the scope partitions.
    ensure_revealed_labels(executor, config.scope, config.dataset.dates)
    counts = source_counts(executor)
    result = prepare(
        config,
        source_id,
        dataset,
        counts,
        TigerGraphObservedLabels(),
        scope=TigerGraphScope(executor),
        cutoffs=TigerGraphCutoffs(executor),
        hub_reader=TigerGraphHubs(executor),
    )
    if source_counts(executor) != counts:
        result["status"] = "source_changed"
        write_manifest(dataset, result)
        raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh dataset")
    return dataset
