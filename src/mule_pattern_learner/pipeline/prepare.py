"""Preparation against the graph, with built-in settings, and the query installation.

`mule train` needs only the TigerGraph credentials in .env. On a fresh graph
the first run installs the training queries, creates the graph's split and reveals the
known mules through the label contract; later runs find all three in place. Settings
are a config.RunConfig (DEFAULT_CONFIG for the command line). A dataset is prepared in
data/<dataset id>/ (paths.DatasetPaths.of), where the dataset id is the fingerprint of
the graph's vertex counts and the dataset settings (data.manifest.dataset_id).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..data.manifest import check_query_hashes, dataset_id, read_manifest, write_manifest
from ..data.preparation import prepare
from ..paths import DATA_DIR, DatasetPaths, datasets
from ..runtime.progress import emit, recording
from ..tigergraph.cutoffs import TigerGraphCutoffReader
from ..tigergraph.hubs import TigerGraphHubReader
from ..tigergraph.installer import drop_retired, install, undefined_queries
from ..tigergraph.labels import TigerGraphObservedLabelReader
from ..tigergraph.provenance import source_counts
from ..tigergraph.reveal import ensure_revealed_labels
from ..tigergraph.scope import TigerGraphScopeReader, ensure_scope
from .connect import Session, connect


def install_queries(config: RunConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """What `mule install` does, on a connection with config's retry budgets.

    It adds the scope vertex type if it is missing and installs the training queries
    whose text differs (installer.install). It is the only command that replaces scope
    types that differ from gsql/schema/scope_vertex.gsql, while the graph holds no scope
    vertex, and then installs every training query (installer.replace_scope_types).
    Once that has passed, so every renamed query is installed, it drops the retired
    queries still installed (installer.drop_retired): it is the only command that drops
    them, because code from before the rename calls them. Last it lists the installed
    queries that no repository file defines, which it leaves in place.
    """
    executor = connect(config.transport)
    result = install(executor, replace_scope=True)
    result["dropped"] = drop_retired(executor)
    result["not_defined"] = undefined_queries(executor)
    return result


def find_datasets(config: RunConfig, data: Path = DATA_DIR) -> list[DatasetPaths]:
    """The datasets under data that config prepares, whatever graph they were prepared from.

    A dataset directory is named by its dataset id, the fingerprint of its graph's
    vertex counts and its dataset settings; it belongs to config when that name is the
    dataset id config gives its recorded counts.
    """
    found: list[DatasetPaths] = []
    for dataset in datasets(data):
        counts = read_manifest(dataset).get("source", {}).get("source_counts")
        if counts and dataset.root.name == dataset_id(counts, config):
            found.append(dataset)
    return found


def prepare_dataset(
    config: RunConfig, data: Path = DATA_DIR, *, session: Session | None = None
) -> DatasetPaths:
    """The ready dataset of config under data, prepared (or resumed) as needed.

    A ready dataset of config is reused without connecting, but only after its GSQL
    hashes match the current ones. Otherwise the graph is brought to a trainable state
    first, on the session's connection if there is one: stale queries are installed,
    the split is created if missing, and known mules are revealed if the graph has none
    or the split is new (tigergraph.reveal.ensure_revealed_labels). The dataset is then
    the one of the graph's vertex counts, resumed if it is being prepared. The events of
    the split, the reveal and the preparation are recorded in the dataset's
    events.jsonl, and a `dataset` event names the dataset ready for use, prepared or
    reused, with its known mules.
    """
    found = find_datasets(config, data)
    if len(found) == 1:
        (dataset,) = found
        manifest = read_manifest(dataset)
        check_query_hashes(manifest, dataset)
        if manifest["status"] == "ready":
            # The trainer re-verifies artifacts before use. No database connection is needed.
            announce(dataset, manifest, "reused", session)
            return dataset
    executor = session.executor() if session is not None else connect(config.transport)
    install(executor)
    counts = source_counts(executor)
    dataset = DatasetPaths.of(dataset_id(counts, config), data)
    split_seed = config.dataset.split_seed
    dataset.root.mkdir(parents=True, exist_ok=True)
    with recording(dataset.events):
        created = ensure_scope(executor, config.scope, split_seed=split_seed)
        # The reveal draws its splits from the split's partitions, so a new split's are
        # revealed again.
        ensure_revealed_labels(executor, config.scope, config.dataset.dates, new_split=created)
        result = prepare(
            config,
            dataset,
            counts,
            TigerGraphObservedLabelReader(),
            scope=TigerGraphScopeReader(executor),
            cutoffs=TigerGraphCutoffReader(executor),
            hub_reader=TigerGraphHubReader(executor),
        )
        if source_counts(executor) != counts:
            result["status"] = "source_changed"
            write_manifest(dataset, result)
            raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh dataset")
        announce(dataset, result, "prepared", session)
    return dataset


def announce(
    dataset: DatasetPaths, manifest: dict[str, Any], how: str, session: Session | None
) -> None:
    """Emit the `dataset` event of a ready dataset: its id, how it came and its known mules.

    ``how`` is "prepared" for a dataset prepared now and "reused" for one found ready. A
    session names each of its datasets once (Session.announced).
    """
    identity = dataset.root.name
    if session is not None:
        if identity in session.announced:
            return
        session.announced.add(identity)
    emit(
        {
            "event": "dataset",
            "dataset_id": identity,
            "status": how,
            "known_mules": manifest.get("known_mules"),
        }
    )
