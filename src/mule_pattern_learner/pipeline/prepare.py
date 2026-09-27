"""Preparation against the live graph, with built-in settings.

`mule-temporal train` needs only the TigerGraph credentials in .env. On a fresh graph
the first run installs the training queries, creates the frozen scope and reveals the
known mules through the label contract; later runs find all three in place. Settings
default to config.DEFAULT_RUN, the dataset identity is read from the scope (or
derived from the graph), and the prepared cache is written inside the run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contract.fingerprints import fingerprint
from ..contract.graph_schema import context_scope
from ..data.manifest import (
    MANIFEST,
    check_query_hashes,
    preparation_mismatches,
    read_manifest,
    write_manifest,
)
from ..data.preparation import prepare
from ..tigergraph.executor import TigerGraphExecutor
from ..tigergraph.installer import install
from ..tigergraph.labels import GraphObservedLabels
from ..tigergraph.provenance import source_counts
from ..tigergraph.reveal import ensure_revealed_labels
from ..tigergraph.scope import ensure_scope, scope_header
from .connect import connect


def derived_dataset_id(graph: str, counts: dict[str, int]) -> str:
    """A stable name for the loaded snapshot: graph name plus a hash of its vertex counts."""
    return f"{graph}_{fingerprint(counts)[:12]}"


def resolve_identity(
    executor: TigerGraphExecutor, config: dict[str, Any], counts: dict[str, int]
) -> dict[str, Any]:
    """Fill dataset_id: an existing scope's source, else a name derived from the graph."""
    if config.get("dataset_id"):
        return config
    attrs = scope_header(executor, config["scope_id"])
    if attrs is not None:
        return {**config, "dataset_id": str(attrs["source_id"])}
    graph = str(getattr(executor.client.conn, "graphname", "graph"))
    return {**config, "dataset_id": derived_dataset_id(graph, counts)}


def prepare_live(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Prepare (or resume) a dataset directory against the live graph.

    A ready directory is reused without connecting, but only after its GSQL
    hashes and preparation settings (PREPARATION_KEYS) match the current ones.
    Otherwise the graph is brought to a trainable state first: stale queries are
    installed, the scope is created if missing, and known mules are revealed if the
    graph has none.
    """
    context_scope(config)
    if (output / MANIFEST).exists():
        manifest = read_manifest(output)
        if not config.get("dataset_id"):
            config = {**config, "dataset_id": manifest["source"]["dataset_id"]}
        check_query_hashes(manifest, output)
        changed = preparation_mismatches(config, manifest)
        if changed:
            raise ValueError(
                f"Preparation settings changed for {output} ({', '.join(changed)}); "
                "train into a new output, or restore the prepared values"
            )
        if manifest["status"] == "ready":
            # The trainer re-verifies artifacts before use. No database connection is needed.
            return manifest
    executor = connect(config)
    install(executor)
    counts = source_counts(executor)
    config = resolve_identity(executor, config, counts)
    ensure_scope(executor, config)
    # The reveal draws its splits from the scope partitions.
    ensure_revealed_labels(executor, config)
    counts = source_counts(executor)
    result = prepare(config, output, executor, counts, GraphObservedLabels())
    if source_counts(executor) != counts:
        result["status"] = "source_changed"
        write_manifest(output, result)
        raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh cache")
    return result
