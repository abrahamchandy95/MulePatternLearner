"""Preparation against the live graph, with built-in settings.

`mule-temporal train` needs only the TigerGraph credentials in .env. On a fresh graph
the first run installs the training queries, creates the frozen scope and reveals the
known mules through the label contract; later runs find all three in place. Settings
are a config.RunConfig (DEFAULT_CONFIG for the command line), the source id is read
from the scope (or derived from the graph), and the prepared dataset is written inside
the run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..contract.fingerprints import fingerprint
from ..data.manifest import (
    MANIFEST,
    check_query_hashes,
    dataset_mismatches,
    read_manifest,
    write_manifest,
)
from ..data.preparation import prepare
from ..tigergraph.cutoffs import TigerGraphCutoffs
from ..tigergraph.executor import TigerGraphExecutor
from ..tigergraph.hubs import TigerGraphHubs
from ..tigergraph.installer import install
from ..tigergraph.labels import GraphObservedLabels
from ..tigergraph.provenance import source_counts
from ..tigergraph.reveal import ensure_revealed_labels
from ..tigergraph.scope import TigerGraphScope, ensure_scope, scope_header
from .connect import connect


def derived_source_id(graph: str, counts: dict[str, int]) -> str:
    """A stable name for the loaded snapshot: graph name plus a hash of its vertex counts."""
    return f"{graph}_{fingerprint(counts)[:12]}"


def resolve_source_id(executor: TigerGraphExecutor, scope_id: str, counts: dict[str, int]) -> str:
    """The source id: an existing scope's source, else a name derived from the graph."""
    attrs = scope_header(executor, scope_id)
    if attrs is not None:
        return str(attrs["source_id"])
    return derived_source_id(str(executor.client.graphname), counts)


def prepare_live(config: RunConfig, output: Path) -> dict[str, Any]:
    """Prepare (or resume) a dataset directory against the live graph.

    A ready directory is reused without connecting, but only after its GSQL hashes and
    dataset settings (data.manifest.dataset_settings) match the current ones.
    Otherwise the graph is brought to a trainable state first: stale queries are
    installed, the scope is created if missing, and known mules are revealed if the
    graph has none. A directory being prepared keeps its source id.
    """
    source_id: str | None = None
    if (output / MANIFEST).exists():
        manifest = read_manifest(output)
        check_query_hashes(manifest, output)
        changed = dataset_mismatches(config, manifest)
        if changed:
            raise ValueError(
                f"Dataset settings changed for {output} ({', '.join(changed)}); "
                "train into a new output, or restore the prepared values"
            )
        if manifest["status"] == "ready":
            # The trainer re-verifies artifacts before use. No database connection is needed.
            return manifest
        source_id = manifest["source"]["source_id"]
    executor = connect(config.transport)
    install(executor)
    counts = source_counts(executor)
    if source_id is None:
        source_id = resolve_source_id(executor, config.scope.id, counts)
    split_seed = config.dataset.split_seed
    ensure_scope(executor, config.scope, source_id=source_id, split_seed=split_seed)
    # The reveal draws its splits from the scope partitions.
    ensure_revealed_labels(executor, config.scope, config.dataset.dates)
    counts = source_counts(executor)
    result = prepare(
        config,
        source_id,
        output,
        counts,
        GraphObservedLabels(),
        scope=TigerGraphScope(executor),
        cutoffs=TigerGraphCutoffs(executor),
        hubs=TigerGraphHubs(executor),
    )
    if source_counts(executor) != counts:
        result["status"] = "source_changed"
        write_manifest(output, result)
        raise ValueError("Graph counts changed; freeze ingestion and prepare a fresh cache")
    return result
