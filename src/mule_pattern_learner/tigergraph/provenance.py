"""A prepared dataset's provenance: the graph's vertex counts, source id, queries and scope.

source_counts and resolve_source_id name the loaded snapshot before preparation;
verify_frozen_source rechecks it before a streamed run (see scope.py for the scope
checks).
"""

from __future__ import annotations

from typing import Any

from ..contract.fingerprints import fingerprint
from ..contract.server import GRAPH_NAME, SCOPE_VERTEX
from ..data.manifest import prepared_source
from .executor import ConnectionExecutor
from .installer import verify_sources
from .scope import scope_header, verify_scope

# Experiment metadata written by preparation itself; never part of source identity.
EXPERIMENT_METADATA_TYPES = frozenset({SCOPE_VERTEX})


def source_counts(executor: ConnectionExecutor) -> dict[str, int]:
    """The graph's vertex counts by type, excluding experiment metadata vertex types."""
    raw = executor.call(lambda conn: conn.getVertexCount("*", realtime=True), what="getVertexCount")
    if not isinstance(raw, dict):
        raise ValueError("TigerGraph did not return counts by vertex type")
    return {
        str(name): int(count)
        for name, count in raw.items()
        if str(name) not in EXPERIMENT_METADATA_TYPES
    }


def derived_source_id(counts: dict[str, int]) -> str:
    """A stable name for the loaded snapshot: the graph name plus a hash of its vertex counts."""
    return f"{GRAPH_NAME}_{fingerprint(counts)[:12]}"


def resolve_source_id(executor: ConnectionExecutor, scope_id: str, counts: dict[str, int]) -> str:
    """The source id: an existing scope's source, else a name derived from the graph."""
    attrs = scope_header(executor, scope_id)
    if attrs is not None:
        return str(attrs["source_id"])
    return derived_source_id(counts)


def verify_frozen_source(executor: ConnectionExecutor, manifest: dict[str, Any]) -> None:
    """Recheck the graph's provenance on every streamed run, including prepared-data reuse.

    Counts and headers catch drift, but cannot prove absence of same-count edits.
    The experiment still requires an operationally frozen source. Scope vertices
    are experiment metadata, so creating another scope does not invalidate data.
    """
    verify_sources(executor)
    source = prepared_source(manifest)
    if source_counts(executor) != source.counts:
        raise ValueError("Graph counts changed; freeze the source and prepare a new dataset")
    try:
        verify_scope(
            executor,
            source.scope_id,
            unowned=source.unowned,
            source_id=source.source_id,
            split_seed=source.split_seed,
            shares=source.shares,
        )
    except ValueError as error:
        raise ValueError(f"Prepared experiment scope is no longer valid: {error}") from None
