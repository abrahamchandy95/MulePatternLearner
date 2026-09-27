"""The live provenance of a prepared dataset: vertex counts, queries and its scope.

verify_frozen_source rechecks it before a streamed run (see scope.py for the scope
checks).
"""

from __future__ import annotations

from typing import Any

from ..contract.server import SCOPE_VERTEX
from ..data.manifest import recorded_settings
from .executor import ConnectionExecutor
from .installer import verify_sources
from .scope import verify_scope

# Experiment metadata written by preparation itself; never part of source identity.
EXPERIMENT_METADATA_TYPES = frozenset({SCOPE_VERTEX})


def source_counts(executor: ConnectionExecutor) -> dict[str, int]:
    """Live vertex counts by type, excluding experiment metadata vertex types."""
    raw = executor.call(lambda conn: conn.getVertexCount("*", realtime=True), what="getVertexCount")
    if not isinstance(raw, dict):
        raise ValueError("TigerGraph did not return counts by vertex type")
    return {
        str(name): int(count)
        for name, count in raw.items()
        if str(name) not in EXPERIMENT_METADATA_TYPES
    }


def verify_frozen_source(executor: ConnectionExecutor, manifest: dict[str, Any]) -> None:
    """Recheck live provenance on every streamed run, including prepared-data reuse.

    Counts and headers catch drift, but cannot prove absence of same-count edits.
    The experiment still requires an operationally frozen source. Scope vertices
    are experiment metadata, so creating another scope does not invalidate data.
    """
    verify_sources(executor)
    recorded = {
        name: count
        for name, count in manifest["source"]["source_counts"].items()
        if name not in EXPERIMENT_METADATA_TYPES
    }
    if source_counts(executor) != recorded:
        raise ValueError("Live graph counts changed; freeze the source and prepare a new dataset")
    settings = recorded_settings(manifest)
    try:
        verify_scope(
            executor,
            settings["scope"]["id"],
            unowned=settings["scope"]["unowned"],
            source_id=settings["source_id"],
            split_seed=settings["dataset"]["split_seed"],
        )
    except ValueError as error:
        raise ValueError(f"Prepared experiment scope is no longer valid: {error}") from None
