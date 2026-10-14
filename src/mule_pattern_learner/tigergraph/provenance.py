"""A prepared dataset's provenance: the graph's vertex counts, queries, split and labels.

source_counts names the loaded snapshot before preparation (data.manifest.dataset_id);
verify_frozen_source rechecks it before a streamed run (see scope.py for the split
checks, and reveal.py for the label check).
"""

from __future__ import annotations

from typing import Any

from ..contract.server import SCOPE_VERTEX
from ..data.manifest import prepared_reveal, prepared_source, recorded_dataset_id
from .executor import ConnectionExecutor
from .installer import verify_sources
from .reveal import verify_labels
from .scope import verify_scope

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


def verify_frozen_source(executor: ConnectionExecutor, manifest: dict[str, Any]) -> None:
    """Recheck the graph's provenance on every streamed run, including prepared-data reuse.

    Counts and headers catch drift, but cannot prove absence of same-count edits.
    The experiment still requires an operationally frozen source. The split must be the
    one the dataset was prepared in, and the labels those of its reveal
    (reveal.verify_labels). A refusal says to move the dataset aside before `mule train`
    is to create a split: it reuses a ready dataset without creating one.
    """
    verify_sources(executor)
    source = prepared_source(manifest)
    if source_counts(executor) != source.counts:
        raise ValueError("Graph counts changed; freeze the source and prepare a new dataset")
    try:
        verify_scope(executor, split_seed=source.split_seed, shares=source.shares)
    except ValueError as error:
        raise ValueError(
            f"The dataset's split is no longer valid: {error}. `mule train` creates a split "
            "only when it prepares a dataset, so to have it create one, first move this "
            f"dataset (data/{recorded_dataset_id(manifest)}/) aside"
        ) from None
    verify_labels(executor, *prepared_reveal(manifest))
