"""The composition root's connection: only the pipeline builds TigerGraph adapters.

connect reads the connection settings from the repository .env when it is called and
builds the executor with a transport section's retry budgets. context_source is the
one place a context source is built, for a configuration on a connection:
open_context_source opens that of a prepared dataset on the frozen graph, with the
dataset's disk tier, and the pipeline hands it to the use cases that open one (a
data.contexts.ContextOpener).
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig, TransportConfig
from ..contract.feature_groups import extraction_plan
from ..contract.sampler_plan import sampler_pools
from ..data.context_cache import ContextCache
from ..data.contexts import ContextSource, build_context_source
from ..data.manifest import recorded_settings
from ..paths import DatasetPaths
from ..tigergraph.connection import ConnectionSettings
from ..tigergraph.context_query import TigerGraphContextFetcher
from ..tigergraph.executor import QueryExecutor, TigerGraphExecutor
from ..tigergraph.provenance import verify_frozen_source


def connect(transport: TransportConfig) -> TigerGraphExecutor:
    """A connected executor with the retry budgets of a transport section."""
    return TigerGraphExecutor(
        settings=ConnectionSettings(),
        max_attempts=transport.max_query_attempts,
        max_outage_s=transport.max_outage_s,
    )


def context_source(
    executor: QueryExecutor, config: RunConfig, cache: ContextCache | None = None
) -> ContextSource:
    """The source of a configuration's contexts on a connection.

    It requests the model's groups and hop-2 flags (extraction_plan) with the
    configuration's candidate pools, LRU, request size and concurrency. ``cache`` gives
    it the disk tier of a prepared dataset, whose entries are the rows of the dataset's
    frozen source: pass one only on a connection verify_frozen_source has checked.
    """
    return build_context_source(
        TigerGraphContextFetcher(executor),
        extraction_plan(config.feature_plan()),
        config.sampler,
        config.transport,
        cache,
    )


def open_context_source(
    dataset: DatasetPaths, manifest: dict[str, Any], config: RunConfig
) -> ContextSource:
    """Open the live source of a prepared dataset for a training or scoring configuration.

    It is the configuration's context_source, whose candidate pools must be the
    prepared ones, on a connection with the configuration's transport section whose
    source is checked to be the frozen one; only then does it get the dataset's disk
    tier (data.context_cache.ContextCache.of).
    """
    if sampler_pools(config.sampler) != recorded_settings(manifest)["sampler_pools"]:
        raise ValueError(
            f"Sampler candidate pools differ from the dataset in {dataset.root}; prepare a new "
            "dataset or restore the prepared sampler pools"
        )
    executor = connect(config.transport)
    verify_frozen_source(executor, manifest)
    return context_source(executor, config, ContextCache.of(dataset, manifest))
