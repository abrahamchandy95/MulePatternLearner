"""The composition root's connection: only the pipeline builds TigerGraph adapters.

connect reads the connection settings from the repository .env when it is called and
builds the executor with a transport section's retry budgets. open_context_source
opens the context source of a prepared dataset on the frozen graph; the pipeline hands
it to the use cases that open one (a data.contexts.ContextOpener).
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig, TransportConfig
from ..contract.feature_groups import extraction_plan
from ..contract.sampler_plan import sampler_pools
from ..data.contexts import StreamingContextSource, streaming_source
from ..data.manifest import recorded_settings
from ..paths import DatasetPaths
from ..tigergraph.connection import Settings
from ..tigergraph.context_query import TigerGraphContextFetcher
from ..tigergraph.executor import TigerGraphExecutor
from ..tigergraph.provenance import verify_frozen_source


def connect(transport: TransportConfig) -> TigerGraphExecutor:
    """A connected executor with the retry budgets of a transport section."""
    return TigerGraphExecutor(
        settings=Settings(),
        max_attempts=transport.max_query_attempts,
        max_outage_s=transport.max_outage_s,
    )


def open_context_source(
    dataset: DatasetPaths, manifest: dict[str, Any], config: RunConfig
) -> StreamingContextSource:
    """Open the live source of a prepared dataset for a training or scoring configuration.

    It requests the model's groups and hop-2 flags (extraction_plan) with the prepared
    candidate pools, on a connection with the configuration's transport section whose
    source is checked to be the frozen one.
    """
    if sampler_pools(config.sampler) != recorded_settings(manifest)["sampler_pools"]:
        raise ValueError(
            f"Sampler candidate pools differ from the dataset in {dataset.root}; prepare a new "
            "dataset or restore the prepared sampler pools"
        )
    executor = connect(config.transport)
    verify_frozen_source(executor, manifest)
    return streaming_source(
        TigerGraphContextFetcher(executor),
        extraction_plan(config.feature_plan()),
        config.sampler,
        config.transport,
    )
