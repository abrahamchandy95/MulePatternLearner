"""The composition root's connection: only the pipeline builds TigerGraph adapters.

connect reads the connection settings from the repository .env when it is called and
builds the executor with a transport section's retry budgets. A use case connects on
its own, unless it is given a Session: the one connection a suite of runs shares, opened
when the first use case needs it. context_source is the one place a context source is
built, for a configuration on a connection: open_context_source opens that of a
prepared dataset on the frozen graph, with the dataset's disk tier unless `mule check`
asks for none, and the pipeline hands it to the use cases that open one (a
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
from ..tigergraph.executor import ConnectionExecutor, QueryExecutor, TigerGraphExecutor
from ..tigergraph.provenance import verify_frozen_source


def connect(transport: TransportConfig) -> TigerGraphExecutor:
    """A connected executor with the retry budgets of a transport section."""
    return TigerGraphExecutor(
        settings=ConnectionSettings(),
        max_attempts=transport.max_query_attempts,
        max_outage_s=transport.max_outage_s,
    )


class Session:
    """One TigerGraph connection that the use cases of a suite share, opened on first use.

    executor() connects with the transport section's retry budgets when it is first
    called and returns that executor after, so a suite of runs that prepares, trains and
    audits through one session connects at most once, and not at all when none of its
    use cases needs the graph. Each use case still checks the frozen source on it.
    ``announced`` holds the ids of the datasets a `dataset` event has named on the
    session, so a suite names its dataset once rather than once a run.
    """

    def __init__(self, transport: TransportConfig) -> None:
        self.transport = transport
        self._executor: ConnectionExecutor | None = None
        self.announced: set[str] = set()

    @property
    def connected(self) -> bool:
        return self._executor is not None

    def executor(self) -> ConnectionExecutor:
        if self._executor is None:
            self._executor = connect(self.transport)
        return self._executor


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
    dataset: DatasetPaths,
    manifest: dict[str, Any],
    config: RunConfig,
    *,
    cached: bool = True,
    session: Session | None = None,
) -> ContextSource:
    """Open a prepared dataset's context source for a training or scoring configuration.

    It is the configuration's context_source, whose candidate pools must be the
    prepared ones, on a connection with the configuration's transport section (the
    session's, with a session) whose source is checked to be the frozen one; only then
    does it get the dataset's disk tier (data.context_cache.ContextCache.of).
    ``cached=False`` leaves the disk tier out, so every context memory lacks is
    requested from TigerGraph (`mule check`).
    """
    if sampler_pools(config.sampler) != recorded_settings(manifest)["sampler_pools"]:
        raise ValueError(
            f"Sampler candidate pools differ from the dataset in {dataset.root}; prepare a new "
            "dataset or restore the prepared sampler pools"
        )
    executor = session.executor() if session is not None else connect(config.transport)
    verify_frozen_source(executor, manifest)
    cache = ContextCache.of(dataset, manifest) if cached else None
    return context_source(executor, config, cache)
