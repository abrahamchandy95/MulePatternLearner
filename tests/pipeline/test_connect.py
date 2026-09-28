"""The pipeline's connection: retry budgets, the transport section and the frozen-source check."""

# Tests inspect transport internals (the encoding cadence) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, TransportConfig
from mule_pattern_learner.contract.server import SCOPE_POLICY_QUERY
from mule_pattern_learner.data.context_cache import ContextCache
from mule_pattern_learner.data.manifest import dataset_id, dataset_settings, source_fingerprint
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.pipeline import connect
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph

# open_context_source reads the dataset from its manifest; nothing fetches, so the
# directory of its context cache is not read.
UNUSED = DatasetPaths(Path("unused"))


def test_the_transport_section_sets_the_source_and_the_retry_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}
    settings = SimpleNamespace()
    monkeypatch.setattr(connect, "verify_frozen_source", lambda executor, manifest: None)
    # The connection settings would come from .env; the test never reads it.
    monkeypatch.setattr(connect, "ConnectionSettings", lambda: settings)

    class Executor:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(connect, "TigerGraphExecutor", Executor)
    manifest = {
        "source": {
            "source_counts": {"Account": 10},
            "settings": dataset_settings("d", DEFAULT_CONFIG),
        }
    }
    transport = {
        "request_batch_size": 32,
        "query_concurrency": 4,
        "context_lru_capacity": 1024,
        "encoding_check_every": 8,
        "max_query_attempts": 3,
        "max_outage_s": 120,
    }
    training = DEFAULT_CONFIG.with_changes({"transport": transport})
    store = connect.open_context_source(UNUSED, manifest, training)
    assert seen == {"settings": settings, "max_attempts": 3, "max_outage_s": 120}
    assert (store.request_batch_size, store.concurrency, store.capacity) == (32, 4, 1024)
    assert store.encoding_check_every == 8
    # The source reads and writes the dataset's context cache, named by the dataset and
    # the frozen source its manifest records.
    assert store.disk is not None
    assert store.disk.cache == ContextCache(
        UNUSED.contexts, dataset_id("d", DEFAULT_CONFIG), source_fingerprint(manifest)
    )
    store.close()
    # mule check's source has none.
    with connect.open_context_source(UNUSED, manifest, training, cached=False) as store:
        assert store.disk is None
    changed = training.with_changes({"sampler": {"roots": {"recent": 5}}})
    seen.clear()
    with pytest.raises(ValueError, match="pools differ"):
        connect.open_context_source(UNUSED, manifest, changed)
    # Mismatched pools are refused before connecting.
    assert seen == {}


def test_a_resumed_stream_checks_the_frozen_source_before_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = {"ready": True, "source_id": "snapshot", "split_seed": 42}
    graph = FakeTigerGraph(counts={"Account": 10}, scopes={"scope": header})
    budgets: list[tuple[int, int]] = []

    def connected(transport: TransportConfig) -> Any:
        budgets.append((transport.max_query_attempts, transport.max_outage_s))
        return graph

    monkeypatch.setattr(connect, "connect", connected)
    config = DEFAULT_CONFIG.with_changes(
        {
            "scope": {"id": "scope"},
            "dataset": {"split_seed": 42},
            "transport": {"max_query_attempts": 3, "max_outage_s": 60},
        }
    )
    manifest = {
        "source": {
            "source_counts": {"Account": 10},
            "settings": dataset_settings("snapshot", config),
        },
    }
    backend = connect.open_context_source(UNUSED, manifest, config)
    backend.close()
    assert budgets == [(3, 60)]
    assert graph.calls == [(SCOPE_POLICY_QUERY, {"scope_id": "scope"})]
    # Queries whose installed text differs from the repository's are refused.
    graph.stale = frozenset({SCOPE_POLICY_QUERY})
    with pytest.raises(ValueError, match="mule install"):
        connect.open_context_source(UNUSED, manifest, config)
    graph.stale = frozenset()
    # A scope created with another scope.unowned rule than the configured one is refused.
    graph.scope_policy = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope.unowned = 'independent'"):
        connect.open_context_source(UNUSED, manifest, config)
    graph.scope_policy = "linked"
    graph.counts["Account"] += 1
    with pytest.raises(ValueError, match="counts changed"):
        connect.open_context_source(UNUSED, manifest, config)
    graph.counts["Account"] -= 1
    # A session's sources share its one connection, and each checks the frozen source.
    session = connect.Session(config.with_changes({"transport": {"max_outage_s": 90}}).transport)
    budgets.clear()
    graph.calls.clear()
    for _ in range(2):
        connect.open_context_source(UNUSED, manifest, config, session=session).close()
    assert budgets == [(3, 90)]
    assert graph.names() == [SCOPE_POLICY_QUERY] * 2
    graph.scopes["scope"]["ready"] = False
    with pytest.raises(ValueError, match="no longer valid"):
        connect.open_context_source(UNUSED, manifest, config)
