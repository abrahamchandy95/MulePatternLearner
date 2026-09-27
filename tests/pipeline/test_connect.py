"""The pipeline's connection: retry budgets, the transport section and the frozen-source check."""

# Tests inspect transport internals (the encoding cadence) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, TransportConfig
from mule_pattern_learner.data.manifest import dataset_settings
from mule_pattern_learner.pipeline import connect
from mule_pattern_learner.testing.fake_graph import scope_counts
from mule_pattern_learner.tigergraph import provenance


def test_the_transport_section_sets_the_source_and_the_retry_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}
    settings = SimpleNamespace()
    monkeypatch.setattr(connect, "verify_frozen_source", lambda executor, manifest: None)
    # The connection settings would come from .env; the test never reads it.
    monkeypatch.setattr(connect, "Settings", lambda: settings)

    class Executor:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(connect, "TigerGraphExecutor", Executor)
    manifest = {"source": {"settings": dataset_settings("d", DEFAULT_CONFIG)}}
    transport = {
        "request_batch_size": 32,
        "query_concurrency": 4,
        "context_lru_capacity": 1024,
        "encoding_check_every": 8,
        "max_query_attempts": 3,
        "max_outage_s": 120,
    }
    training = DEFAULT_CONFIG.with_changes({"transport": transport})
    store = connect.open_context_source(Path("unused"), manifest, training)
    assert seen == {"settings": settings, "max_attempts": 3, "max_outage_s": 120}
    assert (store.request_batch_size, store.concurrency, store.capacity) == (32, 4, 1024)
    assert store._cadence.every == 8
    store.close()
    changed = training.with_changes({"sampler": {"roots": {"recent": 5}}})
    seen.clear()
    with pytest.raises(ValueError, match="pools differ"):
        connect.open_context_source(Path("unused"), manifest, changed)
    # Mismatched pools are refused before connecting.
    assert seen == {}


def test_resumed_stream_checks_live_source_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"Account": 10}
    header = {"ready": True, "source_id": "snapshot", "split_seed": 42}
    policy = {"scope_unowned": "linked"}
    conn = SimpleNamespace(
        getVertexCount=lambda *args, **kwargs: dict(counts),
        getVerticesById=lambda *args: [{"attributes": dict(header)}],
    )
    policy_calls: list[dict[str, Any]] = []

    def run(name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        assert name == "temporal_scope_policy"
        policy_calls.append(params)
        return [{"status": "ok", **scope_counts(policy["scope_unowned"])}]

    executor = SimpleNamespace(
        client=SimpleNamespace(conn=conn), run=run, call=lambda operation, what: operation(conn)
    )
    checked = []
    monkeypatch.setattr(provenance, "verify_sources", lambda client: checked.append(client))
    budgets: list[tuple[int, int]] = []

    def connected(transport: TransportConfig) -> Any:
        budgets.append((transport.max_query_attempts, transport.max_outage_s))
        return executor

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
            "source_counts": dict(counts),
            "settings": dataset_settings("snapshot", config),
        },
    }
    backend = connect.open_context_source(Path("unused"), manifest, config)
    backend.close()
    assert checked == [executor] and budgets == [(3, 60)]
    assert policy_calls == [{"scope_id": "scope"}]
    # A scope created with another scope.unowned rule than the configured one is refused.
    policy["scope_unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope.unowned = 'independent'"):
        connect.open_context_source(Path("unused"), manifest, config)
    policy["scope_unowned"] = "linked"
    counts["Account"] += 1
    with pytest.raises(ValueError, match="counts changed"):
        connect.open_context_source(Path("unused"), manifest, config)
    counts["Account"] -= 1
    header["ready"] = False
    with pytest.raises(ValueError, match="no longer valid"):
        connect.open_context_source(Path("unused"), manifest, config)
