"""The pipeline's connection: retry budgets, transport settings and the frozen-source check."""

# Tests inspect transport internals (the encoding cadence) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mule_pattern_learner.config import transport_settings
from mule_pattern_learner.pipeline import connect
from mule_pattern_learner.testing.fake_graph import scope_counts
from mule_pattern_learner.tigergraph import provenance


def test_transport_settings_come_from_the_training_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    settings = SimpleNamespace()
    monkeypatch.setattr(connect, "verify_frozen_source", lambda executor, manifest: None)
    # The connection settings would come from .env; the test never reads it.
    monkeypatch.setattr(connect, "Settings", lambda: settings)

    class Executor:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(connect, "TigerGraphExecutor", Executor)
    prepared = {"dataset_id": "d", "scope_id": "scope"}
    manifest = {"config": prepared, "source": {}}
    training = {
        **prepared,
        "request_batch_size": 32,
        "query_concurrency": 4,
        "context_lru_capacity": 1024,
        "encoding_check_every": 8,
        "max_query_attempts": 3,
        "max_outage_s": 120,
    }
    store = connect.open_context_source(Path("unused"), manifest, training)
    assert seen == {"settings": settings, "max_attempts": 3, "max_outage_s": 120}
    assert (store.request_batch_size, store.concurrency, store.capacity) == (32, 4, 1024)
    assert store._cadence.every == 8
    store.close()
    changed = {**training, "sampler": {"recent": 5}}
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

    def connected(config: dict[str, Any]) -> Any:
        transport = transport_settings(config)
        budgets.append((transport["max_query_attempts"], transport["max_outage_s"]))
        return executor

    monkeypatch.setattr(connect, "connect", connected)
    manifest = {
        "config": {"dataset_id": "snapshot", "scope_id": "scope", "split_seed": 42},
        "source": {"source_counts": dict(counts)},
    }
    backend = connect.open_context_source(
        Path("unused"), manifest, {"max_query_attempts": 3, "max_outage_s": 60}
    )
    backend.close()
    assert checked == [executor] and budgets == [(3, 60)]
    assert policy_calls == [{"scope_id": "scope"}]
    # A scope created with another scope_unowned rule than the configured one is refused.
    policy["scope_unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope_unowned = 'independent'"):
        connect.open_context_source(Path("unused"), manifest)
    policy["scope_unowned"] = "linked"
    counts["Account"] += 1
    with pytest.raises(ValueError, match="counts changed"):
        connect.open_context_source(Path("unused"), manifest)
    counts["Account"] -= 1
    header["ready"] = False
    with pytest.raises(ValueError, match="no longer valid"):
        connect.open_context_source(Path("unused"), manifest)
