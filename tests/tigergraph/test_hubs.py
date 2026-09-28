"""The hub query: rows parsed into a registry, and contract violations refused."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.contract.graph_schema import HUB_COLUMNS
from mule_pattern_learner.contract.server import HUB_QUERY
from mule_pattern_learner.data.hub_registry import (
    HubRegistry,
    hub_manifest,
    hub_threshold,
    load_hub_registry,
)
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.testing.builders import SMALL_SAMPLER, hub_rows
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs


def test_hub_registry_parse_save_load_and_stub_semantics(tmp_path: Path) -> None:
    cutoffs = [1000, 2000]
    fake = FakeTigerGraph(answers={HUB_QUERY: lambda params: hub_rows(cutoffs, params["scope_id"])})
    registry = TigerGraphHubs(fake).hub_registry([2000, 1000], threshold=1024)
    assert fake.calls == [(HUB_QUERY, {"cutoff_seqs": cutoffs, "threshold": 1024, "scope_id": ""})]
    assert registry.is_stub("Account", "H1", 1000) and not registry.is_stub("Account", "H1", 2000)
    assert registry.is_stub("Account", "H2", 2000, 3)
    assert not registry.is_stub("Token", "H1", 1000)
    with pytest.raises(ValueError, match="does not cover"):
        registry.is_stub("Account", "H1", 1500)
    with pytest.raises(ValueError, match="phases 3, not 1"):
        registry.is_stub("Account", "H1", 1000, 1)  # a scoped batch needs a scoped registry
    assert registry.counts() == {"1000": {"3": 1}, "2000": {"3": 1}}
    # A scoped registry is keyed by phase: held-out events can only add later-phase rows.
    scoped = TigerGraphHubs(fake).hub_registry(cutoffs, threshold=1024, scope_id="scope")
    assert fake.calls[-1][1]["scope_id"] == "scope"
    assert [scoped.is_stub("Account", "H1", 1000, phase) for phase in (1, 2, 3)] == [
        False,
        True,
        True,
    ]
    assert scoped.is_stub("Account", "H2", 2000, 1) and not scoped.is_stub("Account", "H2", 2000)
    dataset = DatasetPaths(tmp_path)
    path = dataset.hubs
    scoped.save(path)
    assert tuple(pd.read_parquet(path).columns) == tuple(HUB_COLUMNS)
    manifest = {
        "cutoff_seqs": {"2024-07-01": 1000, "2024-10-01": 2000},
        "source": {"scope_id": "scope"},
        **hub_manifest(scoped, path),
    }
    assert manifest["hub_scope_id"] == "scope"
    assert manifest["hub_counts"] == {
        "1000": {"1": 0, "2": 1, "3": 1},
        "2000": {"1": 1, "2": 0, "3": 0},
    }
    loaded = load_hub_registry(dataset, manifest)
    assert loaded.is_stub("Account", "H1", 1000, 2) and len(loaded) == 3
    with pytest.raises(ValueError, match="computed for scope 'scope'"):
        load_hub_registry(dataset, {**manifest, "source": {"scope_id": "other"}})
    HubRegistry(loaded.frame.iloc[:1], cutoff_seqs=cutoffs, threshold=1, scope_id="scope").save(
        path
    )
    with pytest.raises(ValueError, match="changed"):
        load_hub_registry(dataset, manifest)
    empty = HubRegistry.empty()
    assert not empty.is_stub("Account", "H1", 123, 1) and len(empty) == 0
    # The threshold is the children pool's history bound.
    narrow = replace(SMALL_SAMPLER, children=replace(SMALL_SAMPLER.children, max_history=1024))
    assert hub_threshold(narrow) == 1024


@pytest.mark.parametrize(
    "scope_id, change",
    [
        ("", {"reason": "scan_cost"}),
        ("", {"cutoff_seq": 3000}),
        # All-time degree never makes a hub: a row whose visible count is within the
        # threshold is rejected however large the account grows after the cutoff.
        ("", {"max_visible": 10, "max_degree": 300_000}),
        ("", {"max_visible": 1024}),
        ("", {"visibility_phase": 1}),  # unscoped rows are phase 3
        ("scope", {"visibility_phase": 4}),
        ("scope", {"visibility_phase": 0}),
        ("", {"max_degree": -1}),
    ],
)
def test_hub_registry_rejects_contract_violations(scope_id: str, change: dict[str, Any]) -> None:
    rows = hub_rows([1000, 2000], scope_id)
    rows[0]["hubs"][0].update(change)
    with pytest.raises(ValueError, match="contract"):
        TigerGraphHubs(FakeTigerGraph(answers={HUB_QUERY: lambda params: rows})).hub_registry(
            [1000, 2000], threshold=1024, scope_id=scope_id
        )


def test_hub_registry_rejects_stale_or_mismatched_responses() -> None:
    def check(rows: list[dict[str, Any]], message: str, scope_id: str = "") -> None:
        with pytest.raises(ValueError, match=message):
            TigerGraphHubs(FakeTigerGraph(answers={HUB_QUERY: lambda params: rows})).hub_registry(
                [1000, 2000], threshold=1024, scope_id=scope_id
            )

    rows = hub_rows([1000, 2000])
    rows[0]["threshold"] = 2048
    check(rows, "echoed threshold")
    check(hub_rows([1000, 2000], "other"), "echoed scope_id", "scope")
    rows = hub_rows([1000, 2000])
    del rows[0]["hubs"][0]["visibility_phase"]
    check(rows, "Malformed")
    check([{"status": "scope_not_ready"}], "rejected", "scope")
    rows = hub_rows([1000, 2000])
    rows[0]["hubs"].append(dict(rows[0]["hubs"][0]))
    check(rows, "Duplicate")
