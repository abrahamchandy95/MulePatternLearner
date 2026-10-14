"""The hub query: rows parsed into a registry, and contract violations refused."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.contract.graph_schema import HUB_COLUMNS, NO_SPLIT
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
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader


def test_hub_registry_parse_save_load_and_stub_semantics(tmp_path: Path) -> None:
    cutoffs = [1000, 2000]
    fake = FakeTigerGraph(answers={HUB_QUERY: lambda params: hub_rows(cutoffs, params["by_phase"])})
    registry = TigerGraphHubReader(fake).hub_registry([2000, 1000], threshold=1024)
    assert fake.calls == [
        (HUB_QUERY, {"cutoff_seqs": cutoffs, "threshold": 1024, "by_phase": False})
    ]
    assert registry.is_stub("Account", "H1", 1000) and not registry.is_stub("Account", "H1", 2000)
    assert registry.is_stub("Account", "H2", 2000, NO_SPLIT)
    assert not registry.is_stub("Token", "H1", 1000)
    with pytest.raises(ValueError, match="does not cover"):
        registry.is_stub("Account", "H1", 1500)
    with pytest.raises(ValueError, match="phases 0, not 1"):
        registry.is_stub("Account", "H1", 1000, 1)  # a batch in the split needs it by phase
    assert registry.counts() == {"1000": {"0": 1}, "2000": {"0": 1}}
    # A registry by phase of the split: held-out events can only add later-phase rows.
    by_phase = TigerGraphHubReader(fake).hub_registry(cutoffs, threshold=1024, by_phase=True)
    assert fake.calls[-1][1]["by_phase"] is True
    assert [by_phase.is_stub("Account", "H1", 1000, phase) for phase in (1, 2, 3)] == [
        False,
        True,
        True,
    ]
    assert by_phase.is_stub("Account", "H2", 2000, 1)
    assert not by_phase.is_stub("Account", "H2", 2000, 3)
    with pytest.raises(ValueError, match="phases 1, 2, 3, not 0"):
        by_phase.is_stub("Account", "H2", 2000)  # a batch without the split needs phase 0
    dataset = DatasetPaths(tmp_path)
    path = dataset.hubs
    by_phase.save(path)
    assert tuple(pd.read_parquet(path).columns) == tuple(HUB_COLUMNS)
    manifest = {
        "cutoff_seqs": {"2024-07-01": 1000, "2024-10-01": 2000},
        **hub_manifest(by_phase, path),
    }
    assert manifest["hub_by_phase"] is True and "hub_scope_id" not in manifest
    assert manifest["hub_counts"] == {
        "1000": {"1": 0, "2": 1, "3": 1},
        "2000": {"1": 1, "2": 0, "3": 0},
    }
    loaded = load_hub_registry(dataset, manifest)
    assert loaded.is_stub("Account", "H1", 1000, 2) and len(loaded) == 3
    # A dataset's contexts are in the split's phases, so a registry without them is refused.
    for unsplit in (False, None):
        with pytest.raises(ValueError, match="not by phase of the split"):
            load_hub_registry(dataset, {**manifest, "hub_by_phase": unsplit})
    HubRegistry(loaded.frame.iloc[:1], cutoff_seqs=cutoffs, threshold=1, by_phase=True).save(path)
    with pytest.raises(ValueError, match="changed"):
        load_hub_registry(dataset, manifest)
    empty = HubRegistry.empty()
    assert not empty.is_stub("Account", "H1", 123, 1) and len(empty) == 0
    # The threshold is the children pool's history bound.
    narrow = replace(SMALL_SAMPLER, children=replace(SMALL_SAMPLER.children, max_history=1024))
    assert hub_threshold(narrow) == 1024


@pytest.mark.parametrize(
    "by_phase, change",
    [
        (False, {"cutoff_seq": 3000}),
        # A row whose visible count is within the threshold is rejected.
        (False, {"max_visible": 10}),
        (False, {"max_visible": 1024}),
        # Rows without the split are phase 0, and rows by phase are 1, 2 or 3.
        (False, {"visibility_phase": 1}),
        (False, {"visibility_phase": 3}),
        (True, {"visibility_phase": 4}),
        (True, {"visibility_phase": 0}),
    ],
)
def test_hub_registry_rejects_contract_violations(by_phase: bool, change: dict[str, Any]) -> None:
    rows = hub_rows([1000, 2000], by_phase)
    rows[0]["hubs"][0].update(change)
    with pytest.raises(ValueError, match="contract"):
        TigerGraphHubReader(FakeTigerGraph(answers={HUB_QUERY: lambda params: rows})).hub_registry(
            [1000, 2000], threshold=1024, by_phase=by_phase
        )


def test_hub_registry_rejects_malformed_responses() -> None:
    def check(rows: list[dict[str, Any]], message: str, by_phase: bool = False) -> None:
        with pytest.raises(ValueError, match=message):
            TigerGraphHubReader(
                FakeTigerGraph(answers={HUB_QUERY: lambda params: rows})
            ).hub_registry([1000, 2000], threshold=1024, by_phase=by_phase)

    # Rows of a registry without phases are phase 0, which a registry by phase refuses.
    check(hub_rows([1000, 2000]), "contract", True)
    check(hub_rows([1000, 2000], True), "contract")
    rows = hub_rows([1000, 2000])
    del rows[0]["hubs"][0]["visibility_phase"]
    check(rows, "Malformed")
    check([{"status": "scope_not_ready"}], "rejected", True)
    rows = hub_rows([1000, 2000])
    rows[0]["hubs"].append(dict(rows[0]["hubs"][0]))
    check(rows, "Duplicate")
