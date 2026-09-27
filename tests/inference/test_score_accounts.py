"""Scoring new accounts and prepared splits: outputs, precision and rejections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mule_pattern_learner.data.hub_registry import HubRegistry
from mule_pattern_learner.inference import score_accounts
from mule_pattern_learner.inference.rejections import rejection_summary
from mule_pattern_learner.testing.builders import (
    base_config,
    hub_registry,
    prepared_dataset,
    saved_model,
)
from mule_pattern_learner.testing.fake_graph import FakeSource, ScoringExecutor
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs


def test_score_new_writes_only_ok_rows_and_lists_rejected_ids(tmp_path: Path) -> None:
    config = base_config()
    model = saved_model(tmp_path / "model.pt", config)
    executor = ScoringExecutor()
    source = FakeSource(config, reject=frozenset({"ghost_1", "ghost_2"}))
    ids = [f"new_{i}" for i in range(14)]
    ids[1:1], ids[9:9] = ["ghost_1"], ["ghost_2"]
    rejected_ids = ["ghost_1", "ghost_2"]
    output = tmp_path / "scores.parquet"
    rejected_file = tmp_path / "scores_rejected.txt"
    result = score_accounts.score_new_accounts(
        model,
        iter(ids),
        "2025-01-01",
        output,
        rejected_output=rejected_file,
        cutoffs=TigerGraphCutoffs(executor),
        hub_reader=TigerGraphHubs(executor),
        contexts=source,
    )
    frame = pd.read_parquet(output)
    assert frame.account_id.tolist() == [v for v in ids if v not in rejected_ids]
    assert frame.score.between(0, 1).all() and (frame.date == "2025-01-01").all()
    assert rejected_file.read_text().split() == rejected_ids
    assert result["accounts"] == len(frame) and result["rejected"] == len(rejected_ids)
    assert result["rejected_roots_by_status"] == {"missing_entity": len(rejected_ids)}
    assert result["rejected_children"] == 0 and result["rejected_children_by_status"] == {}
    assert result["rejection_events_by_status"] == {"missing_entity": len(rejected_ids)}
    hub_call = next(p for name, p in executor.calls if name == "temporal_hub_registry")
    assert hub_call["cutoff_seqs"] == [30_000] and hub_call["threshold"] == 2048
    assert source.closed and not (tmp_path / "scores.parquet.pending").exists()


def test_score_new_keeps_float64_resolution_near_one(tmp_path: Path) -> None:
    config = base_config()
    # Logits near 20, where a float32 probability is exactly 1 for every account.
    model = saved_model(tmp_path / "model.pt", config, logit_shift=20.0)
    output = tmp_path / "scores.parquet"
    score_accounts.score_new_accounts(
        model,
        iter([f"new_{i}" for i in range(12)]),
        "2025-01-01",
        output,
        rejected_output=tmp_path / "scores_rejected.txt",
        cutoffs=TigerGraphCutoffs(ScoringExecutor()),
        hub_reader=TigerGraphHubs(ScoringExecutor()),
        contexts=FakeSource(config),
    )
    assert pq.read_schema(output).field("score").type == pa.float64()
    frame = pd.read_parquet(output)
    assert (frame.score < 1).all() and (frame.score.astype(np.float32) == 1).all()
    assert frame.score.nunique() == len(frame) == 12 and frame.predicted_mule.all()


def test_score_new_reports_root_and_child_rejections_separately(tmp_path: Path) -> None:
    config = base_config()
    model = saved_model(tmp_path / "model.pt", config)
    # P5 and P7 are peers (children) of the scored accounts, never roots.
    source = FakeSource(config, reject=frozenset({"ghost", "P5", "P7"}))
    ids = ["ghost", *(f"new_{i}" for i in range(12))]
    result = score_accounts.score_new_accounts(
        model,
        iter(ids),
        "2025-01-01",
        tmp_path / "scores.parquet",
        rejected_output=tmp_path / "scores_rejected.txt",
        cutoffs=TigerGraphCutoffs(ScoringExecutor()),
        hub_reader=TigerGraphHubs(ScoringExecutor()),
        contexts=source,
    )
    assert result["accounts"] == 12 and result["rejected"] == 1
    assert result["rejected_roots_by_status"] == {"missing_entity": 1}
    children = result["rejected_children_by_status"]["missing_entity"]
    assert result["rejected_children"] > 0 and children > 0
    assert result["rejection_events_by_status"] == {"missing_entity": 1 + children}
    # The root statuses are the source's hop-1 counts, never the mixed counter.
    plain = rejection_summary(source, 1, Counter({"rejected_children": 3}))
    assert plain["rejected_roots_by_status"] == {"missing_entity": 1}
    assert plain["rejected_children"] == 3


def test_inference_score_uses_the_dataset_hub_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    dataset, _, _ = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = saved_model(tmp_path / "model.pt", config, dataset)
    loaded = []

    def registry(path: Path, manifest: dict[str, Any]) -> HubRegistry:
        loaded.append(path)
        return hub_registry()

    monkeypatch.setattr(score_accounts, "load_hub_registry", registry)
    source = FakeSource(config, reject=frozenset({"A002"}))
    output, rejected = tmp_path / "test.parquet", tmp_path / "test_rejected.txt"
    result = score_accounts.score(
        model, dataset, "2025-01-01", "test", output, rejected_output=rejected, contexts=source
    )
    frame = pd.read_parquet(output)
    assert loaded == [dataset]
    assert len(frame) == 23 and "A002" not in set(frame.account_id)
    assert frame.score.dtype == np.float64
    assert result["rejected"] == 1 and rejected.read_text().split() == ["A002"]
