"""Scoring new accounts, without a dataset or labels: outputs, precision and rejections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import contract_fingerprint
from mule_pattern_learner.contract.server import CONTEXT_QUERY, HUB_QUERY
from mule_pattern_learner.contract.time_basis import BASIS_ID
from mule_pattern_learner.inference import score_accounts
from mule_pattern_learner.inference.rejections import rejection_summary
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.pipeline.connect import context_source
from mule_pattern_learner.runtime import console
from mule_pattern_learner.testing.builders import HUB, RUNTIME_CHANGES, saved_model, unit_config
from mule_pattern_learner.testing.fake_graph import FakeSource, FakeTigerGraph
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader


def scoring_graph() -> FakeTigerGraph:
    """The graph at the test cutoff: its last event is 29,999, and HUB is a hub then."""
    return FakeTigerGraph(last_visible=lambda index, ms: 29_999, hubs=[(HUB, 30_000)])


def test_score_new_writes_only_ok_rows_and_lists_rejected_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = unit_config(RUNTIME_CHANGES)
    model = saved_model(tmp_path / "model.pt", config)
    executor = scoring_graph()
    source = FakeSource(config, reject=frozenset({"ghost_1", "ghost_2"}))
    ids = [f"new_{i}" for i in range(14)]
    ids[1:1], ids[9:9] = ["ghost_1"], ["ghost_2"]
    rejected_ids = ["ghost_1", "ghost_2"]
    output = tmp_path / "scores.parquet"
    rejected_file = tmp_path / "scores_rejected.txt"
    # On a terminal scoring shows in place how many of the ids it has scored.
    monkeypatch.setattr(console, "is_terminal", lambda: True)
    result = score_accounts.score_new_accounts(
        model,
        iter(ids),
        "2025-01-01",
        output,
        rejected_output=rejected_file,
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
        contexts=source,
        total=len(ids),
    )
    console.end_progress()
    shown = capsys.readouterr().out
    assert shown.rstrip().rsplit("\r", 1)[-1].rstrip() == f"scoring accounts 16/{len(ids)}"
    frame = pd.read_parquet(output)
    assert frame.account_id.tolist() == [v for v in ids if v not in rejected_ids]
    assert frame.score.between(0, 1).all() and (frame.date == "2025-01-01").all()
    assert rejected_file.read_text().split() == rejected_ids
    assert result["accounts"] == len(frame) and result["rejected"] == len(rejected_ids)
    assert result["rejected_roots_by_status"] == {"missing_entity": len(rejected_ids)}
    assert result["rejected_children"] == 0 and result["rejected_children_by_status"] == {}
    assert result["rejection_events_by_status"] == {"missing_entity": len(rejected_ids)}
    hub_call = next(p for name, p in executor.calls if name == HUB_QUERY)
    assert hub_call["cutoff_seqs"] == [30_000] and hub_call["threshold"] == 2048
    # Scoring arbitrary accounts is unscoped: phase-3 rows over all visible history.
    assert hub_call["scope_id"] == ""
    assert source.closed and not (tmp_path / "scores.parquet.pending").exists()


def test_score_new_keeps_float64_resolution_near_one(tmp_path: Path) -> None:
    config = unit_config(RUNTIME_CHANGES)
    # Logits near 20, where a float32 probability is exactly 1 for every account.
    model = saved_model(tmp_path / "model.pt", config, logit_shift=20.0)
    output = tmp_path / "scores.parquet"
    score_accounts.score_new_accounts(
        model,
        iter([f"new_{i}" for i in range(12)]),
        "2025-01-01",
        output,
        rejected_output=tmp_path / "scores_rejected.txt",
        cutoffs=TigerGraphCutoffReader(scoring_graph()),
        hub_reader=TigerGraphHubReader(scoring_graph()),
        contexts=FakeSource(config),
    )
    assert pq.read_schema(output).field("score").type == pa.float64()
    frame = pd.read_parquet(output)
    assert (frame.score < 1).all() and (frame.score.astype(np.float32) == 1).all()
    assert frame.score.nunique() == len(frame) == 12 and frame.predicted_mule.all()


def test_score_new_reports_root_and_child_rejections_separately(tmp_path: Path) -> None:
    config = unit_config(RUNTIME_CHANGES)
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
        cutoffs=TigerGraphCutoffReader(scoring_graph()),
        hub_reader=TigerGraphHubReader(scoring_graph()),
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


def test_new_account_scoring_needs_neither_training_dataset_nor_labels(tmp_path: Path) -> None:
    pool = {"recent": 1, "older": 0, "distinct": 0}
    config = DEFAULT_CONFIG.with_changes(
        {
            "model": {"hidden": 16, "heads": 4, "dropout": 0.0},
            "training": {"batch_size": 4},
            "sampler": {
                "fanouts": [2, 2],
                "roots": pool | {"associations": 2},
                "children": pool | {"associations": 0},
            },
        }
    )
    plan = config.feature_plan()
    model = build_model(config.model, plan, config.sampler.fanouts[0])
    model_file = tmp_path / "model.pt"
    torch.save(
        {
            "format": SavedModel.FORMAT,
            "state_dict": model.state_dict(),
            "contract": contract_fingerprint(),
            "basis_id": BASIS_ID,
            "threshold": 0.5,
            "config": config.to_dict(),
            "input_fingerprint": plan.fingerprint(),
        },
        model_file,
    )

    executor = FakeTigerGraph(last_visible=lambda index, ms: 99)
    output = tmp_path / "new.parquet"
    result = score_accounts.score_new_accounts(
        model_file,
        (f"never_trained_{i}" for i in range(13)),
        "2025-01-01",
        output,
        rejected_output=tmp_path / "new_rejected.txt",
        contexts=context_source(executor, config),
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
    )
    frame = pd.read_parquet(output)
    assert result["accounts"] == len(frame) == 13
    assert all(frame.score.between(0, 1))
    assert frame.account_id.tolist() == [f"never_trained_{i}" for i in range(13)]
    # The embedding joins attention, the slot sum and the summary branch of the pool counts.
    assert all(len(v) == 48 for v in frame.embedding)
    assert not (tmp_path / "new.parquet.pending").exists()
    assert not {"is_mule", "known_positive", "pu_label"} & set(frame.columns)
    # Unscoped context requests with the model's pools.
    requests = [params for name, params in executor.calls if name == CONTEXT_QUERY]
    assert requests and all(p["scope_id"] == "" and p["per_relation"] == 1 for p in requests)
    # The hub registry was computed for the requested cutoff only (one past the last event).
    hubs = [params for name, params in executor.calls if name == HUB_QUERY]
    assert [params["cutoff_seqs"] for params in hubs] == [[100]]
    assert result["rejected"] == 0 and result["rejected_output"] is None
