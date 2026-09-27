"""The ground-truth audit of a run's model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.artifacts import read_audit_scores, read_json
from mule_pattern_learner.batching import assemble
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.evaluation.audit import audit
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.testing.builders import (
    CUTOFFS,
    base_config,
    hub_registry,
    prepared_dataset,
    saved_model,
)
from mule_pattern_learner.testing.fake_graph import FakeSource, FakeTigerGraph
from mule_pattern_learner.tigergraph.scope import TigerGraphScope


def split_graph(accounts: pd.DataFrame) -> FakeTigerGraph:
    """A graph whose scope population is these accounts, all in the test partition."""
    rows = [{"account_id": a, "partition": 3, "first_seen_ts_ms": 1} for a in accounts.account_id]
    return FakeTigerGraph(population=rows)


def test_the_audit_scores_through_the_dataset_clock_and_hubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(runtime={"max_rejected_root_fraction": 0.1})
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config, dataset)
    test_accounts = accounts[accounts.split == "test"]
    truth = test_accounts[["account_id"]].assign(is_mule=(np.arange(len(test_accounts)) % 4 == 0))
    truth["is_mule"] = truth.is_mule.astype(int)
    source = FakeSource(config, reject=frozenset({test_accounts.account_id.iloc[1]}))
    seen: list[int] = []
    real = assemble.build_batch

    def record(store: Any, roots: list[ContextKey], **kwargs: Any) -> dict[str, torch.Tensor]:
        seen.extend(k.cutoff_seq for k in roots)
        return real(store, roots, **kwargs)

    monkeypatch.setattr(assemble, "build_batch", record)

    class Truth:
        def read(self) -> pd.DataFrame:
            return truth

    result = audit(
        run,
        Truth(),
        dataset=dataset,
        scope=TigerGraphScope(split_graph(test_accounts)),
        contexts=source,
        hubs=hub_registry(),
    )
    assert set(seen) == {CUTOFFS["2025-01-01"]}
    assert result["rejected_accounts"] == 1 and result["rejected_negatives"] == 1
    assert result["rejected"] == 1 and result["rejected_roots_by_status"] == {"missing_entity": 1}
    assert result["metrics"]["sample_accounts"] == len(test_accounts) - 1
    assert result["metrics"]["evaluation_sample"].endswith("_minus_rejected_negatives")
    for pct in (1, 5, 10):
        assert 0 <= result["metrics"][f"recall_at_{pct}pct"] <= 1
        assert 0 <= result["metrics"][f"precision_at_{pct}pct"] <= 1
    assert read_json(run.audit_report("test")) == result
    scores = read_audit_scores(run.audit_scores("test"))
    assert scores.score.dtype == np.float64 and len(scores) == len(test_accounts) - 1
    assert run.audit_rejected("test").read_text().split() == [test_accounts.account_id.iloc[1]]
    # An audit the run already has is never overwritten.
    with pytest.raises(FileExistsError, match="test.json"):
        audit(
            run,
            Truth(),
            dataset=dataset,
            scope=TigerGraphScope(split_graph(test_accounts)),
            contexts=source,
            hubs=hub_registry(),
        )


@pytest.mark.parametrize(("limit", "rejected_index"), [(1.0, 0), (0.0, 1)])
def test_the_audit_fails_on_censored_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: float, rejected_index: int
) -> None:
    config = base_config(runtime={"max_rejected_root_fraction": limit})
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config, dataset)
    test_accounts = accounts[accounts.split == "test"]
    truth = test_accounts[["account_id"]].assign(
        is_mule=(np.arange(len(test_accounts)) % 4 == 0).astype(int)
    )

    class Truth:
        def read(self) -> pd.DataFrame:
            return truth

    # Index 0 is a test positive (always fatal); index 1 a negative (fatal at limit 0).
    source = FakeSource(config, reject=frozenset({test_accounts.account_id.iloc[rejected_index]}))
    match = "1 test positives" if rejected_index == 0 else "0 test positives"
    with pytest.raises(ValueError, match=match):
        audit(
            run,
            Truth(),
            dataset=dataset,
            scope=TigerGraphScope(split_graph(test_accounts)),
            contexts=source,
            hubs=hub_registry(),
        )
    assert not (run.root / "audit").exists()
