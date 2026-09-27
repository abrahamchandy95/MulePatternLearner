from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch, child_key
from mule_pattern_learner.batching.limits import BatchCapacityError, BatchIndex
from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    DatasetConfig,
    SeedLimits,
    SplitDates,
)
from mule_pattern_learner.contract.feature_groups import (
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.time_basis import BASIS_ID
from mule_pattern_learner.data.accounts import select_accounts
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.inference.score_accounts import score_new_accounts
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.paths import DatasetPaths, RunPaths
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    FrameObservedLabels,
    assigned_accounts,
    context,
    example_config,
    message,
    supplied_labels,
)
from mule_pattern_learner.testing.fake_graph import FakeExecutor
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher, validate_context
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs
from mule_pattern_learner.tigergraph.oracle import TigerGraphTruth
from mule_pattern_learner.tigergraph.scope import TigerGraphScope
from mule_pattern_learner.training.schedule import pu_batches


def test_batch_ids_are_dense_scoped_and_temporal_and_never_global() -> None:
    a = ContextKey("Account", "999999999999999999999999", 100, 1000, "experiment", 1)
    other_type = replace(a, node_type="Token")
    other_time = replace(a, cutoff_seq=90)
    other_scope = replace(a, scope_id="another")
    index = BatchIndex([a, a, other_type, other_time, other_scope])
    assert [index[k] for k in (a, other_type, other_time, other_scope)] == [0, 1, 2, 3]
    assert BatchIndex([other_type])[other_type] == 0
    with pytest.raises(BatchCapacityError):
        BatchIndex([a, other_type], capacity=1)


def test_budget_rejects_before_database_calls_and_tensor_allocation() -> None:
    executor = FakeExecutor({})
    source = ContextSource(TigerGraphContextFetcher(executor))
    roots = [ContextKey("Account", str(i), 100, 1000) for i in range(129)]
    with pytest.raises(BatchCapacityError):
        build_batch(source, roots)
    with pytest.raises(BatchCapacityError):
        build_batch(source, roots[:64], fanouts=(64, 64))
    assert executor.requested == []


def test_scope_follows_recursive_events_and_cache_never_crosses_scope() -> None:
    a = ContextKey("Account", "a", 100, 1000, "strict", 1)
    msg = message(90, 900, a)
    source = FakeExecutor({a: context(a, [msg])})
    backend = ContextSource(TigerGraphContextFetcher(source))
    build_batch(backend, [a], fanouts=(2, 2))
    assert child_key(msg, a) in source.requested
    assert all(key.scope_id == "strict" and key.visibility_phase == 1 for key in source.requested)
    previous = backend.query_calls
    backend.fetch([replace(a, visibility_phase=3)])
    assert backend.query_calls > previous
    with pytest.raises(ValueError, match="differs"):
        validate_context(a, context(replace(a, visibility_phase=3)))


def test_stream_retention_is_bounded_across_many_disjoint_batches() -> None:
    backend = ContextSource(
        TigerGraphContextFetcher(FakeExecutor({})), capacity=8, request_batch_size=16
    )
    for start in range(0, 512, 16):
        backend.fetch([ContextKey("Account", str(i), 100, 1000) for i in range(start, start + 16)])
        assert len(backend.memory) <= 8
    assert backend.query_calls == 32
    backend.close()
    assert not backend.memory


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
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract_fingerprint(),
            "basis_id": BASIS_ID,
            "threshold": 0.5,
            "config": config.to_dict(),
            "input_fingerprint": plan.fingerprint(),
        },
        checkpoint,
    )

    class Executor(FakeExecutor):
        def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
            if name == "temporal_training_context":
                assert params["scope_id"] == "" and params["per_relation"] == 1
            return super().run(name, params, **kwargs)

    executor = Executor({}, last_visible=lambda index, ms: 99)
    output = tmp_path / "new.parquet"
    result = score_new_accounts(
        checkpoint,
        (f"never_trained_{i}" for i in range(13)),
        "2025-01-01",
        output,
        cutoffs=TigerGraphCutoffs(executor),
        hub_reader=TigerGraphHubs(executor),
        fetcher=TigerGraphContextFetcher(executor),
    )
    frame = pd.read_parquet(output)
    assert result["accounts"] == len(frame) == 13
    assert all(frame.score.between(0, 1))
    assert frame.account_id.tolist() == [f"never_trained_{i}" for i in range(13)]
    # The embedding joins attention, the slot sum and the summary branch of the pool counts.
    assert all(len(v) == 48 for v in frame.embedding)
    assert not (tmp_path / "new.parquet.pending").exists()
    assert not {"is_mule", "known_positive", "pu_label"} & set(frame.columns)
    # The hub registry was computed for the requested cutoff only (one past the last event).
    hubs = [params for name, params in executor.calls if name == "temporal_hub_registry"]
    assert [params["cutoff_seqs"] for params in hubs] == [[100]]
    assert result["rejected"] == 0 and result["rejected_output"] is None


def test_bounded_seed_reservoir_does_not_enrich_the_nnpu_marginal() -> None:
    # Server has already assigned ownership groups; the client never holds all owners.
    rows = [
        {
            "account_id": f"A{i:05}",
            "partition": i % 3 + 1,
            "group_id": str(i),
            "first_seen_ts_ms": 1,
            "observed_positive": False,
            "known_from_ms": 0,
        }
        for i in range(10050)
    ]
    calls = []

    class Executor:
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            assert name == "temporal_scope_population" and not params["include_observed"]
            calls.append(params["after_id"])
            page = [r for r in rows if r["account_id"] > params["after_id"]][:10000]
            return [{"status": "ok", "accounts": page}]

    dataset = DatasetConfig(
        dates=SplitDates(("2024-01-01",), ("2024-01-02",), ("2024-01-03",)),
        seed_limits=SeedLimits(10, 10, 10),
        seed=42,
    )
    known = pd.DataFrame(
        {"account_id": ["A00000", "A00001", "A00002"], "known_positive": True, "known_from_ms": 1}
    )
    selected, counts = select_accounts(
        TigerGraphScope(Executor()), "strict", dataset, FrameObservedLabels(known)
    )
    assert len(calls) == 2 and sum(counts.values()) == len(rows)
    assert len(selected) <= 33 and selected.in_marginal.sum() == 30
    assert set(known.account_id) <= set(selected.account_id)
    observed = selected.account_id.isin(known.account_id).to_numpy()
    train = selected.split.eq("train").to_numpy()
    marginal = np.flatnonzero(train & selected.in_marginal.to_numpy())
    positive = np.flatnonzero(train & observed)
    draws = list(
        pu_batches(marginal, observed, np.random.default_rng(42), 8, positive_indices=positive)
    )
    assert sorted(np.concatenate([u for _, u in draws])) == sorted(marginal)
    assert all(observed[p].all() for p, _ in draws)


def test_the_graph_truth_pages_the_label_contract() -> None:
    rows = [
        {"account_id": f"A{i:05}", "is_mule": i % 2, "mule_label_known": i % 3 != 0}
        for i in range(10050)
    ]
    calls: list[dict[str, Any]] = []

    class Executor:
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            assert name == "temporal_get_account_supervision"
            calls.append(params)
            page = [{"attributes": r} for r in rows if r["account_id"] > params["after_id"]]
            return [{"status": "ok"}, {"accounts": page[: params["batch_size"]]}]

    truth = TigerGraphTruth(Executor()).read()
    assert [(c["after_id"], c["batch_size"]) for c in calls] == [("", 10000), ("A09999", 10000)]
    assert truth.account_id.tolist() == [r["account_id"] for r in rows]
    # An account whose label is not known is -1, never a negative.
    assert truth.is_mule.tolist() == [r["is_mule"] if r["mule_label_known"] else -1 for r in rows]

    class Unordered:
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            page = [{"account_id": a, "is_mule": 0, "mule_label_known": True} for a in "BA"]
            return [{"status": "ok", "accounts": page}]

    with pytest.raises(ValueError, match="not strictly increasing"):
        TigerGraphTruth(Unordered()).read()

    class Silent:
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            return [{"status": "ok"}]

    with pytest.raises(ValueError, match="accounts missing from response"):
        TigerGraphTruth(Silent()).read()


def test_strict_preparation_and_nnpu_use_the_correct_phase_end_to_end(tmp_path: Path) -> None:
    from mule_pattern_learner.data.preparation import prepare
    from mule_pattern_learner.training.trainer import train

    cfg = example_config(
        scope={"id": "unit_strict"},
        dataset={"seed_limits": {"train": 10, "validation": 10, "test": 10}},
    )
    rows = assigned_accounts().drop(columns="owner_ids").copy()
    rows["partition"] = rows["split"].map({"train": 1, "validation": 2, "test": 3})
    rows = rows.drop(columns="split")
    run = RunPaths(tmp_path / "run")
    phases = []

    class Executor(FakeExecutor):
        def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
            if name == "temporal_scope_population":
                assert params["include_observed"] is False
                return [{"status": "ok", "accounts": rows.to_dict("records")}]
            if name == "temporal_training_context":
                assert params["scope_id"] == "unit_strict"
                phases.append(params["visibility_phase"])
                if params["visibility_phase"] == 3:
                    assert run.model.exists(), "Test evaluation happened before the model froze"
            return super().run(name, params, **kwargs)

    executor = Executor({}, last_visible=lambda index, ms: 100)
    dataset = DatasetPaths(tmp_path / "dataset")
    manifest = prepare(
        cfg,
        UNIT_SOURCE,
        dataset,
        {"Account": len(rows)},
        FrameObservedLabels(supplied_labels()),
        scope=TigerGraphScope(executor),
        cutoffs=TigerGraphCutoffs(executor),
        hubs=TigerGraphHubs(executor),
    )
    assert manifest["status"] == "ready" and not executor.requested
    source = ContextSource(
        TigerGraphContextFetcher(executor),
        plan=extraction_plan(cfg.feature_plan()),
        sampler=cfg.sampler,
    )
    result = train(cfg, dataset, run, contexts=source)
    assert set(phases) == {1, 2, 3}
    assert result["known_mules"] == {"train": 20, "validation": 20, "test": 20}
    assert result["evaluation_protocol"] == "strict_inductive"
