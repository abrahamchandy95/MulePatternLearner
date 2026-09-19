from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.temporal.encoding import BASIS_ID
from mule_pattern_learner.temporal.live.batching import make_live_batch, child_key
from mule_pattern_learner.temporal.live.contract import ContextKey, contract_fingerprint
from mule_pattern_learner.temporal.live.memory import BatchIndex, BatchCapacityError
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.source import StreamingContextSource, validate_context
from mule_pattern_learner.temporal.live.predictor import score_new_accounts
from mule_pattern_learner.temporal.live.cohort import scoped_cohort
from mule_pattern_learner.temporal.live.sampling import pu_batches
from mule_pattern_learner.temporal.live.supervision import FrameObservedLabels
from test_live_pipeline import FakeExecutor, context, message


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
    source = StreamingContextSource(executor)
    roots = [ContextKey("Account", str(i), 100, 1000) for i in range(129)]
    with pytest.raises(BatchCapacityError):
        make_live_batch(source, roots)
    with pytest.raises(BatchCapacityError):
        make_live_batch(source, roots[:64], fanouts=(64, 64))
    assert executor.requested == []


def test_scope_follows_recursive_events_and_cache_never_crosses_scope() -> None:
    a = ContextKey("Account", "a", 100, 1000, "strict", 1)
    msg = message(90, 900, a)
    source = FakeExecutor({a: context(a, [msg])})
    backend = StreamingContextSource(source)
    make_live_batch(backend, [a], fanouts=(2, 2))
    assert child_key(msg, a) in source.requested
    assert all(key.scope_id == "strict" and key.visibility_phase == 1 for key in source.requested)
    previous = backend.query_calls
    backend.fetch([replace(a, visibility_phase=3)])
    assert backend.query_calls > previous
    with pytest.raises(ValueError, match="differs"):
        validate_context(a, context(replace(a, visibility_phase=3)))


def test_stream_retention_is_bounded_across_many_disjoint_batches() -> None:
    backend = StreamingContextSource(FakeExecutor({}), capacity=8)
    for start in range(0, 512, 16):
        backend.fetch([ContextKey("Account", str(i), 100, 1000) for i in range(start, start + 16)])
        assert len(backend.memory) <= 8
    assert backend.query_calls == 32
    backend.close()
    assert not backend.memory


def test_new_account_scoring_needs_neither_training_dataset_nor_labels(tmp_path: Path) -> None:
    model = LiveTGAT(16, 4, 0)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract_fingerprint(),
            "basis_id": BASIS_ID,
            "threshold": 0.5,
            "config": {
                "hidden": 16,
                "heads": 4,
                "dropout": 0.0,
                "batch_size": 4,
                "fanouts": [2, 2],
                "per_relation": 1,
            },
        },
        checkpoint,
    )

    class Executor(FakeExecutor):
        def run(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            if name == "temporal_training_cutoffs":
                return [
                    {"status": "ok", "last_visible_seqs": {str(params["cutoff_times"][0]): 100}}
                ]
            assert params["scope_id"] == "" and params["per_relation"] == 1
            return super().run(name, params)

    executor = Executor({})
    output = tmp_path / "new.parquet"
    result = score_new_accounts(
        checkpoint,
        (f"never_trained_{i}" for i in range(13)),
        "2025-01-01",
        output,
        executor=executor,
    )
    frame = pd.read_parquet(output)
    assert result["accounts"] == len(frame) == 13
    assert all(frame.score.between(0, 1))
    assert frame.account_id.tolist() == [f"never_trained_{i}" for i in range(13)]
    assert all(len(v) == 16 for v in frame.embedding)
    assert not (tmp_path / "new.parquet.pending").exists()
    assert not {"is_mule", "known_positive", "pu_label"} & set(frame.columns)


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
        def run(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            assert name == "temporal_scope_population" and not params["include_observed"]
            calls.append(params["after_id"])
            page = [r for r in rows if r["account_id"] > params["after_id"]][:10000]
            return [{"status": "ok", "accounts": page}]

    cfg = {
        "scope_id": "strict",
        "seed": 42,
        "seed_limits": {"train": 10, "validation": 10, "test": 10},
        "dates": {s: ["2024-01-01"] for s in ("train", "validation", "test")},
    }
    known = pd.DataFrame(
        {"account_id": ["A00000", "A00001", "A00002"], "known_positive": True, "known_from_ms": 1}
    )
    selected, counts = scoped_cohort(Executor(), cfg, FrameObservedLabels(known))
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


def test_strict_preparation_and_nnpu_use_the_correct_phase_end_to_end(tmp_path: Path) -> None:
    from mule_pattern_learner.temporal.live.dataset import prepare
    from mule_pattern_learner.temporal.live.training import train
    from test_live_training import config, assigned_accounts, supplied_labels

    cfg = {
        **config(),
        "evaluation_protocol": "strict_inductive",
        "scope_id": "unit_strict",
        "context_storage": "stream",
        "seed_limits": {"train": 10, "validation": 10, "test": 10},
    }
    rows = assigned_accounts().drop(columns="owner_ids").copy()
    rows["partition"] = rows["split"].map({"train": 1, "validation": 2, "test": 3})
    rows = rows.drop(columns="split")
    checkpoint = tmp_path / "model.pt"
    phases = []

    class Executor(FakeExecutor):
        def run(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            if name == "temporal_scope_population":
                return [{"status": "ok", "accounts": rows.to_dict("records")}]
            if name == "temporal_training_cutoffs":
                return [
                    {
                        "status": "ok",
                        "last_visible_seqs": {str(ms): 100 for ms in params["cutoff_times"]},
                    }
                ]
            assert params["scope_id"] == "unit_strict"
            phases.append(params["visibility_phase"])
            if params["visibility_phase"] == 3:
                assert checkpoint.exists(), "Test evaluation happened before checkpoint was frozen"
            return super().run(name, params)

    executor = Executor({})
    dataset = tmp_path / "dataset"
    manifest = prepare(
        cfg, dataset, executor, {"Account": len(rows)}, FrameObservedLabels(supplied_labels())
    )
    assert manifest["cached_contexts"] == 0
    source = StreamingContextSource(executor)
    result = train(cfg, dataset, checkpoint, contexts=source)
    assert set(phases) == {1, 2, 3}
    assert result["known_mules"] == {"train": 20, "validation": 20, "test": 20}
    assert result["evaluation_protocol"] == "strict_inductive"


def test_resumed_stream_checks_live_source_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from mule_pattern_learner.temporal.live import installation, source

    counts = {"Account": 10}
    header = {"ready": True, "source_id": "snapshot", "split_seed": 42}
    conn = SimpleNamespace(
        getVertexCount=lambda *args, **kwargs: dict(counts),
        getVerticesById=lambda *args: [{"attributes": dict(header)}],
    )
    executor = SimpleNamespace(client=SimpleNamespace(conn=conn))
    checked = []
    monkeypatch.setattr(installation, "verify_sources", lambda client: checked.append(client))
    monkeypatch.setattr(source, "TigerGraphExecutor", lambda: executor)
    manifest = {
        "config": {
            "dataset_id": "snapshot",
            "scope_id": "scope",
            "evaluation_protocol": "strict_inductive",
        },
        "source": {"context_storage": "stream", "source_counts": dict(counts)},
    }
    backend = source.open_context_source(Path("unused"), manifest)
    backend.close()
    assert checked == [executor]
    counts["Account"] += 1
    with pytest.raises(ValueError, match="counts changed"):
        source.open_context_source(Path("unused"), manifest)
    counts["Account"] -= 1
    header["ready"] = False
    with pytest.raises(ValueError, match="no longer valid"):
        source.open_context_source(Path("unused"), manifest)
