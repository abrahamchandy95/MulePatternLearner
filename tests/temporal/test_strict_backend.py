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
from mule_pattern_learner.temporal.live.evaluation import GraphEvaluationTruth
from mule_pattern_learner.temporal.live.memory import BatchIndex, BatchCapacityError
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.contract import SamplerPlan, extraction_plan
from mule_pattern_learner.temporal.live.context_query import validate_context
from mule_pattern_learner.temporal.live.source import StreamingContextSource
from mule_pattern_learner.temporal.live.predictor import score_new_accounts
from mule_pattern_learner.temporal.live.cohort import scoped_cohort
from mule_pattern_learner.temporal.live.sampling import pu_batches
from mule_pattern_learner.temporal.live.supervision import FrameObservedLabels
from temporal_fakes import (
    FakeExecutor,
    assigned_accounts,
    context,
    live_config,
    message,
    scope_counts,
    supplied_labels,
)


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
    backend = StreamingContextSource(FakeExecutor({}), capacity=8, request_batch_size=16)
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
        executor=executor,
    )
    frame = pd.read_parquet(output)
    assert result["accounts"] == len(frame) == 13
    assert all(frame.score.between(0, 1))
    assert frame.account_id.tolist() == [f"never_trained_{i}" for i in range(13)]
    assert all(len(v) == 16 for v in frame.embedding)
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


def test_graph_evaluation_truth_pages_the_label_contract() -> None:
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

    truth = GraphEvaluationTruth(Executor()).read()
    assert [(c["after_id"], c["batch_size"]) for c in calls] == [("", 10000), ("A09999", 10000)]
    assert truth.account_id.tolist() == [r["account_id"] for r in rows]
    # An account whose label is not known is -1, never a negative.
    assert truth.is_mule.tolist() == [r["is_mule"] if r["mule_label_known"] else -1 for r in rows]

    class Unordered:
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            page = [{"account_id": a, "is_mule": 0, "mule_label_known": True} for a in "BA"]
            return [{"status": "ok", "accounts": page}]

    with pytest.raises(ValueError, match="not strictly increasing"):
        GraphEvaluationTruth(Unordered()).read()

    class Silent:
        def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
            return [{"status": "ok"}]

    with pytest.raises(ValueError, match="accounts missing from response"):
        GraphEvaluationTruth(Silent()).read()


@pytest.mark.parametrize("profile", ["legacy", "v5"])
def test_strict_preparation_and_nnpu_use_the_correct_phase_end_to_end(
    tmp_path: Path, profile: str
) -> None:
    from mule_pattern_learner.temporal.live.dataset import prepare
    from mule_pattern_learner.temporal.live.training import train

    cfg = live_config(
        profile,
        evaluation_protocol="strict_inductive",
        scope_id="unit_strict",
        context_storage="stream",
        seed_limits={"train": 10, "validation": 10, "test": 10},
    )
    rows = assigned_accounts().drop(columns="owner_ids").copy()
    rows["partition"] = rows["split"].map({"train": 1, "validation": 2, "test": 3})
    rows = rows.drop(columns="split")
    checkpoint = tmp_path / "model.pt"
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
                    assert checkpoint.exists(), "Test evaluation happened before checkpoint froze"
            return super().run(name, params, **kwargs)

    executor = Executor({}, last_visible=lambda index, ms: 100)
    dataset = tmp_path / "dataset"
    manifest = prepare(
        cfg, dataset, executor, {"Account": len(rows)}, FrameObservedLabels(supplied_labels())
    )
    assert manifest["cached_contexts"] == 0
    source = StreamingContextSource(
        executor, plan=extraction_plan(cfg), sampler=SamplerPlan.from_config(cfg)
    )
    result = train(cfg, dataset, checkpoint, contexts=source)
    assert set(phases) == {1, 2, 3}
    assert result["known_mules"] == {"train": 20, "validation": 20, "test": 20}
    assert result["evaluation_protocol"] == "strict_inductive"


def test_resumed_stream_checks_live_source_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from mule_pattern_learner.temporal.live import installation, source
    from mule_pattern_learner.temporal.live.executor import transport_settings

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

    executor = SimpleNamespace(client=SimpleNamespace(conn=conn), run=run)
    checked = []
    monkeypatch.setattr(installation, "verify_sources", lambda client: checked.append(client))
    budgets: list[tuple[int, int]] = []

    def live_executor(config: dict[str, Any]) -> Any:
        transport = transport_settings(config)
        budgets.append((transport["max_query_attempts"], transport["max_outage_s"]))
        return executor

    monkeypatch.setattr(source, "live_executor", live_executor)
    manifest = {
        "config": {
            "dataset_id": "snapshot",
            "scope_id": "scope",
            "evaluation_protocol": "strict_inductive",
        },
        "source": {"context_storage": "stream", "source_counts": dict(counts)},
    }
    backend = source.open_context_source(
        Path("unused"), manifest, {"max_query_attempts": 3, "max_outage_s": 60}
    )
    backend.close()
    assert checked == [executor] and budgets == [(3, 60)]
    assert policy_calls == [{"scope_id": "scope"}]
    # A scope created with another scope_unowned rule than the configured one is refused.
    policy["scope_unowned"] = "independent"
    with pytest.raises(ValueError, match="no longer valid.*scope_unowned = 'independent'"):
        source.open_context_source(Path("unused"), manifest)
    policy["scope_unowned"] = "linked"
    counts["Account"] += 1
    with pytest.raises(ValueError, match="counts changed"):
        source.open_context_source(Path("unused"), manifest)
    counts["Account"] -= 1
    header["ready"] = False
    with pytest.raises(ValueError, match="no longer valid"):
        source.open_context_source(Path("unused"), manifest)
