"""Preparation, nnPU training and pipeline reuse against fake TigerGraph endpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.device import choose_device
from mule_pattern_learner.temporal.live.batching import make_live_batch
from mule_pattern_learner.temporal.live.cli import build_parser
from mule_pattern_learner.temporal.live.contract import (
    ContextKey,
    FeaturePlan,
    SamplerPlan,
    extraction_plan,
)
from mule_pattern_learner.temporal.live.dataset import (
    load_prepared,
    prepare,
    preparation_view,
    query_hashes,
    sample_keys,
)
from mule_pattern_learner.temporal.live.evaluation import (
    ParquetEvaluationTruth,
    evaluate_predictions,
)
from mule_pattern_learner.temporal.live.hubs import load_hub_registry
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.config_schema import DEFAULT_RUN
from mule_pattern_learner.temporal.live.pipeline import DEFAULT_MODEL, run
from mule_pattern_learner.temporal.live.policy import context_scope
from mule_pattern_learner.temporal.live.sampling import pu_batches
from mule_pattern_learner.temporal.live.source import StreamingContextSource
from mule_pattern_learner.temporal.live.supervision import align_observed_labels, label_summary
from mule_pattern_learner.temporal.live.training import train
from mule_pattern_learner.temporal.loss import NonNegativePULoss
from temporal_fakes import (
    FakeExecutor,
    FrameObservedLabels,
    assigned_accounts,
    live_config,
    neighbourhood,
    scoped_accounts,
    supplied_labels,
)


def streaming_source(executor: FakeExecutor, config: dict[str, Any], **kwargs: Any):
    """The source a prepared run opens: prepared extraction plan and training sampler."""
    return StreamingContextSource(
        executor, plan=extraction_plan(config), sampler=SamplerPlan.from_config(config), **kwargs
    )


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_choose_device_prefers_available_accelerator(cuda: bool, mps: bool, expected: str) -> None:
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
    ):
        assert choose_device().type == expected
        assert choose_device("auto").type == expected
        assert choose_device("cpu").type == "cpu"
        if not mps:
            with pytest.raises(ValueError, match="MPS"):
                choose_device("mps")


def test_observed_labels_have_no_oracle_and_preserve_split_isolation() -> None:
    a = assigned_accounts()
    labels = align_observed_labels(a, supplied_labels())
    assert label_summary(labels) == {"train": 20, "validation": 20, "test": 20}
    assert labels.pu_label.sum() == 20
    assert not labels.loc[labels.split != "train", "pu_label"].any()
    bad = supplied_labels().assign(is_mule=1)
    with pytest.raises(ValueError, match="Oracle"):
        align_observed_labels(a, bad)
    with pytest.raises(ValueError, match="discovery"):
        align_observed_labels(a, supplied_labels().assign(known_from_ms=0))


def test_strict_claim_fails_before_preparation_or_training() -> None:
    with pytest.raises(ValueError, match="frozen TigerGraph scope_id"):
        context_scope({})
    assert context_scope({"scope_id": "scope"}) == "scope"


def test_nnpu_draws_only_known_positives_and_covers_the_label_blind_marginal() -> None:
    indices = np.arange(100)
    observed = indices < 20
    draws = list(pu_batches(indices, observed, np.random.default_rng(42), 16))
    assert all(observed[p].all() for p, _ in draws)
    assert sorted(np.concatenate([u for _, u in draws]).tolist()) == indices.tolist()
    assert all(len(p) == 4 for p, _ in draws)
    assert len(list(pu_batches(indices, observed, np.random.default_rng(42), 16, max_steps=2))) == 2


class PreparedExecutor(FakeExecutor):
    """The scope population, context, cutoff and hub queries.

    The scope population holds the fixture accounts with their splits as partitions.
    """

    def __init__(self, directory: Path, **kwargs: Any) -> None:
        super().__init__(population=scoped_accounts(), **kwargs)
        self.directory = directory

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        if name == "temporal_training_context":
            # A label mask must already be frozen when the first feature query starts.
            frozen = pd.read_parquet(self.directory / "observed_labels.parquet")
            assert label_summary(frozen) == {"train": 20, "validation": 20, "test": 20}
        return super().run(name, params, **kwargs)


def prepared(tmp_path: Path, config: dict[str, Any], **kwargs: Any) -> tuple[Path, FakeExecutor]:
    dataset = tmp_path / "dataset"
    executor = PreparedExecutor(dataset, **kwargs)
    prepare(
        config,
        dataset,
        executor,
        {"Account": 1000},
        labels=FrameObservedLabels(supplied_labels()),
    )
    return dataset, executor


def test_hidden_truth_cannot_change_updates_or_checkpoint_selection(tmp_path: Path) -> None:
    c = live_config()
    dataset, executor = prepared(tmp_path, c)
    assert executor.names().count("temporal_hub_registry") == 1
    first = train(c, dataset, tmp_path / "first.pt", contexts=streaming_source(executor, c))
    saved_first = torch.load(tmp_path / "first.pt", weights_only=True)
    # The oracle is a separate file that is never opened by training.
    a = pd.read_parquet(dataset / "accounts.parquet")
    assert "is_mule" not in a.columns
    truth = a[["account_id"]].assign(is_mule=np.arange(len(a)) % 2)
    truth_path = tmp_path / "evaluation_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    before = evaluate_predictions(
        tmp_path / "first_run/test_predictions.parquet",
        tmp_path / "first.pt",
        ParquetEvaluationTruth(truth_path),
    )
    truth["is_mule"] = 1 - truth.is_mule
    truth.to_parquet(truth_path, index=False)
    after = evaluate_predictions(
        tmp_path / "first_run/test_predictions.parquet",
        tmp_path / "first.pt",
        ParquetEvaluationTruth(truth_path),
    )
    assert before != after
    second = train(c, dataset, tmp_path / "second.pt", contexts=streaming_source(executor, c))
    saved_second = torch.load(tmp_path / "second.pt", weights_only=True)
    assert first["history"] == second["history"]
    assert first["best_epoch"] == second["best_epoch"]
    assert first["validation_proxy"] == second["validation_proxy"]
    assert saved_first["threshold"] == saved_second["threshold"]
    assert first["observed_label_proxy"] == second["observed_label_proxy"]
    for name, value in saved_first["state_dict"].items():
        torch.testing.assert_close(value, saved_second["state_dict"][name], rtol=0, atol=0)
    assert first["database_calls_during_training"] > 0
    assert first["known_mules"] == {"train": 20, "validation": 20, "test": 20}


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    # `mule-temporal train` needs no flag, file or identifier.
    args = build_parser().parse_args(["train"])
    assert args.config is None and args.dataset is None and args.output == DEFAULT_MODEL
    manifest = {"source": {"dataset_id": "graph_snapshot"}}
    with (
        patch(
            "mule_pattern_learner.temporal.live.pipeline.prepare_live", return_value=manifest
        ) as prep,
        patch(
            "mule_pattern_learner.temporal.live.pipeline.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert run(tmp_path / "model.pt")["status"] == "complete"
        prep.assert_called_once()
        # The prepared cache lives inside the run directory.
        assert prep.call_args.args[1] == tmp_path / "model_run" / "prepared"
        fit.assert_called_once()
        config, dataset, output = fit.call_args.args
        assert output == tmp_path / "model.pt" and dataset == tmp_path / "model_run" / "prepared"
        assert config["dataset_id"] == "graph_snapshot"
        assert config["device"] == "auto"
        assert config["scope_id"] == DEFAULT_RUN["scope_id"]


def test_ready_pipeline_reuses_cache_without_connecting(tmp_path: Path) -> None:
    from mule_pattern_learner.temporal.live.pipeline import prepare_live

    c = live_config()
    manifest = {
        "status": "ready",
        "source": {"query_hashes": query_hashes(), "preparation": preparation_view(c)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with patch("mule_pattern_learner.temporal.live.pipeline.live_executor") as client:
        assert prepare_live(c, tmp_path) == manifest
        # Model settings may change; preparation settings may not, and nothing connects.
        assert prepare_live({**c, "hidden": 32, "learning_rate": 0.01}, tmp_path) == manifest
        with pytest.raises(ValueError, match="seed_limits"):
            prepare_live({**c, "seed_limits": {**c["seed_limits"], "test": 10}}, tmp_path)
        client.assert_not_called()


def test_preparation_requests_no_context_and_training_keeps_a_bounded_lru(tmp_path: Path) -> None:
    c = live_config()
    dataset, executor = prepared(tmp_path, c)
    load_prepared(dataset)
    assert not executor.requested
    source = streaming_source(executor, c, capacity=4)
    result = train(c, dataset, tmp_path / "model.pt", contexts=source)
    assert result["database_calls_during_training"] > 0
    assert len(source.memory) <= 4


def test_streaming_source_serves_repeats_from_its_bounded_lru() -> None:
    keys = [ContextKey("Account", str(i), 100, 1000) for i in range(80)]
    memory = StreamingContextSource(FakeExecutor({}), capacity=3)
    rows = memory.fetch(keys)
    assert len(memory.memory) == 3
    calls = memory.query_calls
    assert memory.fetch(keys[-3:]) == rows[-3:]
    assert memory.query_calls == calls


def test_training_end_to_end_with_v5_neighbour_messages(tmp_path: Path) -> None:
    c = live_config()
    # N3 is a hub at every root cutoff; N5 always exceeds its history capacity.
    dataset, executor = prepared(
        tmp_path,
        c,
        factory=neighbourhood,
        hubs=[("N3", cutoff) for cutoff in (101, 102, 103)],
        statuses={"N5": "history_capacity_exceeded"},
    )
    manifest, accounts = load_prepared(dataset)
    plan, sampler = FeaturePlan.from_config(c), SamplerPlan.from_config(c)
    assert plan.architecture == "split"
    hubs = load_hub_registry(dataset, manifest)
    train_rows = accounts[accounts.split == "train"].iloc[:16]
    keys = sample_keys(train_rows, c["dates"]["train"][0], manifest)
    stats: dict[str, Any] = {}
    with streaming_source(executor, c) as source:
        batch = make_live_batch(
            source,
            keys,
            fanouts=tuple(c["fanouts"]),
            plan=plan,
            sampler=sampler,
            hubs=hubs,
            mode="train",
            step_seed=11,
            stats=stats,
        )
    assert batch["first_mask"].any() and batch["second_mask"].any()
    assert stats["stub_children"] > 0 and stats["rejected_children"] > 0
    assert stats["sampler_backend"] == "torch"
    torch.manual_seed(0)
    model = LiveTGAT(16, 4, 0, plan=plan)
    logits = model(batch)
    targets = torch.zeros_like(logits)
    targets[:4] = 1
    loss, _ = NonNegativePULoss(prior=c["class_prior"], positive_weight=c["class_prior"])(
        logits, targets
    )
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.edge, model.relation, model.rail):
        assert module.weight.grad is not None and torch.count_nonzero(module.weight.grad) > 0

    source = streaming_source(executor, c, capacity=64)
    result = train(c, dataset, tmp_path / "model.pt", contexts=source)
    assert result["status"] == "complete"
    assert all(math.isfinite(epoch["loss"]) for epoch in result["history"])
    assert result["sampler_backend"] == "torch"
    assert result["sampler_totals"]["stub_children"] > 0
    assert result["sampler_totals"]["rejected_children"] > 0
    assert result["rejections"]["history_capacity_exceeded"] > 0
    assert result["database_calls_during_training"] > 0
    # Roots and children were requested with their own pools; hubs were never fetched.
    assert set(executor.pools) == {tuple(sampler.query_params(hop).values()) for hop in (1, 2)}
    assert "N3" not in {k.node_id for k in executor.requested}
    progress = (tmp_path / "model_run/progress.jsonl").read_text().splitlines()
    assert json.loads(progress[-1])["event"] == "complete"


def test_rejected_roots_within_the_limit_are_dropped_and_counted(tmp_path: Path) -> None:
    # Rejected roots fail closed by default; this run tolerates up to 5% per split.
    c = live_config(max_rejected_root_fraction=0.05)
    # An unlabeled validation account, scored (and rejected) in every epoch. A rejected
    # observed positive would fail the run at any limit.
    validation = assigned_accounts().query("split == 'validation'").account_id
    rejected_root = str(validation.iloc[20])
    assert rejected_root not in set(supplied_labels().account_id)
    dataset, executor = prepared(
        tmp_path,
        c,
        factory=neighbourhood,
        statuses={"N5": "history_capacity_exceeded", rejected_root: "invisible_entity"},
    )
    result = train(c, dataset, tmp_path / "model.pt", contexts=streaming_source(executor, c))
    assert result["sampler_totals"]["rejected_children"] > 0
    assert result["rejections"]["invisible_entity"] >= 1
    assert result["rejected_roots"]["validation"]["rejected"] == 1
    assert result["rejected_roots"]["validation"]["positive"] == 0
    assert result["max_rejected_root_fraction"] == 0.05


def test_rejected_roots_fail_the_run_under_the_default_limit(tmp_path: Path) -> None:
    c = live_config()
    validation = assigned_accounts().query("split == 'validation'").account_id
    dataset, executor = prepared(
        tmp_path, c, factory=neighbourhood, statuses={str(validation.iloc[20]): "invisible_entity"}
    )
    with pytest.raises(ValueError, match="validation: TigerGraph rejected 1 of"):
        train(c, dataset, tmp_path / "model.pt", contexts=streaming_source(executor, c))
    assert not (tmp_path / "model.pt").exists()
