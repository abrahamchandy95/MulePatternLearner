"""Training: resumes, determinism, rejections, selection and the saved model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
import math
from pathlib import Path
import threading
from typing import Any, NoReturn
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mule_pattern_learner.batching import assemble
from mule_pattern_learner.batching.assemble import make_live_batch
from mule_pattern_learner.config import RunConfig
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS, extraction_plan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.data.contexts import StreamingContextSource
from mule_pattern_learner.data.hub_registry import HubRegistry, load_hub_registry, warn_hub_stubs
from mule_pattern_learner.data.manifest import load_prepared
from mule_pattern_learner.data.observed_labels import label_summary
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.data.splits import sample_keys
from mule_pattern_learner.evaluation.audit import evaluate_predictions
from mule_pattern_learner.evaluation.truth import ParquetEvaluationTruth
from mule_pattern_learner.model.loss import NonNegativePULoss
from mule_pattern_learner.model.tgat import LiveTGAT
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    FrameObservedLabels,
    assigned_accounts,
    base_config,
    hub_registry,
    live_config,
    neighbourhood,
    prepared_dataset,
    scoped_accounts,
    supplied_labels,
)
from mule_pattern_learner.testing.fake_graph import FakeExecutor, FakeSource
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs
from mule_pattern_learner.tigergraph.scope import TigerGraphScope
from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.schedule import step_seed
from mule_pattern_learner.training.trainer import train


def fit(
    tmp_path: Path,
    name: str,
    config: RunConfig,
    *,
    source: FakeSource | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    return trainer.train(
        config,
        DatasetPaths(tmp_path / "dataset"),
        tmp_path / f"{name}.pt",
        contexts=source or FakeSource(config),
        hubs=hub_registry(),
        resume=resume,
    )


def saved_model(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def assert_same_run(tmp_path: Path, left: str, right: str) -> None:
    for name, value in saved_model(tmp_path / f"{left}.pt").items():
        torch.testing.assert_close(
            value, saved_model(tmp_path / f"{right}.pt")[name], rtol=0, atol=0
        )
    a = json.loads((tmp_path / f"{left}_run/metrics.json").read_text())
    b = json.loads((tmp_path / f"{right}_run/metrics.json").read_text())
    for key in ("history", "best_epoch", "validation_proxy", "observed_label_proxy"):
        assert a[key] == b[key], key
    for split in ("validation", "test"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / f"{left}_run/{split}_predictions.parquet"),
            pd.read_parquet(tmp_path / f"{right}_run/{split}_predictions.parquet"),
        )


def after_validation(nth: int) -> Callable[[list[ContextKey], int, Counter[str]], bool]:
    """Fail on the nth training root fetch that follows the first validation fetch."""

    def fail(keys: list[ContextKey], hop: int, calls: Counter[str]) -> bool:
        if hop == 1 and keys[0].visibility_phase == 1 and calls["hop1_phase2"]:
            calls["after"] += 1
            return calls["after"] == nth
        return False

    return fail


@pytest.mark.parametrize("average", [0.0, 0.9], ids=["raw", "averaged"])
def test_two_epochs_equal_one_epoch_plus_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, average: float
) -> None:
    config = base_config(training={"weight_average_decay": average})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "straight", config)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "resumed", config, source=FakeSource(config, fail=after_validation(1)))
    state = torch.load(tmp_path / "resumed_run/checkpoint_last.pt", weights_only=True)
    assert (state["epoch"], state["step"]) == (1, 0)
    assert not (tmp_path / "resumed.pt").exists()
    with pytest.raises(FileExistsError):
        fit(tmp_path, "resumed", config)
    result = fit(tmp_path, "resumed", config, resume=True)
    assert result["status"] == "complete"
    assert_same_run(tmp_path, "straight", "resumed")
    straight = json.loads((tmp_path / "straight_run/metrics.json").read_text())
    # Reported totals cover both segments of the resumed run.
    for key in ("sampler_totals", "database_calls_during_training", "rejected_roots"):
        assert result[key] == straight[key], key
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "resumed_run/progress.jsonl").read_text().splitlines()
    ]
    assert events.count("start") == 1 and events.count("resume") == 1
    with pytest.raises(FileExistsError, match="complete"):
        fit(tmp_path, "resumed", config, resume=True)


@pytest.mark.parametrize("average", [0.0, 0.9], ids=["raw", "averaged"])
def test_mid_epoch_step_checkpoint_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, average: float
) -> None:
    config = base_config(
        runtime={"checkpoint_every_steps": 1},
        training={"steps_per_epoch": 4, "weight_average_decay": average},
    )
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "straight", config)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "resumed", config, source=FakeSource(config, fail=after_validation(3)))
    state = torch.load(tmp_path / "resumed_run/checkpoint_last.pt", weights_only=True)
    assert state["epoch"] == 1 and 1 <= state["step"] < 4
    # Runtime-only settings may change on resume; results must not.
    runtime = {"prefetch_batches": 0, "log_every_steps": 1}
    resumed = fit(tmp_path, "resumed", config.with_changes({"runtime": runtime}), resume=True)
    assert_same_run(tmp_path, "straight", "resumed")
    straight = json.loads((tmp_path / "straight_run/metrics.json").read_text())
    # Steps replayed after the step checkpoint are counted once.
    assert resumed["sampler_totals"] == straight["sampler_totals"]
    assert resumed["rejected_roots"] == straight["rejected_roots"]


def test_prefetch_depth_and_resample_policy_do_not_break_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "inline", config.with_changes({"runtime": {"prefetch_batches": 0}}))
    fit(tmp_path, "threads", config.with_changes({"runtime": {"prefetch_batches": 4}}))
    assert_same_run(tmp_path, "inline", "threads")


def test_resume_refuses_a_changed_result_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.raises(RuntimeError):
        fit(tmp_path, "run", config, source=FakeSource(config, fail=after_validation(1)))
    changed = config.with_changes({"training": {"learning_rate": 0.02}})
    with pytest.raises(ValueError, match="training.learning_rate"):
        fit(tmp_path, "run", changed, resume=True)


def test_batches_use_train_mode_step_seeds_and_the_hub_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    calls: list[tuple[str, int, int, int]] = []
    backends: set[str | None] = set()
    resolved: list[threading.Thread] = []
    lock = threading.Lock()
    real = assemble.make_live_batch
    real_resolve = trainer.resolve_backend

    def resolve(sampler: SamplerPlan, device: torch.device) -> str:
        resolved.append(threading.current_thread())
        return real_resolve(sampler, device)

    def record(store: Any, roots: list[ContextKey], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert isinstance(kwargs["hubs"], HubRegistry) and len(kwargs["hubs"]) == 1
        stats = kwargs["stats"]
        batch = real(store, roots, **kwargs)
        with lock:
            backends.add(kwargs["sampler_backend"])
            calls.append(
                (
                    kwargs["mode"],
                    kwargs["step_seed"],
                    roots[0].visibility_phase,
                    stats["stub_children"],
                )
            )
        return batch

    monkeypatch.setattr(assemble, "make_live_batch", record)
    monkeypatch.setattr(trainer, "resolve_backend", resolve)
    result = fit(tmp_path, "run", config)
    # One resolution per run, on the main thread; every batch gets its result.
    assert resolved == [threading.main_thread()] and backends == {"torch"}
    state = torch.load(tmp_path / "run_run/checkpoint_last.pt", weights_only=True)
    assert state["sampler_backend"] == "torch" and "cuda_rng" not in state
    model = torch.load(tmp_path / "run.pt", weights_only=True)
    assert model["sampler_backend"] == "torch"
    train_calls = [c for c in calls if c[0] == "train"]
    assert all(phase == 1 for _, _, phase, _ in train_calls)
    expected = {step_seed(7, epoch, step) for epoch in range(2) for step in range(3)}
    assert {seed for _, seed, _, _ in train_calls} == expected
    assert all(mode == "eval" and seed == 0 for mode, seed, phase, _ in calls if phase != 1)
    assert sum(stubs for *_, stubs in train_calls) > 0, "the hub child must become a stub"
    assert result["sampler_backend"] == "torch"
    records = [
        json.loads(line) for line in (tmp_path / "run_run/progress.jsonl").read_text().splitlines()
    ]
    train_records = [r for r in records if r["event"] == "train"]
    assert train_records and all(
        {"query_calls", "rejections", "stub_children", "seconds_per_step", "sampler_backend"}
        <= set(r)
        for r in train_records
    )
    # The unclamped risk is logged beside the loss; they agree on steps without a correction.
    for record in train_records:
        assert 0 <= record["corrected_steps"] <= 2 and math.isfinite(record["objective"])
        if record["corrected_steps"] == 0:
            assert record["objective"] == pytest.approx(record["loss"])
    assert {r["event"] for r in records} >= {"start", "train", "evaluate", "epoch", "complete"}


def test_rejected_roots_are_dropped_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(
        training={"proxy_unlabeled_limit": 100}, runtime={"max_rejected_root_fraction": 0.2}
    )
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    # Unlabeled validation, test and train accounts (4 of the 21 marginal train rows,
    # so every epoch's 18-row marginal draw meets at least one).
    rejected = frozenset({"A001", "A002", "A003", "A006", "A009", "A012"})
    result = fit(tmp_path, "run", config, source=FakeSource(config, reject=rejected))
    assert result["rejections"]["missing_entity"] > 0
    assert result["rejected_evaluation_rows"] == {"validation": 1, "test": 1}
    roots = result["rejected_roots"]
    # 23 validation and 22 test rows: unlabeled rows outside the marginal are not drawn.
    assert roots["validation"] == {"requested": 23, "rejected": 1, "positive": 0, "unlabeled": 1}
    assert roots["test"] == {"requested": 22, "rejected": 1, "positive": 0, "unlabeled": 1}
    train = roots["train"]
    # 2 epochs x 3 steps x 8 roots; every rejected training root is unlabeled.
    assert train["requested"] == 48 and train["positive"] == 0
    assert train["rejected"] == train["unlabeled"] >= 2
    for split in ("validation", "test"):
        frame = pd.read_parquet(tmp_path / f"run_run/{split}_predictions.parquet")
        assert not set(frame.account_id) & rejected
        assert frame.score.notna().all()


def test_rejected_roots_fail_closed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(training={"proxy_unlabeled_limit": 100})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    # Default limit 0: one rejected unlabeled validation root fails epoch 1.
    with pytest.raises(ValueError, match=r"validation: TigerGraph rejected 1 of 23 roots"):
        fit(tmp_path, "one", config, source=FakeSource(config, reject=frozenset({"A001"})))
    assert not (tmp_path / "one.pt").exists()
    # A rejected observed positive always fails, whatever the limit.
    loose = base_config(
        training={"proxy_unlabeled_limit": 100}, runtime={"max_rejected_root_fraction": 1.0}
    )
    with pytest.raises(ValueError, match="1 observed positives"):
        fit(tmp_path, "pos", loose, source=FakeSource(loose, reject=frozenset({"A010"})))
    # A training positive (A015) fails the training epoch before validation.
    with pytest.raises(ValueError, match=r"Epoch 1: .*training roots .*observed positives"):
        fit(tmp_path, "train", loose, source=FakeSource(loose, reject=frozenset({"A015"})))
    # Validation must keep both observed classes after its rejections.
    validation_positives = frozenset({"A010", "A025", "A040", "A055", "A070"})
    unlabeled = frozenset(f"A{i:03}" for i in range(1, 72, 3)) - validation_positives
    with pytest.raises(ValueError, match="both observed classes"):
        fit(tmp_path, "classes", loose, source=FakeSource(loose, reject=unlabeled))


def test_test_split_rejections_fail_after_the_model_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(training={"proxy_unlabeled_limit": 100})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.raises(ValueError, match=r"test: TigerGraph rejected 1 of 22 roots"):
        fit(tmp_path, "run", config, source=FakeSource(config, reject=frozenset({"A002"})))
    assert (tmp_path / "run.pt").exists() and not (tmp_path / "run_run/metrics.json").exists()
    # The limit is a runtime key: raising it lets the run finish from its checkpoint.
    result = fit(
        tmp_path,
        "run",
        config.with_changes({"runtime": {"max_rejected_root_fraction": 0.1}}),
        source=FakeSource(config, reject=frozenset({"A002"})),
        resume=True,
    )
    assert result["status"] == "complete" and result["max_rejected_root_fraction"] == 0.1
    assert result["rejected_roots"]["test"]["rejected"] == 1


def test_no_finite_validation_ap_refuses_to_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    real = trainer.evaluate

    def no_ap(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {**real(*args, **kwargs), "average_precision": None}

    monkeypatch.setattr(trainer, "evaluate", no_ap)
    with pytest.raises(ValueError, match="refusing to save untrained weights"):
        fit(tmp_path, "run", config)
    assert not (tmp_path / "run.pt").exists()


def test_non_finite_evaluation_scores_raise_instead_of_counting_as_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    original = LiveTGAT.forward

    def poisoned(self: LiveTGAT, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = original(self, batch)
        if self.training:
            return logits
        return torch.cat((torch.full_like(logits[:1], float("nan")), logits[1:]))

    monkeypatch.setattr(LiveTGAT, "forward", poisoned)
    with pytest.raises(ValueError, match="Non-finite model probability .* accepted validation"):
        fit(tmp_path, "run", config)


def test_evaluation_scores_keep_float64_resolution_near_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    original = LiveTGAT.forward

    def confident(self: LiveTGAT, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = original(self, batch)
        # Evaluation logits near 20, where a float32 probability is exactly 1.
        return logits if self.training else logits + 20

    monkeypatch.setattr(LiveTGAT, "forward", confident)
    # The validation F1 threshold falls between scores that float32 would tie at 1.
    assert 0.99 < fit(tmp_path, "run", config)["validation_proxy"]["threshold"] < 1
    for split in ("validation", "test"):
        path = tmp_path / f"run_run/{split}_predictions.parquet"
        assert pq.read_schema(path).field("score").type == pa.float64()
        frame = pd.read_parquet(path)
        assert (frame.score < 1).all() and (frame.score.astype(np.float32) == 1).all()
        assert frame.score.nunique() == len(frame)


def test_patience_zero_disables_early_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(training={"epochs": 3, "patience": 0, "steps_per_epoch": 1})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    ap = iter([0.5, 0.4, 0.3])  # validation never improves after epoch 1
    real = trainer.evaluate

    def falling(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = real(*args, **kwargs)
        if len(args) > 2 and args[2] == 0.5:  # the per-epoch validation proxy
            result["average_precision"] = next(ap, result["average_precision"])
        return result

    monkeypatch.setattr(trainer, "evaluate", falling)
    result = fit(tmp_path, "run", config)
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3] and result["best_epoch"] == 1


def test_resume_refuses_a_different_sampler_backend_unless_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "straight", config)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "run", config, source=FakeSource(config, fail=after_validation(1)))
    path = tmp_path / "run_run/checkpoint_last.pt"
    state = torch.load(path, weights_only=True)
    # As if the first segment ran on a cuGraph host.
    torch.save({**state, "sampler_backend": "cugraph"}, path)
    with pytest.raises(ValueError, match="sampled with the cugraph backend .* resolves torch"):
        fit(tmp_path, "run", config, resume=True)
    explicit = base_config(sampler={"backend": "torch"})
    result = fit(tmp_path, "run", explicit, resume=True)
    assert result["status"] == "complete" and result["sampler_backend"] == "torch"
    assert_same_run(tmp_path, "straight", "run")


def test_missing_hub_indicator_warns_once_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups = [g for g in DEFAULT_GROUPS if g != "hub_indicator"]
    config = base_config(features=groups, training={"epochs": 1})
    plan = config.feature_plan()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_hub_stubs(HubRegistry.empty(), plan)
        warn_hub_stubs(hub_registry(), base_config().feature_plan())
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.warns(UserWarning, match="no hub_indicator group") as caught:
        fit(tmp_path, "run", config)
    assert sum("hub_indicator" in str(w.message) for w in caught) == 1


def test_run_directory_is_created_only_after_the_source_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)

    def refuse(dataset: DatasetPaths, manifest: dict[str, Any], config: RunConfig) -> NoReturn:
        raise ValueError("Live graph counts changed")

    with pytest.raises(ValueError, match="counts changed"):
        trainer.train(
            config,
            DatasetPaths(tmp_path / "dataset"),
            tmp_path / "m.pt",
            open_contexts=refuse,
            hubs=hub_registry(),
        )
    assert not (tmp_path / "m_run").exists() and not (tmp_path / "m.pt").exists()
    # A source built with another sampler is rejected before anything is written.
    other = base_config(sampler={"relation_fanouts": [3, 2]})
    with pytest.raises(ValueError, match="sampler"):
        fit(tmp_path, "m", config, source=FakeSource(other))
    assert not (tmp_path / "m_run").exists()


def test_non_finite_loss_stops_before_any_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(runtime={"checkpoint_every_steps": 1})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    original = NonNegativePULoss.forward

    def poisoned(self: NonNegativePULoss, logits: torch.Tensor, targets: torch.Tensor):
        train_loss, objective = original(self, logits, targets)
        return train_loss * float("nan"), objective

    monkeypatch.setattr(NonNegativePULoss, "forward", poisoned)
    with pytest.raises(ValueError, match="Non-finite"):
        fit(tmp_path, "run", config)
    assert not (tmp_path / "run_run/checkpoint_last.pt").exists()


def test_train_restores_global_torch_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = base_config(training={"epochs": 1})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(False)
    try:
        fit(tmp_path, "ok", config)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == threads
        with pytest.raises(RuntimeError):
            fit(tmp_path, "bad", config, source=FakeSource(config, fail=lambda *_: True))
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == threads
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_fake_end_to_end_training_on_mps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = base_config(runtime={"device": "mps"})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    result = fit(tmp_path, "mps", config)
    assert result["device"] == "mps" and result["status"] == "complete"
    assert all(v.device.type == "cpu" for v in saved_model(tmp_path / "mps.pt").values())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fake_end_to_end_training_on_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = base_config(runtime={"device": "cuda", "deterministic": "strict"})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    assert fit(tmp_path, "cuda", config)["status"] == "complete"


def test_averaged_run_validates_and_saves_the_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One epoch, so the selected state is the average at its end.
    config = base_config(training={"weight_average_decay": 0.9, "epochs": 1})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    result = fit(tmp_path, "averaged", config)
    assert result["best_epoch"] == 1
    assert 0.0 <= result["history"][0]["validation_proxy_roc_auc"] <= 1.0
    state = torch.load(tmp_path / "averaged_run/checkpoint_last.pt", weights_only=True)
    averaged, raw = state["weight_average"]["state"], state["model"]
    saved = saved_model(tmp_path / "averaged.pt")
    assert all(torch.equal(saved[k], averaged[k]) for k in saved)
    assert all(torch.equal(state["best_state"][k], averaged[k]) for k in saved)
    assert any(not torch.equal(averaged[k], raw[k]) for k in raw)


def streaming_source(executor: FakeExecutor, config: RunConfig, **kwargs: Any):
    """The source a prepared run opens: prepared extraction plan and training sampler."""
    return StreamingContextSource(
        TigerGraphContextFetcher(executor),
        plan=extraction_plan(config.feature_plan()),
        sampler=config.sampler,
        **kwargs,
    )


class PreparedExecutor(FakeExecutor):
    """The scope population, context, cutoff and hub queries.

    The scope population holds the fixture accounts with their splits as partitions.
    """

    def __init__(self, dataset: DatasetPaths, **kwargs: Any) -> None:
        super().__init__(population=scoped_accounts(), **kwargs)
        self.dataset = dataset

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        if name == "temporal_training_context":
            # A label mask must already be frozen when the first feature query starts.
            frozen = pd.read_parquet(self.dataset.observed_labels)
            assert label_summary(frozen) == {"train": 20, "validation": 20, "test": 20}
        return super().run(name, params, **kwargs)


def prepared(tmp_path: Path, config: RunConfig, **kwargs: Any) -> tuple[DatasetPaths, FakeExecutor]:
    dataset = DatasetPaths(tmp_path / "dataset")
    executor = PreparedExecutor(dataset, **kwargs)
    prepare(
        config,
        UNIT_SOURCE,
        dataset,
        {"Account": 1000},
        FrameObservedLabels(supplied_labels()),
        scope=TigerGraphScope(executor),
        cutoffs=TigerGraphCutoffs(executor),
        hubs=TigerGraphHubs(executor),
    )
    return dataset, executor


def test_hidden_truth_cannot_change_updates_or_checkpoint_selection(tmp_path: Path) -> None:
    c = live_config()
    dataset, executor = prepared(tmp_path, c)
    assert executor.names().count("temporal_hub_registry") == 1
    first = train(c, dataset, tmp_path / "first.pt", contexts=streaming_source(executor, c))
    saved_first = torch.load(tmp_path / "first.pt", weights_only=True)
    # The oracle is a separate file that is never opened by training.
    a = pd.read_parquet(dataset.accounts)
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


def test_preparation_requests_no_context_and_training_keeps_a_bounded_lru(tmp_path: Path) -> None:
    c = live_config()
    dataset, executor = prepared(tmp_path, c)
    load_prepared(dataset)
    assert not executor.requested
    source = streaming_source(executor, c, capacity=4)
    result = train(c, dataset, tmp_path / "model.pt", contexts=source)
    assert result["database_calls_during_training"] > 0
    assert len(source.memory) <= 4


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
    plan, sampler = c.feature_plan(), c.sampler
    assert plan.architecture == "split"
    hubs = load_hub_registry(dataset, manifest)
    train_rows = accounts[accounts.split == "train"].iloc[:16]
    keys = sample_keys(train_rows, c.dataset.dates.train[0], manifest)
    stats: dict[str, Any] = {}
    with streaming_source(executor, c) as source:
        batch = make_live_batch(
            source,
            keys,
            fanouts=sampler.fanouts,
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
    prior = c.loss.class_prior
    loss, _ = NonNegativePULoss(prior=prior, positive_weight=prior)(logits, targets)
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
    c = live_config(runtime={"max_rejected_root_fraction": 0.05})
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
