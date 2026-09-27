"""Training: resumes, determinism, rejections, selection and the saved model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
import math
from pathlib import Path
import threading
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mule_pattern_learner.artifacts import (
    read_epochs,
    read_events,
    read_history,
    read_run_config,
)
from mule_pattern_learner.batching import assemble
from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.config import RunConfig
from mule_pattern_learner.contract.feature_groups import CORE_GROUPS, extraction_plan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY, HUB_QUERY
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.data.hub_registry import HubRegistry, load_hub_registry, warn_hub_stubs
from mule_pattern_learner.data.manifest import dataset_id, load_prepared
from mule_pattern_learner.data.observed_labels import label_summary
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.data.splits import sample_keys
from mule_pattern_learner.model.loss import NonNegativePULoss
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.paths import DatasetPaths, RunPaths
from mule_pattern_learner.testing.builders import (
    RUNTIME_SOURCE,
    UNIT_SOURCE,
    FrameObservedLabels,
    assigned_accounts,
    base_config,
    example_config,
    hub_registry,
    neighbourhood,
    prepared_dataset,
    scoped_accounts,
    supplied_labels,
)
from mule_pattern_learner.testing.fake_graph import FakeSource, FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubs
from mule_pattern_learner.tigergraph.scope import TigerGraphScope
from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.schedule import step_seed
from mule_pattern_learner.training.summary import PACKAGES
from mule_pattern_learner.training.trainer import train


def fit(
    tmp_path: Path,
    name: str,
    config: RunConfig,
    *,
    contexts: FakeSource | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    return trainer.train(
        config,
        DatasetPaths(tmp_path / "dataset"),
        RunPaths(tmp_path / name),
        contexts=contexts or FakeSource(config),
        hubs=hub_registry(),
        resume=resume,
    )


def saved_model(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def assert_same_run(tmp_path: Path, left: str, right: str) -> None:
    one, other = RunPaths(tmp_path / left), RunPaths(tmp_path / right)
    for name, value in saved_model(one.model).items():
        torch.testing.assert_close(value, saved_model(other.model)[name], rtol=0, atol=0)
    a = json.loads(one.metrics.read_text())
    b = json.loads(other.metrics.read_text())
    for key in ("best_epoch", "validation_proxy", "observed_label_proxy"):
        assert a[key] == b[key], key
    pd.testing.assert_frame_equal(read_epochs(one.epochs), read_epochs(other.epochs))
    for split in ("validation", "test"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(one.predictions(split)), pd.read_parquet(other.predictions(split))
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
        fit(tmp_path, "resumed", config, contexts=FakeSource(config, fail=after_validation(1)))
    state = torch.load(RunPaths(tmp_path / "resumed").resume, weights_only=True)
    assert (state["epoch"], state["step"]) == (1, 0)
    assert not RunPaths(tmp_path / "resumed").model.exists()
    with pytest.raises(FileExistsError):
        fit(tmp_path, "resumed", config)
    result = fit(tmp_path, "resumed", config, resume=True)
    assert result["status"] == "complete"
    assert_same_run(tmp_path, "straight", "resumed")
    straight = json.loads(RunPaths(tmp_path / "straight").metrics.read_text())
    # Reported totals cover both segments of the resumed run.
    for key in ("sampler_totals", "database_calls_during_training", "rejected_roots", "contexts"):
        assert result[key] == straight[key], key
    events = [
        json.loads(line)["event"]
        for line in RunPaths(tmp_path / "resumed").events.read_text().splitlines()
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
        fit(tmp_path, "resumed", config, contexts=FakeSource(config, fail=after_validation(3)))
    state = torch.load(RunPaths(tmp_path / "resumed").resume, weights_only=True)
    assert state["epoch"] == 1 and 1 <= state["step"] < 4
    # Runtime-only settings may change on resume; results must not.
    runtime = {"prefetch_batches": 0, "log_every_steps": 1}
    resumed = fit(tmp_path, "resumed", config.with_changes({"runtime": runtime}), resume=True)
    assert_same_run(tmp_path, "straight", "resumed")
    straight = json.loads(RunPaths(tmp_path / "straight").metrics.read_text())
    # Steps replayed after the step checkpoint are counted once.
    assert resumed["sampler_totals"] == straight["sampler_totals"]
    assert resumed["rejected_roots"] == straight["rejected_roots"]
    # Contexts prefetched before the step checkpoint are requested again, but not new.
    assert resumed["contexts"]["distinct"] == straight["contexts"]["distinct"] > 0


def test_a_resume_records_changed_host_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "run", config, contexts=FakeSource(config, fail=after_validation(1)))
    run = RunPaths(tmp_path / "run")
    started = read_run_config(run.config), json.loads(run.config.read_text())["provenance"]
    faster = config.with_changes({"runtime": {"threads": 2, "deterministic": False}})
    fit(tmp_path, "run", faster, resume=True)
    events = read_events(run.events)
    (change,) = [e for e in events if e["event"] == "host_settings"]
    assert change["saved"] == {"deterministic": True, "threads": 1}
    assert change["resumed"] == {"deterministic": False, "threads": 2}
    (resumed,) = [e for e in events if e["event"] == "resume"]
    assert (resumed["device"], resumed["threads"], resumed["deterministic"]) == ("cpu", 2, False)
    # config.json keeps the settings and host the run started with.
    assert (read_run_config(run.config), json.loads(run.config.read_text())["provenance"]) == (
        started
    )


def test_a_resumed_run_logs_each_interval_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every step is logged but only epochs are checkpointed, so the failed segment logs
    # two steps of epoch 2 that the resumed run trains, and logs, again.
    config = base_config(
        training={"steps_per_epoch": 4}, runtime={"log_every_steps": 1, "prefetch_batches": 0}
    )
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "straight", config)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "resumed", config, contexts=FakeSource(config, fail=after_validation(3)))
    resumed = RunPaths(tmp_path / "resumed")
    assert read_history(resumed.history)[["epoch", "step"]].to_numpy().tolist()[-2:] == [
        [2, 1],
        [2, 2],
    ]
    fit(tmp_path, "resumed", config, resume=True)
    trained = ["epoch", "step", "date", "loss", "objective", "corrected_steps", "steps"]
    straight = read_history(RunPaths(tmp_path / "straight").history)
    assert len(straight) == 8
    pd.testing.assert_frame_equal(read_history(resumed.history)[trained], straight[trained])
    assert_same_run(tmp_path, "straight", "resumed")


def test_the_run_records_its_settings_provenance_and_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    result = fit(tmp_path, "run", config)
    run = RunPaths(tmp_path / "run")
    record = json.loads(run.config.read_text())
    assert set(record) == {"config", "fingerprint", "provenance"}
    assert record["config"] == config.to_dict() and record["fingerprint"] == config.fingerprint()
    assert read_run_config(run.config) == config
    provenance = record["provenance"]
    assert set(provenance) == {
        "git_commit",
        "git_dirty",
        "versions",
        "device",
        "threads",
        "deterministic",
        "sampler_backend",
        "dataset_id",
        "started",
    }
    assert set(provenance["versions"]) == set(PACKAGES) and provenance["versions"]["torch"]
    assert (provenance["device"], provenance["sampler_backend"]) == ("cpu", "torch")
    assert (provenance["threads"], provenance["deterministic"]) == (1, True)
    assert provenance["dataset_id"] == dataset_id(RUNTIME_SOURCE, config) == result["dataset_id"]
    # The metrics hold no history: its intervals and epochs have their own files.
    assert "history" not in json.loads(run.metrics.read_text())
    assert result["proxy_unlabeled_limit"] == config.training.proxy_unlabeled_limit
    epochs = read_epochs(run.epochs)
    assert epochs.epoch.tolist() == [1, 2] and epochs.selected.sum() == 1
    assert epochs.selected.tolist()[result["best_epoch"] - 1]
    assert (epochs.weights == "raw").all() and (epochs.steps == 3).all()
    history = read_history(run.history)
    # log_every_steps = 2 over 3 steps: an interval of 2 steps and one of 1, per epoch.
    assert history[["epoch", "step"]].to_numpy().tolist() == [[1, 2], [1, 3], [2, 2], [2, 3]]
    assert history.database_calls.is_monotonic_increasing and history.rejected_roots.eq(0).all()


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
        fit(tmp_path, "run", config, contexts=FakeSource(config, fail=after_validation(1)))
    changed = config.with_changes({"training": {"learning_rate": 0.02}})
    with pytest.raises(ValueError, match="training.learning_rate"):
        fit(tmp_path, "run", changed, resume=True)


def test_resume_refuses_a_dataset_other_than_the_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)

    def first_fetch(keys: list[ContextKey], hop: int, calls: Counter[str]) -> bool:
        return True

    for name, fail in (("checkpointed", after_validation(1)), ("early", first_fetch)):
        with pytest.raises(RuntimeError):
            fit(tmp_path, name, config, contexts=FakeSource(config, fail=fail))
    assert RunPaths(tmp_path / "checkpointed").resume.exists()
    # Stopped before its first checkpoint: config.json names the dataset.
    assert not RunPaths(tmp_path / "early").resume.exists()
    ours = dataset_id(RUNTIME_SOURCE, config)
    # The graph was reloaded and a new dataset prepared for the same settings.
    other, _, _ = prepared_dataset(tmp_path / "other", config, monkeypatch, "another_source")
    theirs = dataset_id("another_source", config)
    for name in ("checkpointed", "early"):
        with pytest.raises(ValueError, match=f"another dataset: dataset id {ours} .given {theirs}"):
            trainer.train(
                config,
                other,
                RunPaths(tmp_path / name),
                contexts=FakeSource(config),
                hubs=hub_registry(),
                resume=True,
            )
    # The same dataset prepared again is another dataset too: its manifest changed.
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    (tmp_path / "dataset" / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="another dataset: dataset manifest sha256"):
        fit(tmp_path, "checkpointed", config, resume=True)


def test_batches_use_train_mode_step_seeds_and_the_hub_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    calls: list[tuple[str, int, int, int]] = []
    backends: set[str | None] = set()
    resolved: list[threading.Thread] = []
    lock = threading.Lock()
    real = assemble.build_batch
    real_resolve = trainer.resolve_backend

    def resolve(sampler: SamplerPlan, device: torch.device) -> str:
        resolved.append(threading.current_thread())
        return real_resolve(sampler, device)

    def record(contexts: Any, roots: list[ContextKey], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert isinstance(kwargs["hubs"], HubRegistry) and len(kwargs["hubs"]) == 1
        stats = kwargs["stats"]
        batch = real(contexts, roots, **kwargs)
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

    monkeypatch.setattr(assemble, "build_batch", record)
    monkeypatch.setattr(trainer, "resolve_backend", resolve)
    result = fit(tmp_path, "run", config)
    # One resolution per run, on the main thread; every batch gets its result.
    assert resolved == [threading.main_thread()] and backends == {"torch"}
    state = torch.load(RunPaths(tmp_path / "run").resume, weights_only=True)
    assert state["sampler_backend"] == "torch" and state["cuda_rng"] is None
    model = torch.load(RunPaths(tmp_path / "run").model, weights_only=True)
    assert model["sampler_backend"] == "torch"
    train_calls = [c for c in calls if c[0] == "train"]
    assert all(phase == 1 for _, _, phase, _ in train_calls)
    expected = {step_seed(7, epoch, step) for epoch in range(2) for step in range(3)}
    assert {seed for _, seed, _, _ in train_calls} == expected
    assert all(mode == "eval" and seed == 0 for mode, seed, phase, _ in calls if phase != 1)
    assert sum(stubs for *_, stubs in train_calls) > 0, "the hub child must become a stub"
    assert result["sampler_backend"] == "torch"
    records = [
        json.loads(line) for line in RunPaths(tmp_path / "run").events.read_text().splitlines()
    ]
    train_records = [r for r in records if r["event"] == "train"]
    counters = {"database_calls", "rejections", "stub_children", "seconds_per_step", "cache_hits"}
    counters |= {"contexts_requested", "contexts_distinct", "sampler_backend"}
    assert train_records and all(counters <= set(r) for r in train_records)
    assert 0 < train_records[-1]["contexts_distinct"] <= train_records[-1]["contexts_requested"]
    # The unclamped risk is logged beside the loss; they agree on steps without a correction.
    for logged in train_records:
        assert 0 <= logged["corrected_steps"] <= 2 and math.isfinite(logged["objective"])
        if logged["corrected_steps"] == 0:
            assert logged["objective"] == pytest.approx(logged["loss"])
    assert {r["event"] for r in records} >= {"start", "train", "score", "epoch", "complete"}


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
    result = fit(tmp_path, "run", config, contexts=FakeSource(config, reject=rejected))
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
        frame = pd.read_parquet(RunPaths(tmp_path / "run").predictions(split))
        assert not set(frame.account_id) & rejected
        assert frame.score.notna().all()


def test_rejected_roots_fail_closed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(training={"proxy_unlabeled_limit": 100})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    # Default limit 0: one rejected unlabeled validation root fails epoch 1.
    with pytest.raises(ValueError, match=r"validation: TigerGraph rejected 1 of 23 roots"):
        fit(tmp_path, "one", config, contexts=FakeSource(config, reject=frozenset({"A001"})))
    assert not RunPaths(tmp_path / "one").model.exists()
    # A rejected observed positive always fails, whatever the limit.
    loose = base_config(
        training={"proxy_unlabeled_limit": 100}, runtime={"max_rejected_root_fraction": 1.0}
    )
    with pytest.raises(ValueError, match="1 observed positives"):
        fit(tmp_path, "pos", loose, contexts=FakeSource(loose, reject=frozenset({"A010"})))
    # A training positive (A015) fails the training epoch before validation.
    with pytest.raises(ValueError, match=r"Epoch 1: .*training roots .*observed positives"):
        fit(tmp_path, "train", loose, contexts=FakeSource(loose, reject=frozenset({"A015"})))
    # Validation must keep both observed classes after its rejections.
    validation_positives = frozenset({"A010", "A025", "A040", "A055", "A070"})
    unlabeled = frozenset(f"A{i:03}" for i in range(1, 72, 3)) - validation_positives
    with pytest.raises(ValueError, match="both observed classes"):
        fit(tmp_path, "classes", loose, contexts=FakeSource(loose, reject=unlabeled))


def test_test_split_rejections_fail_after_the_model_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(training={"proxy_unlabeled_limit": 100})
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.raises(ValueError, match=r"test: TigerGraph rejected 1 of 22 roots"):
        fit(tmp_path, "run", config, contexts=FakeSource(config, reject=frozenset({"A002"})))
    run = RunPaths(tmp_path / "run")
    assert run.model.exists() and not run.metrics.exists()
    # The limit is a runtime key: raising it lets the run finish from its checkpoint.
    result = fit(
        tmp_path,
        "run",
        config.with_changes({"runtime": {"max_rejected_root_fraction": 0.1}}),
        contexts=FakeSource(config, reject=frozenset({"A002"})),
        resume=True,
    )
    assert result["status"] == "complete" and result["max_rejected_root_fraction"] == 0.1
    assert result["rejected_roots"]["test"]["rejected"] == 1


def test_no_finite_validation_ap_refuses_to_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    real = trainer.proxy_metrics

    def no_ap(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {**real(*args, **kwargs), "average_precision": None}

    monkeypatch.setattr(trainer, "proxy_metrics", no_ap)
    with pytest.raises(ValueError, match="refusing to save untrained weights"):
        fit(tmp_path, "run", config)
    assert not RunPaths(tmp_path / "run").model.exists()


def test_non_finite_evaluation_scores_raise_instead_of_counting_as_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    original = TGAT.forward

    def poisoned(self: TGAT, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = original(self, batch)
        if self.training:
            return logits
        return torch.cat((torch.full_like(logits[:1], float("nan")), logits[1:]))

    monkeypatch.setattr(TGAT, "forward", poisoned)
    with pytest.raises(ValueError, match="Non-finite model probability .* accepted validation"):
        fit(tmp_path, "run", config)


def test_evaluation_scores_keep_float64_resolution_near_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    original = TGAT.forward

    def confident(self: TGAT, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = original(self, batch)
        # Evaluation logits near 20, where a float32 probability is exactly 1.
        return logits if self.training else logits + 20

    monkeypatch.setattr(TGAT, "forward", confident)
    # The validation F1 threshold falls between scores that float32 would tie at 1.
    assert 0.99 < fit(tmp_path, "run", config)["validation_proxy"]["threshold"] < 1
    for split in ("validation", "test"):
        path = RunPaths(tmp_path / "run").predictions(split)
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
    real = trainer.proxy_metrics

    def falling(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = real(*args, **kwargs)
        if len(args) > 2 and args[2] == 0.5:  # the per-epoch validation proxy
            result["average_precision"] = next(ap, result["average_precision"])
        return result

    monkeypatch.setattr(trainer, "proxy_metrics", falling)
    result = fit(tmp_path, "run", config)
    epochs = read_epochs(RunPaths(tmp_path / "run").epochs)
    assert epochs.epoch.tolist() == [1, 2, 3] and result["best_epoch"] == 1
    assert epochs.selected.tolist() == [True, False, False] and not epochs.stopped.any()


def test_resume_refuses_a_different_sampler_backend_unless_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "straight", config)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "run", config, contexts=FakeSource(config, fail=after_validation(1)))
    path = RunPaths(tmp_path / "run").resume
    state = torch.load(path, weights_only=True)
    # As if the first segment ran on a cuGraph host.
    torch.save({**state, "sampler_backend": "cugraph"}, path)
    with pytest.raises(ValueError, match="sampled with the cugraph backend .* resolves torch"):
        fit(tmp_path, "run", config, resume=True)
    explicit = base_config(sampler={"backend": "torch"})
    result = fit(tmp_path, "run", explicit, resume=True)
    assert result["status"] == "complete" and result["sampler_backend"] == "torch"
    # The change of stream is recorded with the run's events.
    changes = [e for e in read_events(RunPaths(tmp_path / "run").events) if "saved" in e]
    assert [(e["event"], e["saved"], e["resumed"]) for e in changes] == [
        ("sampler_backend", "cugraph", "torch")
    ]
    assert_same_run(tmp_path, "straight", "run")


def test_missing_hub_indicator_warns_once_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    groups = [g for g in CORE_GROUPS if g != "hub_indicator"]
    config = base_config(features=groups, training={"epochs": 1})
    plan = config.feature_plan()
    warn_hub_stubs(HubRegistry.empty(), plan)
    warn_hub_stubs(hub_registry(), base_config().feature_plan())
    assert capsys.readouterr().out == ""
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "run", config)
    warned = [e for e in read_events(RunPaths(tmp_path / "run").events) if e["event"] == "warning"]
    assert [e["warning"] for e in warned] == ["hub_stubs"]
    assert "no hub_indicator group" in warned[0]["message"]


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
            RunPaths(tmp_path / "m"),
            open_contexts=refuse,
            hubs=hub_registry(),
        )
    assert not RunPaths(tmp_path / "m").root.exists()
    # A source built with another sampler is rejected before anything is written.
    other = base_config(sampler={"relation_fanouts": [3, 2]})
    with pytest.raises(ValueError, match="sampler"):
        fit(tmp_path, "m", config, contexts=FakeSource(other))
    assert not RunPaths(tmp_path / "m").root.exists()


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
    assert not RunPaths(tmp_path / "run").resume.exists()


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
            fit(tmp_path, "bad", config, contexts=FakeSource(config, fail=lambda *_: True))
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
    assert all(
        v.device.type == "cpu" for v in saved_model(RunPaths(tmp_path / "mps").model).values()
    )


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
    epochs = read_epochs(RunPaths(tmp_path / "averaged").epochs)
    assert epochs.weights.tolist() == ["averaged"]
    assert 0.0 <= float(epochs.validation_roc_auc.iloc[0]) <= 1.0
    state = torch.load(RunPaths(tmp_path / "averaged").resume, weights_only=True)
    averaged, raw = state["weight_average"]["state"], state["model"]
    saved = saved_model(RunPaths(tmp_path / "averaged").model)
    assert all(torch.equal(saved[k], averaged[k]) for k in saved)
    assert all(torch.equal(state["best_state"][k], averaged[k]) for k in saved)
    assert any(not torch.equal(averaged[k], raw[k]) for k in raw)


def build_context_source(executor: FakeTigerGraph, config: RunConfig, **kwargs: Any):
    """The source a prepared run opens: prepared extraction plan and training sampler."""
    return ContextSource(
        TigerGraphContextFetcher(executor),
        plan=extraction_plan(config.feature_plan()),
        sampler=config.sampler,
        **kwargs,
    )


class PreparedExecutor(FakeTigerGraph):
    """The scope population, context, cutoff and hub queries.

    The scope population holds the fixture accounts with their splits as partitions.
    """

    def __init__(self, dataset: DatasetPaths, **kwargs: Any) -> None:
        super().__init__(population=scoped_accounts(), **kwargs)
        self.dataset = dataset

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        if name == CONTEXT_QUERY:
            # A label mask must already be frozen when the first feature query starts.
            frozen = pd.read_parquet(self.dataset.observed_labels)
            assert label_summary(frozen) == {"train": 20, "validation": 20, "test": 20}
        return super().run(name, params, **kwargs)


def prepared(
    tmp_path: Path, config: RunConfig, **kwargs: Any
) -> tuple[DatasetPaths, FakeTigerGraph]:
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
        hub_reader=TigerGraphHubs(executor),
    )
    return dataset, executor


def test_hidden_truth_cannot_change_updates_or_checkpoint_selection(tmp_path: Path) -> None:
    c = example_config()
    dataset, executor = prepared(tmp_path, c)
    assert executor.names().count(HUB_QUERY) == 1
    first = train(
        c, dataset, RunPaths(tmp_path / "first"), contexts=build_context_source(executor, c)
    )
    saved_first = torch.load(RunPaths(tmp_path / "first").model, weights_only=True)
    # The oracle is a separate file that is never opened by training: inverting it
    # between two runs changes nothing.
    a = pd.read_parquet(dataset.accounts)
    assert "is_mule" not in a.columns
    truth = a[["account_id"]].assign(is_mule=np.arange(len(a)) % 2)
    truth_path = tmp_path / "evaluation_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    truth["is_mule"] = 1 - truth.is_mule
    truth.to_parquet(truth_path, index=False)
    second = train(
        c, dataset, RunPaths(tmp_path / "second"), contexts=build_context_source(executor, c)
    )
    saved_second = torch.load(RunPaths(tmp_path / "second").model, weights_only=True)
    pd.testing.assert_frame_equal(
        read_epochs(RunPaths(tmp_path / "first").epochs),
        read_epochs(RunPaths(tmp_path / "second").epochs),
    )
    assert first["best_epoch"] == second["best_epoch"]
    assert first["validation_proxy"] == second["validation_proxy"]
    assert saved_first["threshold"] == saved_second["threshold"]
    assert first["observed_label_proxy"] == second["observed_label_proxy"]
    for name, value in saved_first["state_dict"].items():
        torch.testing.assert_close(value, saved_second["state_dict"][name], rtol=0, atol=0)
    assert first["database_calls_during_training"] > 0
    assert first["known_mules"] == {"train": 20, "validation": 20, "test": 20}


def test_preparation_requests_no_context_and_training_keeps_a_bounded_lru(tmp_path: Path) -> None:
    c = example_config()
    dataset, executor = prepared(tmp_path, c)
    load_prepared(dataset)
    assert not executor.requested
    source = build_context_source(executor, c, capacity=4)
    result = train(c, dataset, RunPaths(tmp_path / "model"), contexts=source)
    assert result["database_calls_during_training"] > 0
    assert len(source.memory) <= 4


def test_training_end_to_end_with_candidate_pool_neighbour_messages(tmp_path: Path) -> None:
    c = example_config()
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
    assert plan.architecture == "tgat"
    hubs = load_hub_registry(dataset, manifest)
    train_rows = accounts[accounts.split == "train"].iloc[:16]
    keys = sample_keys(train_rows, c.dataset.dates.train[0], manifest)
    stats: dict[str, Any] = {}
    with build_context_source(executor, c) as source:
        batch = build_batch(
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
    model = TGAT(16, 4, 0, plan=plan, slot_sum=False, first_fanout=8)
    logits = model(batch)
    targets = torch.zeros_like(logits)
    targets[:4] = 1
    prior = c.loss.class_prior
    loss, _ = NonNegativePULoss(prior=prior, positive_weight=prior)(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.edge, model.relation, model.rail):
        assert module.weight.grad is not None and torch.count_nonzero(module.weight.grad) > 0

    source = build_context_source(executor, c, capacity=64)
    result = train(c, dataset, RunPaths(tmp_path / "model"), contexts=source)
    assert result["status"] == "complete"
    assert np.isfinite(read_epochs(RunPaths(tmp_path / "model").epochs).loss).all()
    assert result["sampler_backend"] == "torch"
    assert result["sampler_totals"]["stub_children"] > 0
    assert result["sampler_totals"]["rejected_children"] > 0
    assert result["rejections"]["history_capacity_exceeded"] > 0
    assert result["database_calls_during_training"] > 0
    # Roots and children were requested with their own pools; hubs were never fetched.
    assert set(executor.pools) == {tuple(sampler.query_params(hop).values()) for hop in (1, 2)}
    assert "N3" not in {k.node_id for k in executor.requested}
    progress = RunPaths(tmp_path / "model").events.read_text().splitlines()
    assert json.loads(progress[-1])["event"] == "complete"


def test_rejected_roots_within_the_limit_are_dropped_and_counted(tmp_path: Path) -> None:
    # Rejected roots fail closed by default; this run tolerates up to 5% per split.
    c = example_config(runtime={"max_rejected_root_fraction": 0.05})
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
    result = train(
        c, dataset, RunPaths(tmp_path / "model"), contexts=build_context_source(executor, c)
    )
    assert result["sampler_totals"]["rejected_children"] > 0
    assert result["rejections"]["invisible_entity"] >= 1
    assert result["rejected_roots"]["validation"]["rejected"] == 1
    assert result["rejected_roots"]["validation"]["positive"] == 0
    assert result["max_rejected_root_fraction"] == 0.05


def test_rejected_roots_fail_the_run_under_the_default_limit(tmp_path: Path) -> None:
    c = example_config()
    validation = assigned_accounts().query("split == 'validation'").account_id
    dataset, executor = prepared(
        tmp_path, c, factory=neighbourhood, statuses={str(validation.iloc[20]): "invisible_entity"}
    )
    with pytest.raises(ValueError, match="validation: TigerGraph rejected 1 of"):
        train(c, dataset, RunPaths(tmp_path / "model"), contexts=build_context_source(executor, c))
    assert not RunPaths(tmp_path / "model").model.exists()
