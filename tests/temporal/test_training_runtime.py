"""Training runtime: nnPU parity, schedule, prefetch, determinism, checkpoints and scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.device import torch_runtime
from mule_pattern_learner.temporal.common import digest, timestamp
from mule_pattern_learner.temporal.encoding import BASIS_ID
from mule_pattern_learner.temporal.live import (
    batching,
    cli,
    inference,
    pipeline,
    predictor,
    training,
)
from mule_pattern_learner.temporal.live import dataset as dataset_module
from mule_pattern_learner.temporal.live.checkpoint import restore_cuda_rng
from mule_pattern_learner.temporal.live.config_schema import validate_config
from mule_pattern_learner.temporal.live.contract import (
    DEFAULT_GROUPS,
    RELATIONS,
    ContextKey,
    FeaturePlan,
    SamplerPlan,
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.temporal.live.dataset import preparation_view
from mule_pattern_learner.temporal.live.evaluation import evaluate_final_population
from mule_pattern_learner.temporal.live.experiments import feature_experiments
from mule_pattern_learner.temporal.live.hubs import HUB_COLUMNS, HubRegistry, warn_hub_stubs
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.sampling import (
    BatchPrefetcher,
    PUSample,
    epoch_schedule,
    evaluation_indices,
    pu_batches,
    step_seed,
)
from mule_pattern_learner.temporal.live.source import rejection_summary
from mule_pattern_learner.temporal.live.supervision import align_observed_labels
from mule_pattern_learner.temporal.loss import NonNegativePULoss

ROOT = Path(__file__).resolve().parents[2]
DATES = {"train": ["2024-07-01"], "validation": ["2024-10-01"], "test": ["2025-01-01"]}
CUTOFFS = {"2024-07-01": 10_000, "2024-10-01": 20_000, "2025-01-01": 30_000}
HUB = "P3"


# nnPU ------------------------------------------------------------------------------


def reference_nnpu(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    prior: float,
    positive_weight: float,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The former NonNegativePULoss.forward, with its host-side branch.

    l_pos is 1 - sigmoid(f), as the loss now computes it (the float32 gradient fix).
    """
    positive = (targets == 1).to(logits.dtype)
    unlabeled = (targets == 0).to(logits.dtype)
    n_positive = torch.clamp(positive.sum(), min=1.0)
    n_unlabeled = torch.clamp(unlabeled.sum(), min=1.0)
    l_pos, l_neg = 1 - torch.sigmoid(logits), torch.sigmoid(logits)
    positive_risk = positive_weight * torch.sum(positive * l_pos) / n_positive
    negative_risk = (
        torch.sum(unlabeled * l_neg) / n_unlabeled
        - prior * torch.sum(positive * l_neg) / n_positive
    )
    objective = positive_risk + negative_risk
    if negative_risk.item() < -beta:
        return gamma * (-negative_risk), objective
    return objective, objective


def loss_cases() -> list[dict[str, Any]]:
    cases = []
    generator = torch.Generator().manual_seed(0)
    for index in range(48):
        n = int(torch.randint(2, 40, (1,), generator=generator))
        scale = (0.5, 3.0, 12.0)[index % 3]
        logits = torch.randn(n, generator=generator, dtype=torch.float64) * scale
        targets = (torch.rand(n, generator=generator) < 0.3).long()
        if index % 7 == 0:
            targets[:] = 0  # no positives in the batch
        if index % 5 == 0:
            # Confident separation drives the negative risk below zero.
            logits = torch.where(targets == 1, logits.abs() + 8, -logits.abs() - 8)
        cases.append(
            {
                "logits": logits,
                "targets": targets,
                "prior": (0.001, 0.05, 0.3, 0.5)[index % 4],
                "positive_weight": (None, 0.1, 0.5)[index % 3],
                "beta": (0.0, 0.0, 0.25)[index % 3],
                "gamma": (1.0, 1.0, 0.5, 2.0)[index % 4],
            }
        )
    return cases


def _loss_outputs(
    module: torch.nn.Module, case: dict[str, Any], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits = case["logits"].to(dtype).clone().requires_grad_(True)
    train_loss, objective = module(logits, case["targets"])
    train_loss.backward()
    return train_loss.detach(), objective.detach(), logits.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nnpu_values_and_gradients_equal_the_former_branch(dtype: torch.dtype) -> None:
    fired = set()
    for case in loss_cases():
        # positive_weight None exercises the constructor's default (the prior).
        weight = case["positive_weight"] or case["prior"]
        new = NonNegativePULoss(
            case["prior"],
            beta=case["beta"],
            gamma=case["gamma"],
            positive_weight=case["positive_weight"],
        )
        logits = case["logits"].to(dtype).clone().requires_grad_(True)
        old_train, old_objective = reference_nnpu(
            logits,
            case["targets"],
            prior=case["prior"],
            positive_weight=weight,
            beta=case["beta"],
            gamma=case["gamma"],
        )
        old_train.backward()
        train_loss, objective, gradient = _loss_outputs(new, case, dtype)
        fired.add(bool(old_train.detach() != old_objective.detach()))
        torch.testing.assert_close(train_loss, old_train.detach(), rtol=0, atol=0)
        torch.testing.assert_close(objective, old_objective.detach(), rtol=0, atol=0)
        torch.testing.assert_close(gradient, logits.grad, rtol=0, atol=0)
    assert fired == {True, False}, "Both the corrected and the plain branch must be covered"


def test_nnpu_forward_never_reads_a_value_on_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object) -> None:
        raise AssertionError("host synchronization")

    logits = torch.tensor([9.0, 9.0, -9.0, -9.0], requires_grad=True)
    targets = torch.tensor([1, 1, 0, 0])
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    train_loss, objective = NonNegativePULoss(0.5)(logits, targets)
    train_loss.backward()
    monkeypatch.undo()
    assert train_loss.item() > 0 > objective.item()


# Sampling, schedule and prefetch ----------------------------------------------------


def test_evaluation_sample_draws_unlabeled_rows_only_from_the_marginal() -> None:
    indices = np.arange(40)
    observed = np.zeros(40, dtype=bool)
    observed[[1, 5, 30]] = True
    marginal = np.ones(40, dtype=bool)
    marginal[[5, 30, 31, 32, 33]] = False  # label-selected pool rows
    chosen = evaluation_indices(indices, observed, limit=10, seed=3, marginal=marginal)
    assert {1, 5, 30} <= set(chosen)
    unlabeled = set(chosen) - {1, 5, 30}
    assert len(unlabeled) == 10 and not unlabeled & {31, 32, 33}
    everything = evaluation_indices(indices, observed, limit=None, seed=3, marginal=marginal)
    assert set(everything) == set(indices) - {31, 32, 33}
    # Without a marginal mask the former behaviour is unchanged.
    assert np.array_equal(evaluation_indices(indices, observed, limit=None, seed=3), indices)


def test_step_seeds_are_stable_distinct_and_63_bit() -> None:
    seeds = {step_seed(7, epoch, step) for epoch in range(3) for step in range(50)}
    assert len(seeds) == 150 and all(0 <= s < 2**63 for s in seeds)
    expected = hashlib.sha256(b"temporal_live_step:7:1:2").digest()
    assert step_seed(7, 1, 2) == int.from_bytes(expected[:8], "big") >> 1


def test_epoch_schedule_equals_lazy_pu_batches_and_leaves_the_same_generator_state() -> None:
    observed = np.zeros(100, dtype=bool)
    observed[:9] = True
    samples = [
        PUSample("a", np.arange(10, 60), observed, np.arange(0, 5)),
        PUSample("b", np.arange(60, 100), observed, np.arange(5, 9)),
    ]
    lazy_rng, eager_rng = np.random.default_rng(11), np.random.default_rng(11)
    lazy = [
        (s.date, p, m)
        for s in samples
        for p, m in pu_batches(s.marginal, s.observed, lazy_rng, 8, positive_indices=s.positives)
    ]
    steps = epoch_schedule(samples, eager_rng, 8, epoch=2, seed=5)
    assert len(steps) == len(lazy)
    for index, (step, (date, positives, marginal)) in enumerate(zip(steps, lazy, strict=True)):
        assert (step.step, step.date, step.seed) == (index, date, step_seed(5, 2, index))
        assert np.array_equal(step.indices, np.r_[positives, marginal])
    assert lazy_rng.bit_generator.state == eager_rng.bit_generator.state


def _prefetch_threads() -> int:
    return sum(t.name.startswith("temporal-batch") for t in threading.enumerate())


def _wait_for_no_prefetch_threads(timeout: float = 5.0) -> int:
    """Workers abandoned after an error finish on their own; wait for them."""
    deadline = time.monotonic() + timeout
    while _prefetch_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    return _prefetch_threads()


def test_prefetcher_preserves_order_and_bounds_work_in_flight() -> None:
    lock, active, peak, pulled = threading.Lock(), [0], [0], []

    def items():
        for value in range(30):
            pulled.append(value)
            yield value

    def build(value: int) -> int:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep((value * 7 % 5) / 1000)
        with lock:
            active[0] -= 1
        return value * value

    results = []
    with BatchPrefetcher(build, items(), depth=3) as batches:
        for value in batches:
            # Lookahead never exceeds the prefetch depth.
            assert len(pulled) <= len(results) + 1 + 3
            results.append(value)
    assert results == [v * v for v in range(30)]
    assert peak[0] <= 3
    assert _prefetch_threads() == 0


def test_prefetcher_raises_at_the_failing_item_and_shuts_down() -> None:
    def build(value: int) -> int:
        if value == 4:
            raise KeyError("boom")
        return value

    seen = []
    with pytest.raises(KeyError, match="boom"):
        with BatchPrefetcher(build, range(20), depth=2) as batches:
            for value in batches:
                seen.append(value)
    assert seen == [0, 1, 2, 3]
    assert _wait_for_no_prefetch_threads() == 0


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_prefetcher_error_does_not_wait_for_running_builds(error: type[BaseException]) -> None:
    release = threading.Event()

    def build(value: int) -> int:
        if value:
            release.wait(30)  # an in-flight REST call with retries
        return value

    started = time.perf_counter()
    try:
        with pytest.raises(error):
            with BatchPrefetcher(build, range(10), depth=3) as batches:
                for _ in batches:
                    raise error("consumer failure")
        assert time.perf_counter() - started < 5
        assert _prefetch_threads() > 0, "running builds are abandoned, not joined"
    finally:
        release.set()
    assert _wait_for_no_prefetch_threads() == 0


def test_prefetcher_build_error_does_not_wait_for_other_running_builds() -> None:
    release = threading.Event()

    def build(value: int) -> int:
        if value == 1:
            raise KeyError("boom")
        if value > 1:
            release.wait(30)
        return value

    started = time.perf_counter()
    try:
        with pytest.raises(KeyError, match="boom"):
            with BatchPrefetcher(build, range(10), depth=2) as batches:
                for _ in batches:
                    pass
        assert time.perf_counter() - started < 5
    finally:
        release.set()
    assert _wait_for_no_prefetch_threads() == 0


def test_prefetcher_cancels_queued_work_on_early_exit_and_runs_inline() -> None:
    built = []

    def build(value: int) -> int:
        built.append(value)
        time.sleep(0.01)
        return value

    with BatchPrefetcher(build, range(100), depth=2) as batches:
        for value in batches:
            if value == 1:
                break
    assert len(built) <= 5 and _prefetch_threads() == 0
    inline = list(BatchPrefetcher(lambda v: (v, threading.current_thread()), range(3), depth=0))
    assert inline == [(v, threading.current_thread()) for v in range(3)]
    with pytest.raises(ValueError):
        BatchPrefetcher(build, range(1), depth=9)


# Determinism runtime -----------------------------------------------------------------


def test_torch_runtime_applies_modes_and_restores_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    cuda = torch.device("cuda")  # a flag test only; CUDA is never initialized
    with torch_runtime(cuda, deterministic=True, threads=1):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_num_threads() == 1
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    with torch_runtime(cuda, deterministic="strict"):
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    with torch_runtime(torch.device("cpu"), deterministic=True):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(RuntimeError), torch_runtime(torch.device("cpu"), deterministic=False):
        assert not torch.are_deterministic_algorithms_enabled()
        raise RuntimeError
    after = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_num_threads(),
    )
    assert after == before
    with pytest.raises(ValueError):
        with torch_runtime(torch.device("cpu"), deterministic="sometimes"):
            pass


def test_cli_sets_the_cublas_workspace_at_import_and_keeps_user_values() -> None:
    code = (
        "import os, sys\n"
        "import mule_pattern_learner.temporal.live.cli\n"
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ":4096:8"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "CUBLAS_WORKSPACE_CONFIG": ":16:8"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ":16:8"
    # The assignment precedes every import that can load torch.
    source = (ROOT / "src/mule_pattern_learner/temporal/live/cli.py").read_text()
    assert source.index("CUBLAS_WORKSPACE_CONFIG") < source.index("\nimport argparse")


# Fake prepared dataset and context source --------------------------------------------


def _hash(*parts: object) -> int:
    value = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "big")


def _message(parent: ContextKey, j: int) -> dict[str, Any]:
    h = _hash(parent.node_type, parent.node_id, parent.cutoff_seq, j)
    relation = RELATIONS[h % 4]
    seq = parent.cutoff_seq - 1 - 5 * j - h % 5
    ts = parent.cutoff_ms - (j + 1) * 3_600_000 - h % 997
    gap = j % 3 != 0
    return {
        "node_type": "Account",
        "node_id": f"P{h % 11}",
        "relation": relation,
        "rail": "zelle" if relation.startswith("zelle") else "ach",
        "channel": "digital",
        "stratum": ("recent", "older", "distinct")[j % 3],
        "event_id": f"E{parent.node_id}.{seq}",
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": float(h % 500),
        "amount_present": True,
        "age_ms": parent.cutoff_ms - ts,
        "gap_ms": (h % 50) * 60_000 if gap else 0,
        "gap_present": gap,
        "pair_count_1h": 0,
        "pair_count_1d": 1,
        "pair_count_7d": 2,
        "pair_prior_count": h % 4,
        "pair_first_age_seconds": float(h % 10_000),
        "pair_first_present": True,
        "flow_delay_seconds": 0.0,
        "flow_present": False,
        "flow_censored": True,
        "flow_observation_seconds": 600.0,
        "flow_amount_ratio": 0.0,
        "flow_ratio_present": False,
        "flow_same_rail": False,
        "device_age_seconds": 0.0,
        "device_present": False,
        "ip_age_seconds": 0.0,
        "ip_present": False,
        "peer_first_ms": 1_000,
        "peer_external": h % 5 == 0,
        "peer_deposit": h % 5 != 0,
    }


def _association(parent: ContextKey, j: int) -> dict[str, Any]:
    template = _message(parent, 0)
    zero = {k: 0 for k, v in template.items() if isinstance(v, (int, float))}
    return {
        **zero,
        "node_type": "Party",
        "node_id": f"Q{_hash(parent.node_id, j) % 5}",
        "relation": "Account_Owned_By_Party",
        "rail": "unknown",
        "channel": "unknown",
        "stratum": "association",
        "event_id": "",
        "event_seq": parent.cutoff_seq,
        "event_ts_ms": parent.cutoff_ms,
        "amount_present": False,
        "gap_present": False,
        "pair_first_present": False,
        "flow_present": False,
        "flow_censored": False,
        "flow_ratio_present": False,
        "flow_same_rail": False,
        "device_present": False,
        "ip_present": False,
        "peer_external": False,
        "peer_deposit": False,
        "peer_first_ms": 1_000,
    }


def fake_context(key: ContextKey) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if key.node_type == "Account":
        count = 3 + _hash(key.node_id, key.cutoff_seq) % 6
        messages = [_message(key, j) for j in range(count) if key.cutoff_seq - 5 * j > 10]
        messages += [_association(key, j) for j in range(_hash(key.node_id) % 2)]
    return {
        **asdict(key),
        "status": "ok",
        "features": {"is_external": 0.0, "is_deposit": 1.0},
        "messages": messages,
    }


class FakeSource:
    """Thread-safe in-memory ContextSource with optional rejections and failures."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        reject: frozenset[str] = frozenset(),
        fail: Callable[[list[ContextKey], int, Counter[str]], bool] | None = None,
    ) -> None:
        self.plan = extraction_plan(config)
        self.sampler = SamplerPlan.from_config(config)
        self.reject, self.fail = reject, fail
        self.query_calls = 0
        self.rejections: Counter[str] = Counter()
        self.rejections_by_hop: dict[int, Counter[str]] = {}
        self.calls: Counter[str] = Counter()
        self.lock = threading.Lock()
        self.closed = False

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        with self.lock:
            phases = {k.visibility_phase for k in keys}
            for phase in phases:
                self.calls[f"hop{hop}_phase{phase}"] += 1
            if self.fail is not None and self.fail(keys, hop, self.calls):
                raise RuntimeError("injected source failure")
            self.query_calls += 1
            rows: list[dict[str, Any] | None] = []
            for key in keys:
                if key.node_id in self.reject:
                    self.rejections["missing_entity"] += 1
                    self.rejections_by_hop.setdefault(hop, Counter())["missing_entity"] += 1
                    rows.append(None)
                else:
                    rows.append(fake_context(key))
            return rows

    def close(self) -> None:
        self.closed = True


def base_config(**overrides: Any) -> dict[str, Any]:
    value = {
        "dataset_id": "unit_runtime",
        "evaluation_protocol": "strict_inductive",
        "scope_id": "unit_scope",
        "label_policy": "observed",
        "context_storage": "stream",
        "dates": deepcopy(DATES),
        "feature_groups": list(DEFAULT_GROUPS),
        "architecture": "split",
        "fanouts": [4, 2],
        "hidden": 16,
        "heads": 2,
        "dropout": 0.2,
        "epochs": 2,
        "steps_per_epoch": 3,
        "patience": 5,
        "batch_size": 8,
        "learning_rate": 0.01,
        "class_prior": 0.05,
        "positive_weight": 0.5,
        "seed": 7,
        "split_seed": 7,
        "device": "cpu",
        "threads": 1,
        "evaluation_unlabeled_limit": 12,
        "log_every_steps": 2,
        "prefetch_batches": 2,
        "sampler": {
            "policy": "resample",
            "recent": 3,
            "older": 1,
            "distinct": 1,
            "associations": 1,
            "relation_fanouts": [2, 2],
        },
    }
    value.update(overrides)
    return validate_config(value)


def accounts_frame() -> pd.DataFrame:
    rows = []
    for i in range(72):
        split = ("train", "validation", "test")[i % 3]
        rows.append(
            {
                "account_id": f"A{i:03}",
                "first_seen_seq": 5,
                "first_seen_ts_ms": timestamp("2024-01-01"),
                "group_id": f"G{i:03}",
                "observed_positive": False,
                "known_from_ms": 0,
                "split": split,
                # Label-selected pool rows sit outside the marginal reservoir.
                "in_marginal": i % 11 != 0,
            }
        )
    return pd.DataFrame(rows)


def labels_frame(accounts: pd.DataFrame) -> pd.DataFrame:
    chosen = accounts[accounts.index % 5 == 0]
    return pd.DataFrame(
        {
            "account_id": chosen.account_id,
            "known_positive": True,
            "known_from_ms": timestamp("2024-03-01"),
        }
    )


def hub_registry() -> HubRegistry:
    """The scoped registry: one hub at the training cutoff in phase 1 (training batches)."""
    row = {
        "account_id": HUB,
        "cutoff_seq": CUTOFFS["2024-07-01"],
        "visibility_phase": 1,
        "max_visible": 5000,
        "max_degree": 5000,
        "reason": "visible_history",
    }
    frame = pd.DataFrame([[row[name] for name in HUB_COLUMNS]], columns=list(HUB_COLUMNS))
    return HubRegistry(frame, cutoff_seqs=CUTOFFS.values(), threshold=2048, scope_id="unit_scope")


def prepared_dataset(
    path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    """A prepared-dataset directory; load_prepared is replaced by its in-memory copy."""
    path.mkdir(parents=True, exist_ok=True)
    accounts = accounts_frame()
    align_observed_labels(accounts, labels_frame(accounts)).to_parquet(
        path / "observed_labels.parquet", index=False
    )
    manifest = {
        "status": "ready",
        "config": config,
        "cutoff_seqs": dict(CUTOFFS),
        "cohort": "bounded_internal_deposit_seeds",
        "observed_labels_sha256": digest(path / "observed_labels.parquet"),
        "source": {"context_storage": "stream", "preparation": preparation_view(config)},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))

    def load(dataset: Path) -> tuple[dict[str, Any], pd.DataFrame]:
        assert dataset == path
        return deepcopy(manifest), accounts.copy()

    for module in (training, inference, dataset_module):
        monkeypatch.setattr(module, "load_prepared", load)
    return path, manifest, accounts


def fit(
    tmp_path: Path,
    name: str,
    config: dict[str, Any],
    *,
    source: FakeSource | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    return training.train(
        config,
        tmp_path / "dataset",
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


# Training end to end ------------------------------------------------------------------


def test_two_epochs_equal_one_epoch_plus_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
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


def test_mid_epoch_step_checkpoint_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(checkpoint_every_steps=1, steps_per_epoch=4)
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    fit(tmp_path, "straight", config)
    with pytest.raises(RuntimeError, match="injected"):
        fit(tmp_path, "resumed", config, source=FakeSource(config, fail=after_validation(3)))
    state = torch.load(tmp_path / "resumed_run/checkpoint_last.pt", weights_only=True)
    assert state["epoch"] == 1 and 1 <= state["step"] < 4
    # Runtime-only settings may change on resume; results must not.
    resumed = fit(
        tmp_path, "resumed", {**config, "prefetch_batches": 0, "log_every_steps": 1}, resume=True
    )
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
    fit(tmp_path, "inline", {**config, "prefetch_batches": 0})
    fit(tmp_path, "threads", {**config, "prefetch_batches": 4})
    assert_same_run(tmp_path, "inline", "threads")


def test_resume_refuses_a_changed_result_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.raises(RuntimeError):
        fit(tmp_path, "run", config, source=FakeSource(config, fail=after_validation(1)))
    with pytest.raises(ValueError, match="learning_rate"):
        fit(tmp_path, "run", {**config, "learning_rate": 0.02}, resume=True)


def test_batches_use_train_mode_step_seeds_and_the_hub_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    calls: list[tuple[str, int, int, int]] = []
    backends: set[str | None] = set()
    resolved: list[threading.Thread] = []
    lock = threading.Lock()
    real = batching.make_live_batch
    real_resolve = training.resolve_backend

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

    monkeypatch.setattr(batching, "make_live_batch", record)
    monkeypatch.setattr(training, "resolve_backend", resolve)
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
    config = base_config(evaluation_unlabeled_limit=100, max_rejected_root_fraction=0.2)
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
    config = base_config(evaluation_unlabeled_limit=100)
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    # Default limit 0: one rejected unlabeled validation root fails epoch 1.
    with pytest.raises(ValueError, match=r"validation: TigerGraph rejected 1 of 23 roots"):
        fit(tmp_path, "one", config, source=FakeSource(config, reject=frozenset({"A001"})))
    assert not (tmp_path / "one.pt").exists()
    # A rejected observed positive always fails, whatever the limit.
    loose = base_config(evaluation_unlabeled_limit=100, max_rejected_root_fraction=1.0)
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
    config = base_config(evaluation_unlabeled_limit=100)
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.raises(ValueError, match=r"test: TigerGraph rejected 1 of 22 roots"):
        fit(tmp_path, "run", config, source=FakeSource(config, reject=frozenset({"A002"})))
    assert (tmp_path / "run.pt").exists() and not (tmp_path / "run_run/metrics.json").exists()
    # The limit is a runtime key: raising it lets the run finish from its checkpoint.
    result = fit(
        tmp_path,
        "run",
        {**config, "max_rejected_root_fraction": 0.1},
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
    real = training.evaluate

    def no_ap(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {**real(*args, **kwargs), "average_precision": None}

    monkeypatch.setattr(training, "evaluate", no_ap)
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


def test_patience_zero_disables_early_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(epochs=3, patience=0, steps_per_epoch=1)
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    ap = iter([0.5, 0.4, 0.3])  # validation never improves after epoch 1
    real = training.evaluate

    def falling(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = real(*args, **kwargs)
        if len(args) > 2 and args[2] == 0.5:  # the per-epoch validation proxy
            result["average_precision"] = next(ap, result["average_precision"])
        return result

    monkeypatch.setattr(training, "evaluate", falling)
    result = fit(tmp_path, "run", config)
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3] and result["best_epoch"] == 1


def test_cuda_rng_restore_tolerates_a_different_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[Any] = []

    def set_rng_state(state: torch.Tensor, device: int | torch.device = 0) -> None:
        restored.append(device)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng_state)
    states = [torch.zeros(16, dtype=torch.uint8) for _ in range(8)]
    # A checkpoint written with one state per GPU of an 8-GPU node, resumed on one GPU.
    restore_cuda_rng(states, torch.device("cuda"))
    assert restored == [0]
    restored.clear()
    restore_cuda_rng(states[0], torch.device("cuda", 0))
    assert restored == [torch.device("cuda", 0)]
    restore_cuda_rng(states, torch.device("cpu"))
    restore_cuda_rng(None, torch.device("cuda"))
    assert len(restored) == 1
    assert "get_rng_state_all" not in inspect.getsource(training)


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
    explicit = base_config(sampler={**config["sampler"], "backend": "torch"})
    result = fit(tmp_path, "run", explicit, resume=True)
    assert result["status"] == "complete" and result["sampler_backend"] == "torch"
    assert_same_run(tmp_path, "straight", "run")


def test_missing_hub_indicator_warns_once_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups = [g for g in DEFAULT_GROUPS if g != "hub_indicator"]
    config = base_config(feature_groups=groups, epochs=1)
    plan = FeaturePlan.from_config(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_hub_stubs(HubRegistry.empty(), plan)
        warn_hub_stubs(hub_registry(), FeaturePlan.from_config(base_config()))
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    with pytest.warns(UserWarning, match="no hub_indicator group") as caught:
        fit(tmp_path, "run", config)
    assert sum("hub_indicator" in str(w.message) for w in caught) == 1


def test_run_directory_is_created_only_after_the_source_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)

    def refuse(*_: object) -> None:
        raise ValueError("Live graph counts changed")

    monkeypatch.setattr(training, "open_context_source", refuse)
    with pytest.raises(ValueError, match="counts changed"):
        training.train(config, tmp_path / "dataset", tmp_path / "m.pt", hubs=hub_registry())
    assert not (tmp_path / "m_run").exists() and not (tmp_path / "m.pt").exists()
    # A source built with another sampler is rejected before anything is written.
    other = base_config(sampler={**config["sampler"], "relation_fanouts": [3, 2]})
    with pytest.raises(ValueError, match="sampler"):
        fit(tmp_path, "m", config, source=FakeSource(other))
    assert not (tmp_path / "m_run").exists()


def test_non_finite_loss_stops_before_any_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(checkpoint_every_steps=1)
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
    config = base_config(epochs=1)
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
    config = base_config(device="mps")
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    result = fit(tmp_path, "mps", config)
    assert result["device"] == "mps" and result["status"] == "complete"
    assert all(v.device.type == "cpu" for v in saved_model(tmp_path / "mps.pt").values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fake_end_to_end_training_on_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = base_config(device="cuda", deterministic="strict")
    prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    assert fit(tmp_path, "cuda", config)["status"] == "complete"


# Scoring ------------------------------------------------------------------------------


def checkpoint(path: Path, config: dict[str, Any], manifest_path: Path | None = None) -> Path:
    plan = FeaturePlan.from_config(config)
    torch.manual_seed(0)
    model = LiveTGAT(config["hidden"], config["heads"], 0.0, plan=plan)
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "contract": contract_fingerprint(),
        "basis_id": BASIS_ID,
        "threshold": 0.5,
        "input_fingerprint": plan.fingerprint(),
        "selected_on": "validation_observed_label_proxy_ap",
    }
    if manifest_path is not None:
        payload["dataset_manifest_sha256"] = digest(manifest_path)
        payload["dataset"] = str(manifest_path.parent)
    torch.save(payload, path)
    return path


class ScoringExecutor:
    def __init__(self, accounts: pd.DataFrame | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.accounts = accounts

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params))
        if name == "temporal_training_cutoffs":
            return [{"status": "ok", "last_visible_seqs": {str(params["cutoff_times"][0]): 29_999}}]
        if name == "temporal_hub_registry":
            (cutoff,) = params["cutoff_seqs"]
            # Score-new is unscoped: phase-3 rows over all visible history.
            assert not params.get("scope_id")
            echo = {k: params[k] for k in ("threshold", "cutoff_seqs", "scope_id") if k in params}
            return [
                {
                    "status": "ok",
                    **echo,
                    "hubs": [
                        {
                            "account_id": HUB,
                            "cutoff_seq": cutoff,
                            "visibility_phase": 3,
                            "max_visible": 9000,
                            "max_degree": 9000,
                            "reason": "visible_history",
                        }
                    ],
                }
            ]
        if name == "temporal_scope_population":
            assert params["include_observed"] is False and self.accounts is not None
            page = [
                {"account_id": a, "partition": 3, "first_seen_ts_ms": 1}
                for a in self.accounts.account_id
                if a > params["after_id"]
            ]
            return [{"status": "ok", "accounts": page}]
        raise AssertionError(name)


def test_score_new_writes_only_ok_rows_and_lists_rejected_ids(tmp_path: Path) -> None:
    config = base_config()
    model = checkpoint(tmp_path / "model.pt", config)
    executor = ScoringExecutor()
    source = FakeSource(config, reject=frozenset({"ghost_1", "ghost_2"}))
    ids = [f"new_{i}" for i in range(14)]
    ids[1:1], ids[9:9] = ["ghost_1"], ["ghost_2"]
    rejected_ids = ["ghost_1", "ghost_2"]
    output = tmp_path / "scores.parquet"
    result = predictor.score_new_accounts(
        model, iter(ids), "2025-01-01", output, executor=executor, contexts=source
    )
    frame = pd.read_parquet(output)
    assert frame.account_id.tolist() == [v for v in ids if v not in rejected_ids]
    assert frame.score.between(0, 1).all() and (frame.date == "2025-01-01").all()
    rejected_file = tmp_path / "scores.parquet.rejected.txt"
    assert rejected_file.read_text().split() == rejected_ids
    assert result["accounts"] == len(frame) and result["rejected"] == len(rejected_ids)
    assert result["rejected_roots_by_status"] == {"missing_entity": len(rejected_ids)}
    assert result["rejected_children"] == 0 and result["rejected_children_by_status"] == {}
    assert result["rejection_events_by_status"] == {"missing_entity": len(rejected_ids)}
    hub_call = next(p for name, p in executor.calls if name == "temporal_hub_registry")
    assert hub_call["cutoff_seqs"] == [30_000] and hub_call["threshold"] == 2048
    assert "scan_cap" not in hub_call
    assert source.closed and not (tmp_path / "scores.parquet.pending").exists()


def test_score_new_reports_root_and_child_rejections_separately(tmp_path: Path) -> None:
    config = base_config()
    model = checkpoint(tmp_path / "model.pt", config)
    # P5 and P7 are peers (children) of the scored accounts, never roots.
    source = FakeSource(config, reject=frozenset({"ghost", "P5", "P7"}))
    ids = ["ghost", *(f"new_{i}" for i in range(12))]
    result = predictor.score_new_accounts(
        model,
        iter(ids),
        "2025-01-01",
        tmp_path / "scores.parquet",
        executor=ScoringExecutor(),
        contexts=source,
    )
    assert result["accounts"] == 12 and result["rejected"] == 1
    assert result["rejected_roots_by_status"] == {"missing_entity": 1}
    children = result["rejected_children_by_status"]["missing_entity"]
    assert result["rejected_children"] > 0 and children > 0
    assert result["rejection_events_by_status"] == {"missing_entity": 1 + children}
    # Without per-hop counts the root statuses are unknown, never the mixed counter.
    del source.rejections_by_hop
    plain = rejection_summary(source, 1, Counter({"rejected_children": 3}))
    assert plain["rejected_roots_by_status"] is None and plain["rejected_children"] == 3


def test_inference_score_uses_the_dataset_hub_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config()
    dataset, _, _ = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = checkpoint(tmp_path / "model.pt", config, dataset / "manifest.json")
    loaded = []

    def registry(path: Path, manifest: dict[str, Any]) -> HubRegistry:
        loaded.append(path)
        return hub_registry()

    monkeypatch.setattr(inference, "load_hub_registry", registry)
    source = FakeSource(config, reject=frozenset({"A002"}))
    output = tmp_path / "test.parquet"
    result = inference.score(model, dataset, "2025-01-01", "test", output, contexts=source)
    frame = pd.read_parquet(output)
    assert loaded == [dataset]
    assert len(frame) == 23 and "A002" not in set(frame.account_id)
    assert result["rejected"] == 1 and (tmp_path / "test.parquet.rejected.txt").exists()


def test_final_population_audit_scores_through_the_dataset_clock_and_hubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_config(max_rejected_root_fraction=0.1)
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = checkpoint(tmp_path / "model.pt", config, dataset / "manifest.json")
    test_accounts = accounts[accounts.split == "test"]
    truth = test_accounts[["account_id"]].assign(is_mule=(np.arange(len(test_accounts)) % 4 == 0))
    truth["is_mule"] = truth.is_mule.astype(int)
    source = FakeSource(config, reject=frozenset({test_accounts.account_id.iloc[1]}))
    seen: list[int] = []
    real = batching.make_live_batch

    def record(store: Any, roots: list[ContextKey], **kwargs: Any) -> dict[str, torch.Tensor]:
        seen.extend(k.cutoff_seq for k in roots)
        return real(store, roots, **kwargs)

    monkeypatch.setattr(batching, "make_live_batch", record)

    class Truth:
        def read(self) -> pd.DataFrame:
            return truth

    result = evaluate_final_population(
        model,
        Truth(),
        tmp_path / "final.json",
        executor=ScoringExecutor(test_accounts),
        contexts=source,
        hubs=hub_registry(),
    )
    assert set(seen) == {CUTOFFS["2025-01-01"]}
    assert result["rejected_accounts"] == 1 and result["rejected_negatives"] == 1
    assert result["rejected"] == 1 and result["rejected_roots_by_status"] == {"missing_entity": 1}
    assert result["metrics"]["sample_accounts"] == len(test_accounts) - 1
    assert result["metrics"]["evaluation_cohort"].endswith("_minus_rejected_negatives")
    assert (tmp_path / "final.rejected.txt").read_text().split() == [
        test_accounts.account_id.iloc[1]
    ]


@pytest.mark.parametrize(("limit", "rejected_index"), [(1.0, 0), (0.0, 1)])
def test_final_population_audit_fails_on_censored_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: float, rejected_index: int
) -> None:
    config = base_config(max_rejected_root_fraction=limit)
    dataset, _, accounts = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    model = checkpoint(tmp_path / "model.pt", config, dataset / "manifest.json")
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
        evaluate_final_population(
            model,
            Truth(),
            tmp_path / "final.json",
            executor=ScoringExecutor(test_accounts),
            contexts=source,
            hubs=hub_registry(),
        )
    assert not list(tmp_path.glob("final*"))


# CLI and experiment matrix ------------------------------------------------------------


def test_cli_needs_no_config_truth_or_dataset() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train"])
    assert args.config is None and args.dataset is None
    assert parser.parse_args(["prepare"]).config is None
    final = parser.parse_args(["evaluate-final", "--checkpoint", "m.pt", "--output", "o.json"])
    assert final.dataset is None and final.truth is None
    assert isinstance(cli.truth_source(None), cli.GraphEvaluationTruth)
    assert isinstance(cli.truth_source(Path("t.parquet")), cli.ParquetEvaluationTruth)
    scoring = parser.parse_args(
        ["score", "--checkpoint", "m.pt", "--date", "2025-01-01", "--output", "s.parquet"]
    )
    assert scoring.dataset is None


def test_cli_install_passes_force_and_optional(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def install(executor: object, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"installed": []}

    monkeypatch.setattr(cli, "install", install)
    monkeypatch.setattr(cli, "TigerGraphExecutor", lambda: object())
    for argv, expected in (
        (["install"], {"include_optional": False, "force": False}),
        (["install", "--force", "--include-optional"], {"include_optional": True, "force": True}),
    ):
        monkeypatch.setattr(sys, "argv", ["mule-temporal", *argv])
        cli.main()
        assert calls[-1] == expected
    assert capsys.readouterr().out.count('"installed": []') == 2


def test_train_command_prepares_then_trains_or_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Like run_config, without a dataset_id: preparation resolves it.
    config = {k: v for k, v in base_config().items() if k != "dataset_id"}
    prepared, trained = [], []

    def prepare(c: dict[str, Any], path: Path) -> dict[str, Any]:
        prepared.append((c, path))
        return {"source": {"dataset_id": "derived"}}

    # `mule-temporal train` is pipeline.run with resume: patch the pipeline's steps.
    monkeypatch.setattr(pipeline, "run_config", lambda path: dict(config))
    monkeypatch.setattr(pipeline, "prepare_live", prepare)
    monkeypatch.setattr(
        pipeline, "train", lambda c, d, o, *, resume: trained.append((c, d, o, resume)) or {}
    )
    output = tmp_path / "model.pt"
    cli.train_command(cli.build_parser().parse_args(["train", "--output", str(output)]))
    # One command prepares into the run directory, then trains (resuming if interrupted).
    assert prepared[-1][1] == tmp_path / "model_run" / "prepared"
    c, d, o, resume = trained[-1]
    assert c["dataset_id"] == "derived" and d == prepared[-1][1] and o == output and resume


def test_feature_arms_keep_the_base_extraction() -> None:
    groups = sorted(
        {
            *DEFAULT_GROUPS,
            "event_channel",
            "decayed_activity",
            "history_support",
            "identity_order",
            "device_ip_context",
            "rolling_windows",
            "amount_ratios",
            "recency",
            "association_counts",
            "entity_age",
            "pair_window_counts",
        }
    )
    base = base_config(extraction_groups=groups)
    arms = feature_experiments(base)
    assert all(arm["extraction_groups"] == groups for arm in arms.values())
    with pytest.raises(ValueError, match="window_free_device_ip"):
        feature_experiments(
            base_config(extraction_groups=[g for g in groups if g != "device_ip_context"])
        )


def test_nnpu_objective_resolves_the_named_positive_weights() -> None:
    for weight, resolved, name in (
        ("prior", 0.001, "nnPU"),
        ("balanced", 0.999, "imbalanced_nnPU"),
        (0.5, 0.5, "positive_reweighted_nnPU"),
    ):
        prior, value = training.nnpu_objective({"class_prior": 0.001, "positive_weight": weight})
        assert (prior, value) == (0.001, pytest.approx(resolved))
        assert training.objective_name(prior, value) == name
