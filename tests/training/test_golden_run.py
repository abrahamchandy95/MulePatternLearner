"""Golden run: the built-in training profile on the fakes, pinned to recorded numbers.

The run prepares a strict-inductive cohort from the fake scope, with labels revealed in
the graph as the built-in run reads them, then trains 2 epochs of 4 steps on the CPU
with the built-in feature groups, slot sum, balanced positive weight and weight
average. The literals at the end record its first training batch, every step's loss
and objective, each epoch's validation AP, the selected epoch, the threshold picked on
validation, the test AP and the sha256 of the rendered context query. A refactoring
leaves every one of them unchanged. A commit that changes one on purpose says so and
pastes the new values, which the failing assertion prints as Python source.

The literals hold on macOS arm64 and on Linux x86_64:
- Integer and boolean tensors come from integer arithmetic and hash-seeded draws, so
  their sha256 must match exactly.
- Floating tensors, losses and APs match within a relative tolerance of 1e-5, because
  each machine's math libraries round sin, cos, log and matrix products differently.
  Floating tensors are compared through the summaries of `tensor_digests`.
- Dropout is off. On Intel processors torch draws CPU dropout masks with MKL's
  generator and elsewhere with its own, so the masks, and every number after them,
  would differ between hosts.
- In the recorded run every validation and test score lies at least 5e-5 (relative)
  from the next one, far above that rounding, so the APs and the selected epoch do not
  flip between hosts.

`benchmark_batch.py` prints the same digests and first loss for the live parity
check; the second test pins it to these literals.
"""

from __future__ import annotations

from collections.abc import Generator
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any
from unittest.mock import patch

import pytest

from mule_pattern_learner.batching.assemble import RootBatch, tensor_digests
from mule_pattern_learner.config import run_config, validate_config
from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.data.contexts import StreamingContextSource, streaming_source
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.testing.builders import neighbourhood, scope_population
from mule_pattern_learner.testing.fake_graph import FakeExecutor
from mule_pattern_learner.tigergraph.render import render_context_query
from mule_pattern_learner.tigergraph.labels import GraphObservedLabels
from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.schedule import step_seed

RELATIVE = 1e-5
# The built-in run (run_config) with a smaller cohort, batch and run. Dropout is the one
# modelling change (see the module docstring); log_every_steps = 1 logs every step.
GOLDEN_CHANGES: dict[str, Any] = {
    "dataset_id": "golden_fixture",
    "seed_limits": {"train": 64, "validation": 24, "test": 24},
    "epochs": 2,
    "steps_per_epoch": 4,
    "batch_size": 16,
    "dropout": 0.0,
    "device": "cpu",
    "threads": 1,
    "log_every_steps": 1,
}
# Accounts in the fake scope, and its three split cutoffs (FakeExecutor.last_visible).
POPULATION = 200
CUTOFF_SEQS = (101, 102, 103)


def golden_config() -> dict[str, Any]:
    return validate_config({**run_config(), **GOLDEN_CHANGES})


def golden_executor() -> FakeExecutor:
    """The fake graph: v5 neighbourhoods, one hub and one child over its history capacity."""
    return FakeExecutor(
        factory=neighbourhood,
        hubs=[("N3", cutoff) for cutoff in CUTOFF_SEQS],
        statuses={"N5": "history_capacity_exceeded"},
        population=scope_population(POPULATION),
    )


def golden_source(executor: FakeExecutor, config: dict[str, Any]) -> StreamingContextSource:
    """The source open_context_source builds for a streamed preparation."""
    return streaming_source(
        executor, extraction_plan(config), SamplerPlan.from_config(config), config
    )


def prepare_golden(directory: Path) -> tuple[dict[str, Any], Path, FakeExecutor]:
    """Prepare the golden cohort with the built-in label source (graph_observed)."""
    config = golden_config()
    executor = golden_executor()
    dataset = directory / "dataset"
    prepare(config, dataset, executor, {"Account": POPULATION}, GraphObservedLabels())
    return config, dataset, executor


@dataclass
class Observed:
    batch: dict[str, dict[str, Any]]
    batch_stats: dict[str, Any]
    steps: list[tuple[int, int, float, float, int]]
    epoch_ap: list[float]
    selected_epoch: int
    validation_ap: float
    threshold: float
    test_ap: float
    query_sha256: str


@contextlib.contextmanager
def first_training_batch(config: dict[str, Any]) -> Generator[list[RootBatch]]:
    """Record the batch of epoch 1, step 1, which a prefetch thread may build."""
    seed = step_seed(int(config["seed"]), 0, 0)
    real = trainer.build_root_batch
    found: list[RootBatch] = []
    lock = threading.Lock()

    def build(*args: Any, **kwargs: Any) -> RootBatch:
        batch = real(*args, **kwargs)
        if kwargs["mode"] == "train" and kwargs["step_seed"] == seed:
            with lock:
                found.append(batch)
        return batch

    with patch.object(trainer, "build_root_batch", build):
        yield found


def golden_run(directory: Path) -> Observed:
    config, dataset, executor = prepare_golden(directory)
    with first_training_batch(config) as found:
        result = trainer.train(
            config, dataset, directory / "model.pt", contexts=golden_source(executor, config)
        )
    (first,) = found
    assert first.batch is not None
    progress = (directory / "model_run/progress.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in progress]
    return Observed(
        batch=tensor_digests(first.batch),
        batch_stats={k: v for k, v in first.stats.items() if k != "sampler_backend"},
        steps=[
            (e["epoch"], e["step"], e["loss"], e["objective"], e["corrected_steps"])
            for e in events
            if e["event"] == "train"
        ],
        epoch_ap=[epoch["validation_proxy_ap"] for epoch in result["history"]],
        selected_epoch=result["best_epoch"],
        validation_ap=result["validation_proxy"]["average_precision"],
        threshold=result["validation_proxy"]["threshold"],
        test_ap=result["observed_label_proxy"]["test"]["average_precision"],
        query_sha256=hashlib.sha256(render_context_query().encode()).hexdigest(),
    )


def close(actual: float, expected: float, scale: float = 0.0) -> bool:
    """Equal within RELATIVE of the expected value, or of `scale` (a magnitude)."""
    return math.isclose(actual, expected, rel_tol=RELATIVE, abs_tol=RELATIVE * scale)


def digest_differences(
    actual: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> list[str]:
    """Tensors whose digests differ: exact sha256 for integer and boolean, else summaries.

    A floating sum may cancel to near zero, so it is compared within RELATIVE of the
    tensor's absolute sum, and an element within RELATIVE of the largest recorded one.
    """
    if set(actual) != set(expected):
        return [f"tensor names differ: {sorted(set(actual) ^ set(expected))}"]
    problems = []
    for name, want in expected.items():
        have = actual[name]
        if (have["dtype"], have["shape"]) != (want["dtype"], want["shape"]):
            problems.append(f"{name}: {have['dtype']} {have['shape']}")
        elif "sum" not in want:
            if have["sha256"] != want["sha256"]:
                problems.append(f"{name}: sha256 {have['sha256']}")
        elif set(have["elements"]) != set(want["elements"]):
            problems.append(f"{name}: element positions {sorted(have['elements'])}")
        else:
            scale = want["abs_sum"]
            peak = max(abs(v) for v in want["elements"].values())
            checks = {
                "sum": close(have["sum"], want["sum"], scale),
                "abs_sum": close(have["abs_sum"], want["abs_sum"]),
                "weighted_sum": close(have["weighted_sum"], want["weighted_sum"], 7 * scale),
            }
            checks |= {
                f"element {i}": close(have["elements"][i], value, peak)
                for i, value in want["elements"].items()
            }
            problems += [f"{name}: {label} {have}" for label, ok in checks.items() if not ok]
    return problems


def literals(observed: Observed) -> str:
    """A run's golden literals as Python source; floating tensors keep no sha256."""
    batch = {
        name: {k: v for k, v in digest.items() if k != "sha256" or "sum" not in digest}
        for name, digest in observed.batch.items()
    }
    return "\n".join(
        [
            f"GOLDEN_BATCH = {batch!r}",
            f"GOLDEN_BATCH_STATS = {observed.batch_stats!r}",
            f"GOLDEN_STEPS = {observed.steps!r}",
            f"GOLDEN_EPOCH_AP = {observed.epoch_ap!r}",
            f"GOLDEN_SELECTED_EPOCH = {observed.selected_epoch!r}",
            f"GOLDEN_VALIDATION_AP = {observed.validation_ap!r}",
            f"GOLDEN_THRESHOLD = {observed.threshold!r}",
            f"GOLDEN_TEST_AP = {observed.test_ap!r}",
            f"GOLDEN_QUERY_SHA256 = {observed.query_sha256!r}",
        ]
    )


def run_differences(observed: Observed) -> list[str]:
    problems = digest_differences(observed.batch, GOLDEN_BATCH)
    if observed.batch_stats != GOLDEN_BATCH_STATS:
        problems.append(f"first batch statistics {observed.batch_stats}")
    if [(e, s, c) for e, s, _, _, c in observed.steps] != [
        (e, s, c) for e, s, _, _, c in GOLDEN_STEPS
    ]:
        problems.append("logged steps or corrected-step counts")
    else:
        for (epoch, step, loss, risk, _), (*_, want_loss, want_risk, _) in zip(
            observed.steps, GOLDEN_STEPS, strict=True
        ):
            if not (close(loss, want_loss) and close(risk, want_risk, abs(want_loss))):
                problems.append(f"epoch {epoch} step {step}: loss {loss}, objective {risk}")
    if len(observed.epoch_ap) != len(GOLDEN_EPOCH_AP) or not all(
        close(have, want) for have, want in zip(observed.epoch_ap, GOLDEN_EPOCH_AP)
    ):
        problems.append(f"validation AP per epoch {observed.epoch_ap}")
    if observed.selected_epoch != GOLDEN_SELECTED_EPOCH:
        problems.append(f"selected epoch {observed.selected_epoch}")
    if not close(observed.validation_ap, GOLDEN_VALIDATION_AP):
        problems.append(f"validation AP {observed.validation_ap}")
    if not close(observed.threshold, GOLDEN_THRESHOLD):
        problems.append(f"threshold {observed.threshold}")
    if not close(observed.test_ap, GOLDEN_TEST_AP):
        problems.append(f"test AP {observed.test_ap}")
    if observed.query_sha256 != GOLDEN_QUERY_SHA256:
        problems.append(f"rendered context query sha256 {observed.query_sha256}")
    return problems


def test_built_in_run_reproduces_the_golden_numbers(tmp_path: Path) -> None:
    observed = golden_run(tmp_path)
    problems = run_differences(observed)
    assert not problems, "\n".join([*problems, "", "Observed:", literals(observed)])


def load_benchmark() -> Any:
    path = REPOSITORY_ROOT / "scripts/benchmark_batch.py"
    spec = importlib.util.spec_from_file_location("script_benchmark_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_the_golden_first_batch_and_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset, executor = prepare_golden(tmp_path)
    overrides = tmp_path / "golden.json"
    overrides.write_text(json.dumps(GOLDEN_CHANGES))
    module = load_benchmark()

    def open_source(path: Path, manifest: dict[str, Any], settings: dict[str, Any]) -> Any:
        assert path == dataset and settings == config
        return golden_source(executor, settings)

    monkeypatch.setattr(module, "open_context_source", open_source)
    report_path = tmp_path / "report.json"
    argv = ["benchmark_batch", "--config", str(overrides), "--dataset", str(dataset)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(report_path), "--train-step"])
    module.main()
    report = json.loads(report_path.read_text())
    assert digest_differences(report["tensor_digests"], GOLDEN_BATCH) == []
    assert {k: v for k, v in report["batch"].items() if k != "sampler_backend"} == (
        GOLDEN_BATCH_STATS
    )
    _, _, loss, objective, _ = GOLDEN_STEPS[0]
    assert close(report["loss"], loss) and close(report["objective"], objective, abs(loss))


# Golden literals, recorded on macOS arm64 (torch 2.12, CPU). Floating tensors keep no sha256.
GOLDEN_BATCH = {
    "first_channel": {
        "dtype": "int64",
        "shape": [16, 16],
        "sha256": "631bce7e17744e6042328642018387ad7ea8f15c41696609807938422d743b43",
    },
    "first_edge": {
        "dtype": "float32",
        "shape": [16, 16, 142],
        "sum": 6759.390104,
        "abs_sum": 10960.97978,
        "weighted_sum": 26962.18545,
        "elements": {"0": 2.944438934, "12117": -0.7533016801, "24234": 0.0, "36351": 0.0},
    },
    "first_mask": {
        "dtype": "bool",
        "shape": [16, 16],
        "sha256": "35c1eba48b3b820a2b81c3944faa9f6ed8fe4e41d532c56352b7515eee401785",
    },
    "first_rail": {
        "dtype": "int64",
        "shape": [16, 16],
        "sha256": "c549daa06909775f867444848a41297f46669c42b7f15481aa4b78a8d9737379",
    },
    "first_relation": {
        "dtype": "int64",
        "shape": [16, 16],
        "sha256": "54fcc45d700af1bdbfffa63e5420f9ec2e5cdd43160bfa3df73f084b26705f51",
    },
    "first_stratum": {
        "dtype": "int64",
        "shape": [16, 16],
        "sha256": "c1c01d1935f62b8c41d1a08181fa8a3644c60e59a4e55b82f2afeb275e28d124",
    },
    "neighbor_positions": {
        "dtype": "int64",
        "shape": [16, 16],
        "sha256": "f6c38848f05ef172be53aa60573700fc8bb162f07b500590ea657ac997b4c1e3",
    },
    "root_positions": {
        "dtype": "int64",
        "shape": [16],
        "sha256": "80cc42a62fa32b0eeb4adf1ee097e5b7898344ac437283690faf74e38b6c049f",
    },
    "second_channel": {
        "dtype": "int64",
        "shape": [59, 4],
        "sha256": "e0f67333cb960fffd19a018fdb0292e21643e50b62e5eb710cfe58624ddabec6",
    },
    "second_edge": {
        "dtype": "float32",
        "shape": [59, 4, 142],
        "sum": 12062.29732,
        "abs_sum": 19478.25679,
        "weighted_sum": 48204.02638,
        "elements": {"0": 3.135494232, "11170": 0.5604585409, "22341": -0.7533016801, "33511": 0.0},
    },
    "second_mask": {
        "dtype": "bool",
        "shape": [59, 4],
        "sha256": "6478ecec79cd81a88cd2629e13a869890acbae49902deb48c0016135e976d35b",
    },
    "second_rail": {
        "dtype": "int64",
        "shape": [59, 4],
        "sha256": "348e9097b0e023e06af14d4afe5903e3acf8259ae7317ac8a49386ce7a4050c1",
    },
    "second_relation": {
        "dtype": "int64",
        "shape": [59, 4],
        "sha256": "80891153bd481822469a6019dc465925ba647fa0f5bcfe5659df6adac858c309",
    },
    "second_stratum": {
        "dtype": "int64",
        "shape": [59, 4],
        "sha256": "7db7237b6d90804bfb6ed0835cec260800baf202d40becbeac137fb18fa0dafe",
    },
    "second_x": {
        "dtype": "float32",
        "shape": [59, 4, 9],
        "sum": 420.0,
        "abs_sum": 420.0,
        "weighted_sum": 1700.0,
        "elements": {"0": 1.0, "708": 0.0, "1416": 0.0, "2123": 0.0},
    },
    "x": {
        "dtype": "float32",
        "shape": [59, 24],
        "sum": 318.9795589,
        "abs_sum": 318.9795589,
        "weighted_sum": 1271.042767,
        "elements": {"0": 1.0, "472": 0.0, "944": 0.0, "1415": 0.0},
    },
}
GOLDEN_BATCH_STATS = {
    "rejected_roots": 0,
    "roots": 16,
    "contexts": 59,
    "stub_children": 7,
    "rejected_children": 7,
    "first_edges": 125,
    "second_edges": 196,
}
GOLDEN_STEPS = [
    (1, 1, 0.9973645210266113, 0.9973645210266113, 0),
    (1, 2, 1.0063931941986084, 1.0063931941986084, 0),
    (1, 3, 1.0002868175506592, 1.0002868175506592, 0),
    (1, 4, 0.9952985048294067, 0.9952985048294067, 0),
    (2, 1, 0.9950098991394043, 0.9950098991394043, 0),
    (2, 2, 0.9898823499679565, 0.9898823499679565, 0),
    (2, 3, 0.9771919846534729, 0.9771919846534729, 0),
    (2, 4, 0.9766665697097778, 0.9766665697097778, 0),
]
GOLDEN_EPOCH_AP = [0.47464285714285714, 0.47692307692307695]
GOLDEN_SELECTED_EPOCH = 2
GOLDEN_VALIDATION_AP = 0.47692307692307695
GOLDEN_THRESHOLD = 0.5102718956479596
GOLDEN_TEST_AP = 0.3216137566137566
GOLDEN_QUERY_SHA256 = "16647ae2e7f8728cc92fbe678b8e3be78158a3f8fe17f261e4e0a8a4d9f994a8"
