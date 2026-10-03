"""Golden run: the built-in training profile on the fakes, pinned to recorded numbers.

The run prepares a strict-inductive dataset from the fake scope, with labels revealed in
the graph as the built-in run reads them, then trains 2 epochs of 4 steps on the CPU
with the built-in feature groups, slot sum, balanced positive weight and weight
average. The literals at the end record its first training batch, every step's loss
and objective, each epoch's validation AP, the selected epoch, the threshold picked on
validation, the test AP, the sha256 of the rendered context query and the rows of the
prepared accounts and observed labels. A refactoring leaves every one of them
unchanged. A commit that changes one on purpose says so and pastes the new values,
which the failing assertion prints as Python source.

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

`mule check` prints the same digests and first loss for the parity check on the graph
(pipeline.check.first_step); a test pins it to these literals.

The golden run itself has no context cache. Another test trains it twice more with the
dataset's disk tier (data.context_cache): cold, then warm from the entries the cold run
wrote, and both give the same literals.

A last test audits the golden run's validation and test splits against the builders'
ground truth (testing.builders.ground_truth_rows) and pins GOLDEN_AUDIT: each split's
sample, point estimates and ring-clustered intervals. They depend only on how the scores
rank, so they hold on both machines as the APs do.
"""

from __future__ import annotations

from collections.abc import Generator
import contextlib
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import threading
from typing import Any
from unittest.mock import patch

import pandas as pd

from mule_pattern_learner.artifacts import read_epochs, read_history, read_json
from mule_pattern_learner.batching.assemble import RootBatch, tensor_digests
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.server import CONTEXT_QUERY
from mule_pattern_learner.data.context_cache import ContextCache
from mule_pattern_learner.data.contexts import ContextSource, build_context_source
from mule_pattern_learner.data.manifest import read_manifest
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.evaluation.audit import audit, audit_inputs
from mule_pattern_learner.paths import DatasetPaths, RunPaths
from mule_pattern_learner.pipeline.check import first_step
from mule_pattern_learner.testing.builders import (
    ground_truth_rows,
    neighbourhood,
    scope_population,
)
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader
from mule_pattern_learner.tigergraph.oracle import TigerGraphTruthReader
from mule_pattern_learner.tigergraph.render import render_context_query
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader
from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.schedule import step_seed

RELATIVE = 1e-5
# The built-in run (DEFAULT_CONFIG) with a smaller dataset, batch and run. Dropout is the
# one modelling change (see the module docstring); log_every_steps = 1 logs every step.
GOLDEN_CHANGES: dict[str, Any] = {
    "dataset": {"seed_limits": {"train": 64, "validation": 24, "test": 24}},
    "model": {"dropout": 0.0},
    "training": {"epochs": 2, "steps_per_epoch": 4, "batch_size": 16},
    "runtime": {"device": "cpu", "threads": 1, "log_every_steps": 1},
}
# The source id of the fake graph's data.
GOLDEN_SOURCE = "golden_fixture"
# Accounts in the fake scope, and its three split cutoffs (FakeTigerGraph.last_visible).
POPULATION = 200
CUTOFF_SEQS = (101, 102, 103)


def golden_config() -> RunConfig:
    return DEFAULT_CONFIG.with_changes(GOLDEN_CHANGES)


def golden_executor() -> FakeTigerGraph:
    """The fake graph: fixed neighbourhoods, one hub and one child over its history capacity."""
    return FakeTigerGraph(
        factory=neighbourhood,
        hubs=[("N3", cutoff) for cutoff in CUTOFF_SEQS],
        statuses={"N5": "history_capacity_exceeded"},
        population=scope_population(POPULATION),
    )


def golden_source(executor: FakeTigerGraph, config: RunConfig) -> ContextSource:
    """The source open_context_source builds for a streamed preparation."""
    return build_context_source(
        TigerGraphContextFetcher(executor),
        extraction_plan(config.feature_plan()),
        config.sampler,
        config.transport,
    )


def prepare_golden(directory: Path) -> tuple[RunConfig, DatasetPaths, FakeTigerGraph]:
    """Prepare the golden dataset with the built-in label source (graph_observed)."""
    config = golden_config()
    executor = golden_executor()
    dataset = DatasetPaths(directory / "dataset")
    prepare(
        config,
        GOLDEN_SOURCE,
        dataset,
        {"Account": POPULATION},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(executor),
        cutoffs=TigerGraphCutoffReader(executor),
        hub_reader=TigerGraphHubReader(executor),
    )
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
def first_training_batch(config: RunConfig) -> Generator[list[RootBatch]]:
    """Record the batch of epoch 1, step 1, which a prefetch thread may build."""
    seed = step_seed(config.training.seed, 0, 0)
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
    run = RunPaths(directory / "run")
    with first_training_batch(config) as found:
        result = trainer.train(config, dataset, run, contexts=golden_source(executor, config))
    (first,) = found
    assert first.batch is not None
    history = read_history(run.history)
    columns = ["epoch", "step", "loss", "objective", "corrected_steps"]
    return Observed(
        batch=tensor_digests(first.batch),
        batch_stats={k: v for k, v in first.stats.items() if k != "sampler_backend"},
        steps=[tuple(row) for row in history[columns].to_numpy(object).tolist()],
        epoch_ap=read_epochs(run.epochs).validation_ap.tolist(),
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


@contextlib.contextmanager
def through_context_cache(directory: Path) -> Generator[list[ContextSource]]:
    """Train the golden run with its source given the dataset's disk tier: the cache on.

    The harness's source (golden_source) is closed and replaced by the same source with
    the dataset's context cache, kept in directory so that every run shares it. The
    sources are listed in the order the runs train.
    """
    real = trainer.train
    sources: list[ContextSource] = []

    def train(
        config: RunConfig, dataset: DatasetPaths, run: RunPaths, *, contexts: ContextSource
    ) -> dict[str, Any]:
        contexts.close()
        cache = replace(ContextCache.of(dataset, read_manifest(dataset)), directory=directory)
        cached = build_context_source(
            contexts.fetcher, contexts.plan, contexts.sampler, config.transport, cache
        )
        sources.append(cached)
        return real(config, dataset, run, contexts=cached)

    with patch.object(trainer, "train", train):
        yield sources


def test_the_golden_run_is_the_same_with_the_context_cache_cold_and_warm(
    tmp_path: Path,
) -> None:
    with through_context_cache(tmp_path / "contexts") as sources:
        runs = {name: golden_run(tmp_path / name) for name in ("cold", "warm")}
    for name, observed in runs.items():
        problems = run_differences(observed)
        assert not problems, "\n".join([name, *problems, "", "Observed:", literals(observed)])
    cold, warm = sources
    # The cold run requested its contexts and wrote them, and read from disk only those
    # it asked for again after the LRU had dropped them. The warm one read every context
    # memory did not hold from disk, and requested none from the graph.
    counts = cold.counts
    assert cold.database_calls > 0
    assert counts.memory_hits + counts.disk_hits <= counts.requested - counts.distinct
    assert warm.database_calls == 0 and warm.counts.requested == cold.counts.requested
    fetcher = warm.fetcher
    assert isinstance(fetcher, TigerGraphContextFetcher)
    assert isinstance(fetcher.executor, FakeTigerGraph)
    assert CONTEXT_QUERY not in fetcher.executor.names()
    run = RunPaths(tmp_path / "warm" / "run")
    assert read_history(run.history).database_calls.max() == 0
    contexts = read_json(run.metrics)["contexts"]
    assert contexts["disk_hits"] == contexts["requested"] - contexts["memory_hits"] > 0
    assert contexts["disk_hit_rate"] == 1.0


def frame_digest(frame: pd.DataFrame) -> str:
    """The sha256 of a frame's rows as CSV: its values, whatever the parquet writer."""
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def test_the_golden_settings_select_the_recorded_accounts(tmp_path: Path) -> None:
    # The dataset's accounts and observed labels feed every other literal. They were
    # recorded with the code of commit 08b487e, the last before the layered restructure,
    # which prepared the same rows.
    _, dataset, _ = prepare_golden(tmp_path)
    files = {"accounts": dataset.accounts, "observed_labels": dataset.observed_labels}
    for name, (rows, digest) in GOLDEN_DATASET.items():
        frame = pd.read_parquet(files[name])
        assert (len(frame), frame_digest(frame)) == (rows, digest), name


def test_mule_check_reports_the_golden_first_batch_and_loss(tmp_path: Path) -> None:
    config, dataset, executor = prepare_golden(tmp_path)
    with golden_source(executor, config) as contexts:
        report = first_step(config, dataset, contexts)
    assert digest_differences(report["tensor_digests"], GOLDEN_BATCH) == []
    assert {k: v for k, v in report["batch"].items() if k != "sampler_backend"} == (
        GOLDEN_BATCH_STATS
    )
    _, _, loss, objective, _ = GOLDEN_STEPS[0]
    assert close(report["loss"], loss) and close(report["objective"], objective, abs(loss))


# The numbers of an audit that GOLDEN_AUDIT records for each split.
AUDIT_METRICS = (
    "sample_accounts",
    "sample_positives",
    "estimated_population",
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    *(f"{kind}_at_{pct}pct" for pct in (1, 5, 10) for kind in ("precision", "recall")),
)


def audit_numbers(report: dict[str, Any]) -> dict[str, Any]:
    """What GOLDEN_AUDIT records of a split's audit report."""
    return {
        "metrics": {name: report["metrics"][name] for name in AUDIT_METRICS},
        "intervals": report["intervals"],
        "positives": (report["revealed_positives"], report["hidden_positives"]),
    }


def audit_differences(observed: dict[str, dict[str, Any]]) -> list[str]:
    """Audit numbers that differ from GOLDEN_AUDIT: integers exactly, floats within RELATIVE."""
    problems = []
    for split, want in GOLDEN_AUDIT.items():
        have = observed[split]
        if have["positives"] != want["positives"]:
            problems.append(f"{split}: revealed and hidden positives {have['positives']}")
        pairs = [(name, have["metrics"][name], value) for name, value in want["metrics"].items()]
        for name, bounds in want["intervals"].items():
            got = have["intervals"][name]
            pairs += [(f"{name} interval", a, b) for a, b in zip(got, bounds, strict=True)]
        for name, got, value in pairs:
            same = got == value if isinstance(value, int) else close(got, value, 1.0)
            if not same:
                problems.append(f"{split}: {name} {got}")
    return problems


def test_the_golden_run_audits_validation_and_test(tmp_path: Path) -> None:
    golden_run(tmp_path)
    config, executor = golden_config(), golden_executor()
    oracle = FakeTigerGraph(truth=ground_truth_rows(scope_population(POPULATION)))
    truth = TigerGraphTruthReader(oracle).read()
    inputs = audit_inputs(RunPaths(tmp_path / "run"), dataset=DatasetPaths(tmp_path / "dataset"))
    observed = {}
    with golden_source(executor, config) as contexts:
        for split in GOLDEN_AUDIT:
            scope = TigerGraphScopeReader(executor)
            report = audit(inputs, split, truth=truth, scope=scope, contexts=contexts)
            observed[split] = audit_numbers(report)
    problems = audit_differences(observed)
    assert not problems, "\n".join([*problems, "", f"GOLDEN_AUDIT = {observed!r}"])


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
GOLDEN_QUERY_SHA256 = "21f15c377763febab0e169563284d98eab9958492bf7f68053efb4cb7552f6d2"
# Rows and frame_digest of the prepared accounts.parquet and observed_labels.parquet.
GOLDEN_DATASET = {
    "accounts": (123, "bdfe8b31ed8e0561638d9459c87b9e4dddc2a0cf09d3aad971a25fa2a0650062"),
    "observed_labels": (123, "ceda44c0fbb4040d251dc1832061724f9b4a335006b6e94f1ec0c4ddabe0f118"),
}
# The audits of the golden run (test_the_golden_run_audits_validation_and_test). Each split's
# population is 40 accounts, all sampled; its intervals are ring-clustered 90% intervals.
GOLDEN_AUDIT: dict[str, dict[str, Any]] = {
    "validation": {
        "metrics": {
            "sample_accounts": 40,
            "sample_positives": 11,
            "estimated_population": 40.0,
            "average_precision": 0.35639207813120855,
            "roc_auc": 0.6520376175548589,
            "precision": 0.375,
            "recall": 0.8181818181818182,
            "f1": 0.5142857142857142,
            "precision_at_1pct": 0.0,
            "recall_at_1pct": 0.0,
            "precision_at_5pct": 0.0,
            "recall_at_5pct": 0.0,
            "precision_at_10pct": 0.25,
            "recall_at_10pct": 0.09090909090909091,
        },
        "intervals": {
            "average_precision": [0.27460380830646947, 0.542421933214359],
            "roc_auc": [0.49843260188087773, 0.7836990595611285],
            "precision_at_1pct": [0.0, 0.0],
            "recall_at_1pct": [0.0, 0.0],
            "precision_at_5pct": [0.0, 0.5],
            "recall_at_5pct": [0.0, 0.09090909090909091],
            "precision_at_10pct": [0.0, 0.5121951219512195],
            "recall_at_10pct": [0.0, 0.19000000000000003],
        },
        "positives": (5, 6),
    },
    "test": {
        "metrics": {
            "sample_accounts": 40,
            "sample_positives": 12,
            "estimated_population": 40.0,
            "average_precision": 0.34625358588206884,
            "roc_auc": 0.4285714285714286,
            "precision": 0.2916666666666667,
            "recall": 0.5833333333333334,
            "f1": 0.38888888888888895,
            "precision_at_1pct": 1.0,
            "recall_at_1pct": 0.03333333333333333,
            "precision_at_5pct": 0.5,
            "recall_at_5pct": 0.08333333333333333,
            "precision_at_10pct": 0.25,
            "recall_at_10pct": 0.08333333333333333,
        },
        "intervals": {
            "average_precision": [0.2492364271417604, 0.5337862670172485],
            "roc_auc": [0.2618589743589743, 0.6321915584415584],
            "precision_at_1pct": [0.0, 1.0],
            "recall_at_1pct": [0.0, 0.038],
            "precision_at_5pct": [0.0, 1.0],
            "recall_at_5pct": [0.0, 0.17727272727272728],
            "precision_at_10pct": [0.0, 0.75],
            "recall_at_10pct": [0.0, 0.25],
        },
        "positives": (6, 6),
    },
}
