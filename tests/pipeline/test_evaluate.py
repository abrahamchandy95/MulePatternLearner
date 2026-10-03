"""The audits of a run: validation and test, on one connection opened after the checks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.artifacts import AUDIT_COLUMNS, read_audit_scores, read_events
from mule_pattern_learner.config import RunConfig, TransportConfig
from mule_pattern_learner.contract.feature_groups import extraction_plan
from mule_pattern_learner.contract.graph_schema import PHASE_SPLIT, ContextKey
from mule_pattern_learner.contract.server import CONTEXT_QUERY, TRUTH_QUERY
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.evaluation.audit import AUDIT_SPLITS
from mule_pattern_learner.evaluation.truth import ParquetTruthReader
from mule_pattern_learner.paths import DatasetPaths, RunPaths
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.reporting.run_report import AUDIT_FIGURES
from mule_pattern_learner.runtime.progress import emit, recording
from mule_pattern_learner.testing.builders import (
    RUNTIME_CHANGES,
    UNIT_SOURCE,
    ground_truth_rows,
    prepared_dataset,
    recorded_events,
    saved_model,
    scope_population,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import (
    PREPARED_ACCOUNTS,
    FakeTigerGraph,
    prepared_graph,
)


def test_evaluate_run_connects_after_its_checks_and_reads_truth_once_on_that_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config(RUNTIME_CHANGES)
    dataset, _, _ = prepared_dataset(tmp_path / "dataset", config, monkeypatch)
    run = RunPaths(tmp_path / "run")
    saved_model(run.model, config, dataset)
    connected: list[TransportConfig] = []
    verified: list[Any] = []
    audits: list[dict[str, Any]] = []
    reads: list[int] = []

    class Truth:
        def read(self) -> pd.DataFrame:
            reads.append(1)
            return pd.DataFrame()

    executor = SimpleNamespace()

    def connect(transport: TransportConfig) -> Any:
        connected.append(transport)
        return executor

    def audit(inputs: Any, split: str, **options: Any) -> dict[str, Any]:
        emit({"event": "audit", "split": split})
        audits.append({"inputs": inputs, "split": split, **options})
        return {"split": split}

    def verify(executor: Any, manifest: dict[str, Any]) -> None:
        verified.append(executor)

    def registry(dataset: DatasetPaths, manifest: dict[str, Any]) -> Any:
        return "hubs"

    monkeypatch.setattr(pipeline_evaluate, "connect", connect)
    monkeypatch.setattr(pipeline_evaluate, "verify_frozen_source", verify)
    monkeypatch.setattr(pipeline_evaluate, "audit", audit)
    # The prepared fixture has no hub registry file; the audit's inputs load this one.
    monkeypatch.setattr("mule_pattern_learner.evaluation.audit.load_hub_registry", registry)
    # A model of another dataset is refused before anything connects.
    other = RunPaths(tmp_path / "other")
    saved_model(other.model, config)
    with pytest.raises(ValueError, match="needs the prepared dataset"):
        pipeline_evaluate.evaluate_run(other, data=tmp_path)
    assert connected == []
    # The dataset is the model's own: its dataset id's directory in data.
    result = pipeline_evaluate.evaluate_run(run, truth=Truth(), data=tmp_path)
    assert result == {"validation": {"split": "validation"}, "test": {"split": "test"}}
    # The model's retry budgets, the frozen source checked, the truth read once.
    assert connected == [config.transport] and verified == [executor] and reads == [1]
    validation, test = audits
    assert (validation["split"], test["split"]) == ("validation", "test")
    assert validation["inputs"].dataset == dataset and validation["inputs"].hubs == "hubs"
    contexts = validation["contexts"]
    assert test["contexts"] is contexts and test["truth"] is validation["truth"]
    assert validation["scope"].executor is executor and contexts.fetcher.executor is executor
    # The source requests the model's inputs with its pools, reads and writes the
    # dataset's context cache, and is closed afterwards.
    assert (contexts.plan, contexts.sampler) == (
        extraction_plan(config.feature_plan()),
        config.sampler,
    )
    assert contexts.disk.cache.directory == dataset.contexts
    with pytest.raises(RuntimeError, match="closed"):
        contexts.fetch([ContextKey("Account", "A000", 1, 1)])
    # The lines the audits print go to the run's events.jsonl.
    assert recorded_events(run.events) == [
        {"event": "audit", "split": "validation"},
        {"event": "audit", "split": "test"},
    ]
    # Without a truth reader, the graph's oracle is read on the same connection.
    oracles: list[Any] = []

    class Oracle(Truth):
        def __init__(self, on: Any) -> None:
            oracles.append(on)

    monkeypatch.setattr(pipeline_evaluate, "TigerGraphTruthReader", Oracle)
    fresh = RunPaths(tmp_path / "fresh")
    saved_model(fresh.model, config, dataset)
    pipeline_evaluate.evaluate_run(fresh, data=tmp_path)
    assert oracles == [executor] and reads == [1, 1]


# The accounts of the fake scope.
POPULATION = PREPARED_ACCOUNTS


def audited_graph(
    tmp_path: Path, config: RunConfig, statuses: dict[ContextKey | str, str] | None = None
) -> tuple[FakeTigerGraph, Path]:
    """A fake graph with ground truth, and config's dataset prepared from it in data.

    ``statuses`` are the graph's per-request statuses (FakeTigerGraph).
    """
    data = tmp_path / "data"
    graph, _ = prepared_graph(data, config, statuses=statuses)
    return graph, data


def connecting(graph: FakeTigerGraph | None) -> Callable[[TransportConfig], FakeTigerGraph]:
    """A connect() that reaches the graph; without one, connecting fails the test."""

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        if graph is None:
            pytest.fail("connected")
        return graph

    return connect


def test_evaluate_run_audits_validation_and_test_on_the_fake_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config()
    graph, data = audited_graph(tmp_path, config)
    monkeypatch.setattr(pipeline_evaluate, "connect", connecting(graph))
    dataset = DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    runs = [RunPaths(tmp_path / name) for name in ("first", "second")]
    for run, shift in zip(runs, (0.0, 2.0), strict=True):
        saved_model(run.model, config, dataset, logit_shift=shift)
    reports = [pipeline_evaluate.evaluate_run(runs[0], data=data)]
    requested = graph.names().count(CONTEXT_QUERY)
    reports.append(pipeline_evaluate.evaluate_run(runs[1], data=data))
    # Truth is read once per run, for both splits. Both runs audit the same accounts, so
    # the second reads every context from the dataset's cache.
    assert graph.names().count(TRUTH_QUERY) == 2
    assert graph.names().count(CONTEXT_QUERY) == requested > 0
    population = pd.DataFrame(scope_population(POPULATION))
    truth = pd.DataFrame(ground_truth_rows(scope_population(POPULATION))).set_index("account_id")
    for split in ("validation", "test"):
        members = population[population.partition.map(PHASE_SPLIT) == split]
        mules = set(members.account_id[truth.loc[members.account_id].is_mule.eq(1).to_numpy()])
        samples = [read_audit_scores(run.audit_scores(split)) for run in runs]
        for sample, report in zip(samples, (r[split] for r in reports), strict=True):
            assert tuple(sample.columns) == AUDIT_COLUMNS
            # Every mule of the split, and every other account of this small population.
            assert set(sample.account_id[sample.is_mule == 1]) == mules
            assert set(sample.account_id) == set(members.account_id)
            assert report["population_accounts"] == len(members) and report["split"] == split
            revealed = members.set_index("account_id").observed_positive
            assert sample.revealed.tolist() == revealed.loc[sample.account_id].tolist()
            expected = truth.loc[sample.account_id]
            assert sample.ring_id.tolist() == expected.mule_ring_id.tolist()
            assert sample.label_source.tolist() == expected.mule_label_source.tolist()
            assert report["revealed_positives"] + report["hidden_positives"] == len(mules)
        # Both runs are scored on the same accounts, whatever their scores.
        assert samples[0].account_id.tolist() == samples[1].account_id.tolist()
        assert not samples[0].score.equals(samples[1].score)
    # A run with both audits is reported as it is, without connecting, and its figures
    # and report.md are left as they were.
    drawn = [*runs[0].plots.glob("*.png"), runs[0].report]
    assert len(drawn) == 1 + len(AUDIT_FIGURES)
    written = {path: path.stat().st_mtime_ns for path in drawn}
    monkeypatch.setattr(pipeline_evaluate, "connect", connecting(None))
    noted = tmp_path / "events.jsonl"
    with recording(noted):
        assert pipeline_evaluate.evaluate_run(runs[0], data=data) == reports[0]
    assert {path: path.stat().st_mtime_ns for path in drawn} == written
    # It says that it read the reports, which the command's summary then shows.
    assert recorded_events(noted) == [
        {
            "event": "already_audited",
            "run": str(runs[0].root),
            "reports": [str(runs[0].audit_report(split)) for split in ("validation", "test")],
        }
    ]
    # An interrupted evaluation audits only the split it lacks, and says which it read.
    runs[0].audit_report("test").unlink()
    monkeypatch.setattr(pipeline_evaluate, "connect", connecting(graph))
    with recording(noted):
        assert pipeline_evaluate.evaluate_run(runs[0], data=data) == reports[0]
    assert read_events(noted)[-1]["reports"] == [str(runs[0].audit_report("validation"))]
    audited = [event["split"] for event in read_events(runs[0].events) if event["event"] == "audit"]
    assert audited == ["validation", "test", "test"]
    # A truth reader other than the graph's (a parquet file of the same columns).
    table = pd.DataFrame(
        {
            "account_id": truth.index,
            "is_mule": truth.is_mule.to_numpy(),
            "ring_id": truth.mule_ring_id.to_numpy(),
            "label_source": truth.mule_label_source.to_numpy(),
        }
    )
    table.to_parquet(tmp_path / "truth.parquet", index=False)
    third = RunPaths(tmp_path / "third")
    saved_model(third.model, config, dataset)
    parquet = pipeline_evaluate.evaluate_run(
        third, truth=ParquetTruthReader(tmp_path / "truth.parquet"), data=data
    )
    assert parquet["test"]["metrics"] == reports[0]["test"]["metrics"]


def test_a_session_audits_several_runs_on_one_connection_and_reads_truth_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config()
    graph, data = audited_graph(tmp_path, config)
    connected: list[TransportConfig] = []

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        connected.append(transport)
        return graph

    monkeypatch.setattr(pipeline_connect, "connect", connect)
    monkeypatch.setattr(pipeline_evaluate, "connect", connecting(None))
    dataset = DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    runs = [RunPaths(tmp_path / name) for name in ("first", "second")]
    for run, shift in zip(runs, (0.0, 2.0), strict=True):
        saved_model(run.model, config, dataset, logit_shift=shift)
    session = pipeline_connect.Session(config.transport)
    truth = pipeline_evaluate.SharedTruth(session)
    for run in runs:
        pipeline_evaluate.evaluate_run(run, truth=truth, data=data, session=session)
    assert connected == [config.transport] and graph.names().count(TRUTH_QUERY) == 1
    assert all(run.audit_report(split).exists() for run in runs for split in AUDIT_SPLITS)
    # Runs that are audited already need neither the connection nor the truth.
    idle = pipeline_connect.Session(config.transport)
    pipeline_evaluate.evaluate_run(
        runs[0], truth=pipeline_evaluate.SharedTruth(idle), data=data, session=idle
    )
    assert not idle.connected and connected == [config.transport]


def test_each_split_reports_the_rejections_of_its_own_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unit_config()
    # A child over its history capacity, which both splits' contexts reach.
    graph, data = audited_graph(tmp_path, config, {"N5": "history_capacity_exceeded"})
    monkeypatch.setattr(pipeline_evaluate, "connect", connecting(graph))
    dataset = DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    both, alone = RunPaths(tmp_path / "both"), RunPaths(tmp_path / "alone")
    for run in (both, alone):
        saved_model(run.model, config, dataset)
    reports = pipeline_evaluate.evaluate_run(both, data=data)
    # The same model audits test alone once its validation audit exists.
    alone.audit_report("validation").parent.mkdir(parents=True)
    shutil.copy(both.audit_scores("validation"), alone.audit_scores("validation"))
    shutil.copy(both.audit_report("validation"), alone.audit_report("validation"))
    test = pipeline_evaluate.evaluate_run(alone, data=data)["test"]
    # Both splits share a context source, yet the test report counts only its own audit's.
    assert test == reports["test"]
    for report in reports.values():
        children = report["rejected_children_by_status"]
        assert report["rejected_children"] > 0 and set(children) == {"history_capacity_exceeded"}
        assert report["rejection_events_by_status"] == children
        assert report["rejected"] == 0 and report["rejected_roots_by_status"] == {}
