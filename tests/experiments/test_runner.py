"""A suite on FakeTigerGraph: two variants, two seeds, one epoch each, then kept or redone."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest

from mule_pattern_learner.artifacts import (
    keep_history,
    read_comparison,
    read_run_provenance,
    read_summary,
    write_run_config,
)
from mule_pattern_learner.config import DEFAULT_CONFIG, TransportConfig
from mule_pattern_learner.contract.server import CONTEXT_QUERY, TRUTH_QUERY
from mule_pattern_learner.experiments.runner import (
    ARCHIVE,
    KEEP,
    RESUME,
    TRAIN,
    PlannedRun,
    check_variants,
    is_outage,
    plan_run,
    run_suite,
    time_bound,
)
from mule_pattern_learner.experiments.tables import COMPLETE, FAILED, STOPPED, audited
from mule_pattern_learner.experiments.variants import BASELINE, VARIANTS, Variant, with_model
from mule_pattern_learner.paths import RunPaths, SuitePaths
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.pipeline import prepare as pipeline_prepare
from mule_pattern_learner.reporting.report import AUDIT_FIGURES, SUITE_FIGURES, TRAINING_FIGURES
from mule_pattern_learner.testing.builders import (
    ground_truth_rows,
    neighbourhood,
    scope_population,
    write_run_files,
)
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.executor import TigerGraphUnavailableError

# The built-in run, one small epoch of it: the suite's base run.
BASE = DEFAULT_CONFIG.with_changes(
    {
        "dataset": {"seed_limits": {"train": 64, "validation": 24, "test": 24}},
        "model": {"hidden": 16},
        "training": {"epochs": 1, "steps_per_epoch": 3, "batch_size": 16},
        "runtime": {"device": "cpu", "threads": 1},
    }
)
SEEDS = (42, 43)
# The variant besides the baseline: it requests contexts without their time encodings.
DROP = VARIANTS["drop_time_encoding"]
POPULATION = 200


def suite_graph(
    monkeypatch: pytest.MonkeyPatch, before: Any = None
) -> tuple[FakeTigerGraph, list[TransportConfig]]:
    """The fake graph behind the suite's session, and the connections the session opened.

    Its queries are installed, its scope ready and its known mules revealed, and it holds
    the ground truth. A use case that connects other than through the session fails.
    """
    population = scope_population(POPULATION)
    header = {"ready": True, "source_id": "suite_fixture", "split_seed": 42}
    graph = FakeTigerGraph(
        factory=neighbourhood,
        hubs=[("N3", cutoff) for cutoff in (101, 102, 103)],
        population=population,
        truth=ground_truth_rows(population),
        scopes={BASE.scope.id: header},
        before=before,
    )
    connections: list[TransportConfig] = []

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        connections.append(transport)
        return graph

    def elsewhere(transport: TransportConfig) -> None:
        pytest.fail("a suite connects through its session only")

    monkeypatch.setattr(pipeline_connect, "connect", connect)
    for module in (pipeline_prepare, pipeline_evaluate):
        monkeypatch.setattr(module, "connect", elsewhere)

    def revealed(*args: object) -> None:
        return None

    monkeypatch.setattr(pipeline_prepare, "ensure_revealed_labels", revealed)
    return graph, connections


def suite(results: Path, data: Path, names: tuple[str, ...] = (DROP.name,)) -> dict[str, Any]:
    return run_suite(names, base=BASE, seeds=SEEDS, results=results, data=data)


def events(capsys: pytest.CaptureFixture[str], name: str) -> list[dict[str, Any]]:
    """The lines the suite printed of one event."""
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    return [line for line in lines if line.get("event") == name]


def files(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_a_suite_trains_audits_and_compares_then_keeps_or_archives_what_it_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    graph, connections = suite_graph(monkeypatch)
    data, results = tmp_path / "data", tmp_path / "results"
    result = suite(results, data)
    # Seeds outer, variants inner; one connection and one read of truth for all of it.
    order = [(v, s) for s in SEEDS for v in (BASELINE.name, DROP.name)]
    assert [(r["variant"], r["seed"]) for r in result["runs"]] == order
    assert {(r["action"], r["status"]) for r in result["runs"]} == {(TRAIN, COMPLETE)}
    assert result["status"] == COMPLETE and connections == [BASE.transport]
    assert graph.names().count(TRUTH_QUERY) == 1
    (planned,) = events(capsys, "suite")
    assert planned["runs"] == {
        v: {str(s): TRAIN for s in SEEDS} for v in (BASELINE.name, DROP.name)
    }
    # The run files, with their figures and audits.
    runs = [RunPaths.of(variant, seed, results) for variant, seed in order]
    drawn = {f"plots/{name}.png" for name in (*TRAINING_FIGURES, *AUDIT_FIGURES)}
    for run in runs:
        assert audited(run) and run.model.exists() and run.report.exists()
        assert drawn <= set(files(run.root)), run.root
    # Both tables, the figures and report.md.
    compared = SuitePaths.of(DROP.name, results)
    summary = read_summary(compared.summary)
    assert set(zip(summary.variant, summary.seed, summary.status, strict=True)) == {
        (variant, seed, COMPLETE) for variant, seed in order
    }
    comparison = read_comparison(compared.comparison)
    assert comparison.variant.tolist() == [BASELINE.name, DROP.name]
    assert comparison.seeds.tolist() == ["42 43", "42 43"]
    # Every audit scored the same accounts, so all of them pair.
    assert comparison.unpaired_accounts.tolist() == [0, 0]
    assert not comparison.validation_ap_delta.iloc[1:].isna().any()
    assert sorted(p.stem for p in compared.plots.iterdir()) == sorted(SUITE_FIGURES)
    assert compared.report.read_text().startswith(f"# Suite {DROP.name}\n")
    # A second suite keeps every complete run and connects nowhere; the tables are
    # written again from the same files.
    kept = {run.root: files(run.root) for run in runs}
    requested = graph.names().count(CONTEXT_QUERY)
    again = suite(results, data)
    assert {(r["action"], r["status"]) for r in again["runs"]} == {(KEEP, COMPLETE)}
    assert connections == [BASE.transport] and graph.names().count(CONTEXT_QUERY) == requested
    assert {run.root: files(run.root) for run in runs} == kept
    assert read_summary(compared.summary).equals(summary)
    # A run whose settings differ moves to the archive, whole, and trains again.
    moved = RunPaths.of(DROP.name, 43, results)
    other = DROP.config(BASE, 43).with_changes({"training": {"patience": 1}})
    write_run_config(moved.config, other, read_run_provenance(moved.config))
    capsys.readouterr()
    redone = suite(results, data)
    assert [r["action"] for r in redone["runs"]] == [KEEP, KEEP, KEEP, ARCHIVE]
    assert redone["status"] == COMPLETE and audited(moved)
    (archived,) = events(capsys, "run_archived")
    assert archived["differs"] == ["training.patience"]
    kept_aside = RunPaths(Path(archived["archive"]))
    assert kept_aside.root.parent == results / "archive" / DROP.name / "seed-43"
    assert set(files(kept_aside.root)) == set(kept[moved.root])
    # A run of another dataset is archived too.
    assert plan_run(DROP, 43, BASE, results, "another").differs[-1].startswith("dataset ")


def test_a_variant_that_fails_on_its_own_fails_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def untimed(name: str, params: dict[str, Any]) -> None:
        if name == CONTEXT_QUERY and not params["include_time_encoding"]:
            raise ValueError("this variant's own failure")

    suite_graph(monkeypatch, untimed)
    results = tmp_path / "results"
    result = run_suite(
        (DROP.name,), base=BASE, seeds=(42,), results=results, data=tmp_path / "data"
    )
    outcomes = {r["variant"]: (r["status"], r["error"]) for r in result["runs"]}
    assert outcomes[BASELINE.name] == (COMPLETE, None)
    assert outcomes[DROP.name][0] == FAILED
    assert "train: ValueError: this variant's own failure" in outcomes[DROP.name][1]
    assert result["status"] == FAILED
    comparison = read_comparison(SuitePaths.of(DROP.name, results).comparison)
    assert comparison.seeds.tolist() == ["42", ""]


def test_an_outage_stops_the_suite_and_the_tables_say_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def outage(name: str, params: dict[str, Any]) -> None:
        if name == CONTEXT_QUERY and not params["include_time_encoding"]:
            raise TigerGraphUnavailableError("fetch_training_context failed: unavailable")

    graph, _ = suite_graph(monkeypatch, outage)
    data, results = tmp_path / "data", tmp_path / "results"
    result = suite(results, data)
    assert result["status"] == STOPPED and "unavailable" in result["stopped_by"]
    # The baseline of seed 42 trained; nothing ran after the outage, not even an audit.
    assert {r["status"] for r in result["runs"]} == {STOPPED}
    assert RunPaths.of(BASELINE.name, 42, results).metrics.exists()
    assert not RunPaths.of(BASELINE.name, 43, results).root.exists()
    assert graph.names().count(TRUTH_QUERY) == 0
    summary = read_summary(SuitePaths.of(DROP.name, results).summary)
    assert set(summary.status) == {STOPPED}
    # Once TigerGraph is back, the interrupted run resumes.
    dataset = result["dataset_id"]
    assert plan_run(DROP, 42, BASE, results, dataset).action == RESUME
    assert plan_run(BASELINE, 42, BASE, results, dataset).action == KEEP


def test_variants_are_refused_offline_with_their_names() -> None:
    check_variants([BASELINE, DROP], BASE, SEEDS)
    same = Variant("same", "The baseline again", lambda c: with_model(c, dropout=c.model.dropout))
    with pytest.raises(ValueError, match="baseline and same have the same settings"):
        check_variants([BASELINE, same], BASE, SEEDS)
    reseeded = Variant(
        "reseeded", "Another dataset", lambda c: replace(c, dataset=replace(c.dataset, seed=7))
    )
    with pytest.raises(ValueError, match="Variant reseeded changes dataset settings"):
        check_variants([BASELINE, reseeded], BASE, SEEDS)
    coreless = Variant(
        "coreless", "No message core", lambda c: replace(c, features=("entity_meta",))
    )
    with pytest.raises(ValueError, match="Variant coreless does not build"):
        check_variants([BASELINE, coreless], BASE, SEEDS)


def test_the_time_bound_comes_from_the_latest_graph_run(tmp_path: Path) -> None:
    runs = [
        PlannedRun(v, s, v.config(BASE, s), RunPaths.of(v.name, s, tmp_path), TRAIN)
        for v in (BASELINE, DROP)
        for s in SEEDS
    ]
    assert time_bound(runs, tmp_path) == {"bound_hours": None, "timed_from": None}
    history = write_run_files(RunPaths.of("baseline", 7, tmp_path)).history
    runs[0].action = KEEP
    bound = time_bound(runs, tmp_path)
    assert bound["timed_from"] == str(history)
    # Three runs to train, one epoch of three steps each, at about 3 s per step.
    assert bound["bound_hours"] == pytest.approx(3 * 3 * 3.0 / 3600, abs=0.01)


def test_a_history_that_timed_no_step_times_nothing(tmp_path: Path) -> None:
    runs = [PlannedRun(BASELINE, 42, BASE, RunPaths.of("baseline", 42, tmp_path), TRAIN)]
    older = write_run_files(RunPaths.of("baseline", 7, tmp_path))
    # A run restarted without resume state rewrites its history.csv as a bare header,
    # and an outage may stop the suite before it logs again: the newest history then
    # timed nothing, and the older one bounds the time.
    newer = write_run_files(RunPaths.of("baseline", 8, tmp_path))
    keep_history(newer.history, 0, 0)
    assert newer.history.read_text().count("\n") == 1
    os.utime(older.history, (1, 1))
    bound = time_bound(runs, tmp_path)
    assert bound["timed_from"] == str(older.history)
    assert bound["bound_hours"] == pytest.approx(3 * 3.0 / 3600, abs=0.01)
    # With that history alone, nothing bounds the time, and the suite's line is valid JSON.
    shutil.rmtree(older.root)
    bound = time_bound(runs, tmp_path)
    assert bound == {"bound_hours": None, "timed_from": None}
    assert json.loads(json.dumps(bound, allow_nan=False)) == bound


def test_an_outage_is_found_through_the_errors_it_caused() -> None:
    try:
        try:
            raise TigerGraphUnavailableError("down")
        except TigerGraphUnavailableError as error:
            raise RuntimeError("batch failed") from error
    except RuntimeError as wrapped:
        assert is_outage(wrapped)
    assert not is_outage(ValueError("own"))
