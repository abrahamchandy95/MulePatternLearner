"""`mule diagnose` on the fake graph: every analysis, its tables, study.json and report."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.artifacts import (
    DIAGNOSTIC_TABLES,
    read_diagnostic_table,
    read_events,
    read_feature_table,
    read_json,
    write_json,
    write_predictions,
    write_run_config,
)
from mule_pattern_learner.config import RunConfig, TransportConfig
from mule_pattern_learner.contract.graph_schema import PHASE_SPLIT
from mule_pattern_learner.contract.server import (
    ANALYTICS_CONTEXT_QUERY,
    ANALYTICS_QUERY_FILES,
    TRAINING_QUERY_FILES,
    TRUTH_QUERY,
)
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.diagnostics import study as diagnostics_study
from mule_pattern_learner.diagnostics.baselines import baselines
from mule_pattern_learner.diagnostics.nnpu_simulation import Problem, nnpu_simulation
from mule_pattern_learner.diagnostics.study import (
    ANALYSES,
    COMPLETE,
    INCOMPLETE,
    KEPT,
    SKIPPED,
    WRITTEN,
    analyses,
    diagnose,
)
from mule_pattern_learner.paths import DatasetPaths, DiagnosticsPaths, RunPaths
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.pipeline.connect import Session
from mule_pattern_learner.pipeline.diagnose import TigerGraphStudyReader
from mule_pattern_learner.reporting.study_report import ANALYSIS_FIGURES
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    reveal_inputs,
    saved_model,
    scope_population,
    unit_config,
)
from mule_pattern_learner.testing.fake_graph import (
    PREPARED_ACCOUNTS,
    FakeTigerGraph,
    prepared_graph,
)
from mule_pattern_learner.tigergraph.gsql_text import repository_queries

# A problem small enough for a test, as the simulation's own test has it.
SMALL = Problem(marginal=2_000, test_positives=30, test_negatives=30_000, steps=20, epochs=3)


def without_analytics() -> dict[str, str]:
    """The installed queries of a graph that `mule diagnose` has never run on."""
    return {name: text for name, (_, text) in repository_queries(TRAINING_QUERY_FILES).items()}


def write_proxy(run: RunPaths) -> None:
    """The proxy files of a complete run as diagnose reads them: its threshold and scores.

    The proxy predictions score every account of the validation and test populations,
    the revealed mules highest.
    """
    write_json(run.metrics, {"validation_proxy": {"threshold": 0.5}})
    population = pd.DataFrame(scope_population(PREPARED_ACCOUNTS))
    for split, date in (("validation", "2024-10-01"), ("test", "2025-01-01")):
        members = population[population.partition.map(PHASE_SPLIT) == split]
        observed = members.observed_positive.astype(np.int64).to_numpy()
        frame = pd.DataFrame(
            {
                "account_id": members.account_id.to_numpy(),
                "group_id": members.group_id.to_numpy(),
                "date": date,
                "observed_label": observed,
                "score": np.linspace(0.1, 0.4, len(members)) + 0.5 * observed,
            }
        )
        write_predictions(run.predictions(split), frame)


@pytest.fixture
def studied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RunConfig, FakeTigerGraph, DatasetPaths, RunPaths, list[TransportConfig]]:
    """A fake graph without the analytics queries, a prepared dataset and an audited run."""
    config = unit_config()
    data = tmp_path / "data"
    graph, dataset = prepared_graph(
        data, config, queries=without_analytics(), reveal=reveal_inputs()
    )
    assert dataset == DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    connected: list[TransportConfig] = []

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        connected.append(transport)
        return graph

    monkeypatch.setattr(pipeline_connect, "connect", connect)
    monkeypatch.setattr(pipeline_evaluate, "connect", connect)
    # A run of the dataset, audited on the fake graph, then given its proxy files.
    run = RunPaths.of("baseline", 42, tmp_path / "results")
    run.root.mkdir(parents=True)
    write_run_config(run.config, config, {"dataset_id": dataset.root.name})
    saved_model(run.model, config, dataset)
    pipeline_evaluate.evaluate_run(run, data=data)
    write_proxy(run)
    # The simulation's default problem takes about 15 s and the baselines' 1,000
    # replicates about as long here; the test's are smaller.
    small = partial(nnpu_simulation, seeds=(1,), problem=SMALL)
    monkeypatch.setattr(diagnostics_study, "nnpu_simulation", small)
    monkeypatch.setattr(diagnostics_study, "baselines", partial(baselines, replicates=40))
    connected.clear()
    graph.calls.clear()
    return config, graph, dataset, run, connected


def study_of(
    names: tuple[str, ...], config: RunConfig, dataset: DatasetPaths, run: RunPaths
) -> dict[str, Any]:
    session = Session(config.transport)
    reader = TigerGraphStudyReader(config, dataset, session)
    return diagnose(
        names,
        config=config,
        dataset=dataset,
        run=run,
        graph=reader,
        results=run.root.parent.parent,
    )


def test_every_analysis_writes_its_table_then_the_report(
    studied: tuple[RunConfig, FakeTigerGraph, DatasetPaths, RunPaths, list[TransportConfig]],
) -> None:
    config, graph, dataset, run, connected = studied
    result = study_of(analyses(None), config, dataset, run)
    paths = DiagnosticsPaths.of(dataset.root.name, run.root.parent.parent)
    assert result["status"] == COMPLETE and result["directory"] == str(paths.root)
    assert list(result["analyses"]) == list(ANALYSES)
    assert {outcome["status"] for outcome in result["analyses"].values()} == {WRITTEN}
    # One connection, the run's transport, and the truth read once for every analysis.
    assert connected == [config.transport]
    assert graph.names().count(TRUTH_QUERY) == 1
    # The analytics queries, which the graph lacked, are created and installed, once.
    analytics = list(repository_queries(ANALYTICS_QUERY_FILES))
    installs = [write for write in graph.writes if write.startswith("INSTALL QUERY")]
    assert installs == ["INSTALL QUERY " + ", ".join(analytics)]
    created = [write for write in graph.writes if "CREATE" in write]
    assert len(created) == len(ANALYTICS_QUERY_FILES)
    assert all(ANALYTICS_CONTEXT_QUERY not in write for write in created[1:])
    # The feature table and every analysis' long table, with their columns.
    frame = read_feature_table(paths.features)
    assert set(frame.split) == {"train", "validation", "test"}
    for name in DIAGNOSTIC_TABLES:
        table = read_diagnostic_table(paths.table(name), name)
        assert tuple(table.columns) == DIAGNOSTIC_TABLES[name] and len(table), name
    # The baselines and the curve carry the run's audits; the run is compared.
    table = read_diagnostic_table(paths.table("baselines"), "baselines")
    assert set(table[table.baseline == "model"].features) == {"baseline/seed-42"}
    record = read_json(paths.study)
    assert record["run"] == "baseline/seed-42" and record["run_compared"] is True
    assert record["reveal"] == {
        "salt": config.scope.reveal_salt,
        "budget": config.scope.reveal_per_split,
    }
    assert set(record["analyses"]) == set(ANALYSES)
    # The proxy predictions are scored against the graph's truth.
    proxy = read_diagnostic_table(paths.table("proxy_validity"), "proxy_validity")
    accounts = proxy[(proxy.subset == "all") & (proxy.metric == "n")]
    assert set(accounts.value) == {40.0}
    # The reveal's inputs were read through the executor, and the job never ran.
    assert graph.names().count("reveal inputs") == 1
    # Every line the analyses printed is in the study's events.jsonl.
    events = [event for event in read_events(paths.events) if event["event"] == "diagnose"]
    assert [event["analysis"] for event in events] == list(ANALYSES)
    figures = {name for names in ANALYSIS_FIGURES.values() for name in names}
    assert sorted(path.stem for path in paths.plots.glob("*.png")) == sorted(figures)
    assert paths.report.read_text().startswith(
        f"# Diagnostics of dataset `{dataset.root.name[:12]}`"
    )


def test_a_current_feature_table_is_kept_without_connecting(
    studied: tuple[RunConfig, FakeTigerGraph, DatasetPaths, RunPaths, list[TransportConfig]],
) -> None:
    config, graph, dataset, run, connected = studied
    first = study_of(("univariate",), config, dataset, run)
    # The table analysis built the feature table first.
    assert first["analyses"] == {"univariate": first["analyses"]["univariate"]}
    paths = DiagnosticsPaths.of(dataset.root.name, run.root.parent.parent)
    assert read_json(paths.study)["analyses"]["features"]["status"] == WRITTEN
    requested = len(graph.calls)
    again = study_of(("features", "drift"), config, dataset, run)
    assert again["analyses"]["features"]["status"] == KEPT
    assert len(graph.calls) == requested and connected == [config.transport]
    # study.json keeps the outcome of the analysis this call did not run.
    assert set(read_json(paths.study)["analyses"]) == {"features", "univariate", "drift"}
    # A table read with another query text is read again.
    stale = read_feature_table(paths.features).assign(analytics_contract="analytics_old")
    stale.to_parquet(paths.features, index=False)
    assert study_of(("features",), config, dataset, run)["analyses"]["features"]["status"] == (
        WRITTEN
    )


def test_the_run_analyses_are_skipped_when_the_run_cannot_be_compared(
    studied: tuple[RunConfig, FakeTigerGraph, DatasetPaths, RunPaths, list[TransportConfig]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, dataset, run, _ = studied
    for split in ("validation", "test"):
        run.audit_report(split).unlink()
    # A reason names the run's directory as the console shows a path: from the working
    # directory when it lies under it.
    home = run.root.parents[2]
    monkeypatch.chdir(home)
    result = study_of(("subgroups", "proxy-validity"), config, dataset, run)
    assert result["status"] == INCOMPLETE
    named = run.root.relative_to(home).as_posix()
    # Both read the run's audit samples: the proxy validity takes its hidden and revealed
    # mules from them.
    assert result["skipped"] == {
        "subgroups": f"{named} has no audit; run `mule evaluate`",
        "proxy-validity": f"{named} has no audit; run `mule evaluate`",
    }
    # A run of another dataset is not compared at all.
    write_run_config(run.config, config, {"dataset_id": "other"})
    result = study_of(("baselines", "proxy-validity"), config, dataset, run)
    assert result["analyses"]["baselines"]["status"] == WRITTEN
    assert result["analyses"]["proxy-validity"]["status"] == SKIPPED
    assert "was trained on another dataset" in result["skipped"]["proxy-validity"]
    paths = DiagnosticsPaths.of(dataset.root.name, run.root.parent.parent)
    table = read_diagnostic_table(paths.table("baselines"), "baselines")
    assert not (table.baseline == "model").any()
    assert read_json(paths.study)["run_compared"] is False


def test_the_analyses_are_named_as_the_command_line_names_them() -> None:
    assert analyses(None) == ANALYSES and analyses("drift") == ("drift",)
    with pytest.raises(ValueError, match="Unknown analysis 'graph'"):
        analyses("graph")
    tables = {name.replace("-", "_") for name in ANALYSES if name != "features"}
    assert tables == set(DIAGNOSTIC_TABLES)


def test_an_analysis_that_fails_leaves_the_outcomes_of_those_before_it(
    studied: tuple[RunConfig, FakeTigerGraph, DatasetPaths, RunPaths, list[TransportConfig]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, dataset, run, _ = studied

    def outage(*args: Any) -> pd.DataFrame:
        raise RuntimeError("TigerGraph is unavailable")

    monkeypatch.setattr(diagnostics_study, "reveal_spread", outage)
    with pytest.raises(RuntimeError, match="unavailable"):
        study_of(("univariate", "reveal-spread"), config, dataset, run)
    paths = DiagnosticsPaths.of(dataset.root.name, run.root.parent.parent)
    assert set(read_json(paths.study)["analyses"]) == {"features", "univariate"}
    assert paths.table("univariate").exists() and not paths.report.exists()
