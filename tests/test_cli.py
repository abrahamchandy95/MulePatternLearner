"""mule, the command line (docs/design.md, "Commands"): nine commands with no option, the readiness
check each long command runs first, and the commands run end to end on three small worlds.

The worlds are testing.fixtures.batch_world, three times, each with one more payment than the
last on the first days of the export so that each has an identity of its own, and the two
payments that span the export window (the first on day 0, the last on its last day) that
mule prepare's calendar check asks for. They are the loaded worlds D1, D2 and D3 of the
rotation, each behind a spy executor, and mule prepare, mule fetch, mule train and mule evaluate
run on them as they run on three worlds loaded in turn. The calendar has three scored weeks,
starting on T - 1 week, 2024-06-24, T and T + 1 week, each scored at a time of its own inside it.
The world's mules are active in the days before T, so from the first week's scoring time, in the
last days of that week for the seeds of D1, D2 and D3, they are inside the label rule's window:
two of them, a_mule and a_r1, are identified and active at that time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import date
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from typing import Any

import pandas as pd
import pytest
import torch

from mule_pattern_learner import cli, paths
from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    FULL_DESIGN_CONFIG,
    CalendarConfig,
    Rotation,
    RunConfig,
    ScoreConfig,
    WorldsConfig,
)
from mule_pattern_learner.data.labels import read_identified
from mule_pattern_learner.data.population import list_population
from mule_pattern_learner.data.worlds import LoadedWorld, registered
from mule_pattern_learner.evaluation import audit as auditing
from mule_pattern_learner.evaluation.answer_key import read_answer_key
from mule_pattern_learner.evaluation.audit import (
    CLOCK_GAIN,
    FEATURE_AUDIT,
    GATE_RECORD,
    audit_settings,
)
from mule_pattern_learner.evaluation.clock_invariance import (
    CLOCK_INVARIANCE,
    Compared,
    Invariance,
    week_start_keys,
    with_week_starts,
)
from mule_pattern_learner.evaluation.fallback import by_week, fallback_keys
from mule_pattern_learner.evaluation.label_checks import IDENTIFICATION_HORIZON, LABEL_CHECKS
from mule_pattern_learner.evaluation.sample import audit_sample
from mule_pattern_learner.evaluation.store import AuditRecord, Store
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import DAY_0, Spy, identity_text, world_of
from mule_pattern_learner.testing.parity import Communities, Parity, QueryParity
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph import installer
from mule_pattern_learner.tigergraph.queries import QUERY_FILES, VERSIONS
from mule_pattern_learner.tigergraph.world import cache_key, time_folder
from mule_pattern_learner.training.batches import ScoringBatches

W = 7 * DAY_MS
CALENDAR = CalendarConfig(history_days=175, scored_weeks=3)
LAST_DAY = DAY_0 + (CALENDAR.export_days - 1) * DAY_MS
ROTATION = Rotation("D1", "D2", "D3")
# Three scored weeks; one rotation of the development worlds; small, fast settings: batches of
# 32, two epochs, one seed, the grid of two priors, a few replicates.
CONFIG = replace(
    FULL_DESIGN_CONFIG,
    calendar=CALENDAR,
    worlds=WorldsConfig(rotations=(ROTATION,)),
    data=replace(FULL_DESIGN_CONFIG.data, min_selection_identified=1),
    training=replace(
        FULL_DESIGN_CONFIG.training,
        # The proof of concept's reuse of built examples; the other settings are the full design's,
        # since the reference worlds hold a few hundred accounts, too few to sample.
        reuse_built_examples=True,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=2,
        patience=2,
        seeds=(0,),
        prior_multiples=(1, 3),
    ),
    audit=replace(FULL_DESIGN_CONFIG.audit, replicates=30, nonmule_sample=60),
)
# Two rotations of the development worlds: D2 never trains in the pipeline and D1 never tests, so
# each rotation's second half has no examples of its own.
TWO_ROTATIONS = replace(
    CONFIG, worlds=WorldsConfig(rotations=(ROTATION, Rotation("D2", "D3", "D1")))
)
PASSING = {
    GATE_RECORD: True,
    **dict.fromkeys(LABEL_CHECKS, True),
    IDENTIFICATION_HORIZON: True,
    CLOCK_GAIN: False,
    FEATURE_AUDIT: True,
}


def _world(k: int) -> WorldBuilder:
    """The batch world with the payments that give it an identity of its own, and the two that
    span the export window."""
    w = fixtures.batch_world()
    w.external("x_far", fixtures.OPENED)
    w.pay("x_first", "a_h1", "x_far", DAY_0 + HOUR_MS, 25.0, "ach")
    w.pay("x_last", "x_far", "a_h2", LAST_DAY + 23 * HOUR_MS, 40.0, "ach")
    w.pay("x_variant", "a_h1", "x_far", DAY_0 + 2 * DAY_MS, 1.0 + k, "ach")
    return w


class Live(Spy):
    """A loaded world as a connection executor: a spy over a reference graph, which the
    commands run their queries through."""

    graph_name = "Mule_Pattern_Learner"
    client: Any = None

    def call[T](
        self, operation: Callable[[Any], T], *, what: str, attempts: int | None = None
    ) -> T:
        raise AssertionError("these commands run no connection operation")

    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str:
        raise AssertionError("these commands run no GSQL statement")


def _store_clock_sample(name: str, graph: ReferenceGraph, seed: int) -> None:
    """What mule audit leaves of a test-role world beside its pass record: the audit sample
    stored with the start of each row's week, and the rows at those starts fetched."""
    spy = Spy(graph)
    identity = registered(name).identity
    cache = Cache.of(identity, CONFIG.subgraph)
    times = CONFIG.calendar.scoring_times(seed)
    populations = {t_ms: [row.account_id for row in list_population(spy, t_ms)] for t_ms in times}
    sample = audit_sample(
        read_answer_key(spy), populations, CONFIG.audit.nonmule_sample, CONFIG.data.draw_key, times
    )
    fetch(spy, cache, by_week(sample), CONFIG.transport)
    starts = dict(zip(times, CONFIG.calendar.week_starts(), strict=True))
    scorable = {
        start: [row.account_id for row in list_population(spy, start)] for start in starts.values()
    }
    stored = with_week_starts(sample, starts, scorable)
    Store(identity).write_clock_sample(stored)
    fetch(spy, cache, by_week(week_start_keys(stored)), CONFIG.transport)


def _verified(executor: object) -> None:
    """installer.verify, when the installed queries are the repository's."""


def _no_roles(name: str, rotations: object) -> frozenset[str]:
    """data.worlds.roles of a world that plays none."""
    return frozenset()


@dataclass(frozen=True)
class Pipeline:
    """Three loaded worlds, their folders, and what the commands have made of them."""

    graphs: dict[str, ReferenceGraph]
    seeds: dict[str, int]
    records: Path
    results: Path

    def connect(self, name: str) -> Callable[[], Live]:
        return lambda: Live(self.graphs[name])

    def identity(self, name: str) -> str:
        return registered(name).identity


@pytest.fixture(autouse=True)
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed queries are the repository's: the readiness check's first step, which
    needs a live server, passes."""
    monkeypatch.setattr(installer, "verify", _verified)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pipeline]:
    """mule prepare and mule fetch run on each of the three worlds in turn, the audit records
    are written as a passing audit writes them, and mule train and mule evaluate run on the
    rotation, all under folders of the test's own."""
    folder = tmp_path_factory.mktemp("cli")
    records = folder / "records"
    records.mkdir()
    graphs = {name: _world(k).graph() for k, name in enumerate(("D1", "D2", "D3"))}
    seeds = {"D1": 1001, "D2": 1002, "D3": 1003}
    for name, graph in graphs.items():
        (records / f"world_{seeds[name]}.identity.toml").write_text(
            identity_text(world_of(graph), seeds[name])
        )
    gsql = folder / "gsql"
    gsql.mkdir()
    fixtures.write_parity_record(gsql / "parity.toml")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paths, "DATA_DIR", folder / "data")
        patch.setattr(paths, "RESULTS_DIR", folder / "results")
        patch.setattr(paths, "GSQL_DIR", gsql)
        patch.setattr(installer, "verify", _verified)
        patch.setenv("MULE_PL_RECORDS_DIR", str(records))
        found = Pipeline(graphs, seeds, records, folder / "results")
        for name in graphs:
            cli.prepare_command(CONFIG, found.connect(name))
            cli.fetch_command(CONFIG, found.connect(name))
            identity = found.identity(name)
            store = Store(identity)
            store.write_answer_key(read_answer_key(Spy(graphs[name])))
            store.write_labels(read_identified(Spy(graphs[name])))
            store.write_record(
                AuditRecord(
                    name,
                    "development",
                    identity,
                    cache_key(CONFIG.subgraph),
                    audit_settings(CONFIG),
                    dict(VERSIONS),
                    dict(PASSING),
                )
            )
        # D3 tests in the rotation, so mule audit leaves its clock sample.
        _store_clock_sample("D3", graphs["D3"], seeds["D3"])
        yield found


# The command line.


def test_mule_is_the_command_line_with_the_designs_commands() -> None:
    project = tomllib.loads((paths.REPOSITORY_ROOT / "pyproject.toml").read_text())
    assert project["project"]["scripts"] == {"mule": "mule_pattern_learner.cli:main"}
    names = (
        "install",
        "parity",
        "prepare",
        "fetch",
        "audit",
        "train",
        "evaluate",
        "score",
        "bench",
    )
    assert tuple(cli.COMMANDS) == names and tuple(cli.RUNNERS) == names


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    return tmp_path / "results"


@pytest.mark.parametrize("command", list(cli.COMMANDS))
def test_each_command_parses_and_dispatches_with_the_default_config(
    command: str, results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[RunConfig, object]] = []
    monkeypatch.setitem(
        cli.RUNNERS, command, lambda config, connect: seen.append((config, connect))
    )
    cli.main([command])
    assert seen == [(DEFAULT_CONFIG, cli.connect)]
    # The command appends that it started and finished, in results/commands/ (bench's in its own
    # folder).
    folder = paths.bench_dir() if command == "bench" else results / "commands"
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert [(event["command"], event["event"]) for event in events] == [
        (command, "started"),
        (command, "finished"),
    ]


@pytest.mark.parametrize("command", list(cli.COMMANDS))
@pytest.mark.parametrize("argument", ["--anything", "--help", "-h", "D1", "extra"])
def test_a_command_refuses_every_argument(
    command: str, argument: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main([command, argument])
    assert stopped.value.code == 2
    assert f"unrecognized arguments: {argument}" in capsys.readouterr().err


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main([])
    assert stopped.value.code == 2 and "COMMAND" in capsys.readouterr().err


def test_a_command_that_stops_ends_with_its_message_and_its_event(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stops(config: RunConfig, connect: object) -> None:
        raise cli.CommandError("the audit of D2 failed")

    monkeypatch.setitem(cli.RUNNERS, "audit", stops)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["audit"])
    assert stopped.value.code == "mule audit stopped: the audit of D2 failed"
    events = [
        json.loads(line)
        for line in (results / "commands" / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["started", "stopped"]
    assert events[1]["error"] == "CommandError: the audit of D2 failed"


def test_a_bug_keeps_its_traceback(results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def breaks(config: RunConfig, connect: object) -> None:
        raise KeyError("a_missing")

    monkeypatch.setitem(cli.RUNNERS, "fetch", breaks)
    with pytest.raises(KeyError):
        cli.main(["fetch"])


def test_python_dash_m_runs_the_same_command_line() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mule_pattern_learner", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(command in result.stdout for command in cli.COMMANDS)
    refused = subprocess.run(
        [sys.executable, "-m", "mule_pattern_learner", "prepare", "--anything"],
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2 and "unrecognized arguments: --anything" in refused.stderr


# The readiness check.


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What the commands print."""
    lines: list[str] = []
    monkeypatch.setattr(cli, "show", lines.append)
    return lines


def test_readiness_passes_on_a_loaded_world_and_changes_nothing(
    pipeline: Pipeline, shown: list[str]
) -> None:
    before = sorted(path for path in paths.DATA_DIR.rglob("*") if path.is_file())
    spy = Live(pipeline.graphs["D2"])
    loaded = cli.check_ready(spy, CONFIG)
    assert loaded.config.name == "D2"
    assert [line.split(":")[0] for line in shown] == ["ready", "ready", "ready"]
    assert "one training step ran" in shown[-1]
    # The first scorable account of the first week was fetched, into a scratch folder.
    assert {name for name in spy.names()} >= {"read_world", "list_accounts", "read_payment_events"}
    assert sorted(path for path in paths.DATA_DIR.rglob("*") if path.is_file()) == before


def test_readiness_names_the_installed_queries_that_are_not_the_repositorys(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(executor: object) -> None:
        raise ValueError("The installed queries are not the repository's: read_hubs is missing")

    monkeypatch.setattr(installer, "verify", refuse)
    with pytest.raises(cli.NotReady, match=r"^installed queries: .*read_hubs is missing"):
        cli.check_ready(Live(pipeline.graphs["D1"]), CONFIG)


def test_readiness_names_a_world_that_is_no_configured_one(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No identity file of the folder matches the loaded world.
    monkeypatch.setenv("MULE_PL_RECORDS_DIR", str(tmp_path))
    with pytest.raises(cli.NotReady, match=r"^the world's identity: No identity file of 0 matches"):
        cli.check_ready(Live(pipeline.graphs["D1"]), CONFIG)
    # The world does not hold the calendar's window: its last payment is not on the last day.
    w = _world(0)
    w.pay("x_after", "a_h1", "x_far", LAST_DAY + 30 * DAY_MS, 1.0, "ach")
    graph = w.graph()
    (tmp_path / "world_1001.identity.toml").write_text(identity_text(world_of(graph), 1001))
    with pytest.raises(cli.NotReady, match=r"^the world's identity: .* last payment"):
        cli.check_ready(Live(graph), CONFIG)


def test_readiness_names_a_world_with_no_account_to_fetch(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    w = WorldBuilder()
    w.external("x_a", fixtures.OPENED)
    w.external("x_b", fixtures.OPENED)
    w.pay("x_first", "x_a", "x_b", DAY_0 + HOUR_MS, 1.0, "ach")
    w.pay("x_last", "x_b", "x_a", LAST_DAY + HOUR_MS, 1.0, "ach")
    graph = w.graph()
    monkeypatch.setenv("MULE_PL_RECORDS_DIR", str(tmp_path))
    (tmp_path / "world_1001.identity.toml").write_text(identity_text(world_of(graph), 1001))
    with pytest.raises(cli.NotReady, match=r"^one account fetched: .* no scorable account"):
        cli.check_ready(Live(graph), CONFIG)


def test_readiness_names_a_fetch_that_fails(pipeline: Pipeline) -> None:
    class Failing(Live):
        def run(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            name: str,
            params: dict[str, Any],
            *,
            timeout_s: float | None = None,
            attempts: int | None = None,
            timeout_retries: int = 1,
        ) -> list[dict[str, Any]]:
            if name == "read_payment_events":
                raise RuntimeError("TigerGraph timed out")
            return super().run(
                name,
                params,
                timeout_s=timeout_s,
                attempts=attempts,
                timeout_retries=timeout_retries,
            )

    with pytest.raises(cli.NotReady, match=r"^one account fetched: .*timed out"):
        cli.check_ready(Failing(pipeline.graphs["D1"]), CONFIG)


def test_a_command_stops_at_a_failed_readiness_check_before_its_work(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(executor: object) -> None:
        raise ValueError("The installed queries are not the repository's: read_hubs is missing")

    monkeypatch.setattr(installer, "verify", refuse)
    worked: list[str] = []

    def prepares(*args: object, **keywords: object) -> None:
        worked.append("prepare")

    monkeypatch.setattr(cli, "prepare", prepares)
    # mule score looks for a model of its run before it asks for the graph, so the run holds one
    # (an empty file: the readiness check stops it before the file is read).
    stub = paths.results_dir("stub-run")
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "gnn_seed0.pt").write_bytes(b"")
    config = replace(CONFIG, score=ScoreConfig(("a_mule",), (date(2024, 7, 1),), "stub-run"))
    for command in (
        cli.parity_command,
        cli.prepare_command,
        cli.fetch_command,
        cli.audit_command,
        cli.bench_command,
        cli.score_command,
    ):
        with pytest.raises(cli.NotReady, match=r"^installed queries: "):
            command(config, pipeline.connect("D1"))
    assert worked == []


# The commands on the worlds.


def test_prepare_registered_each_world_and_wrote_the_examples_of_its_role(
    pipeline: Pipeline,
) -> None:
    for name, tables in {
        "D1": {"training.parquet", "report.parquet"},
        "D2": {"selection.parquet", "report.parquet"},
        "D3": {"test.parquet", "report.parquet"},
    }.items():
        held = registered(name)
        found = list(paths.examples_dir(held.identity).glob("*/*/*.parquet"))
        assert {path.name for path in found} == tables
    # The caches hold the keys the roles need: the training draw, the ranked selection keys and
    # every scorable account of every week of the test world, at the week's scoring time; and, in
    # the test world's, the rows of mule audit's clock sample at the start of each week.
    times = CONFIG.calendar.scoring_times(pipeline.seeds["D3"])
    folders = {path.name for path in paths.cache_dir(registered("D3").identity).glob("*/*")}
    assert folders == {time_folder(t_ms) for t_ms in (*times, *CONFIG.calendar.week_starts())}
    assert len(folders) == 6


def test_fetch_makes_no_request_for_a_cache_that_is_full(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = Live(pipeline.graphs["D3"])
    cli.fetch_command(CONFIG, lambda: spy)
    requested = {name for name in spy.names()}
    # The readiness check fetches one account, a scratch folder's, and the world is read; the
    # real fetch finds every row of the test world cached, and asks for the hubs of no week
    # (the account of the readiness check apart).
    assert requested <= {
        "read_world",
        "read_labels",
        "list_accounts",
        "read_hubs",
        "read_payment_events",
        "read_shared_identity",
        "read_community",
        "read_money_flow",
        "read_counterparties",
        "read_activity_change",
    }
    scored = [params["account_ids"] for name, params in spy.calls if name == "read_community"]
    assert len(scored) == 1 and len(scored[0]) == 1


def test_fetch_of_a_world_that_plays_no_role_fetches_nothing_and_says_so(
    pipeline: Pipeline, shown: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 5,000-person test world, and any world no rotation names, plays no role.
    spy = Live(pipeline.graphs["D1"])
    monkeypatch.setattr(cli, "roles", _no_roles)
    cli.fetch_command(CONFIG, lambda: spy)
    assert shown[-1].startswith("nothing to fetch: D1 plays no role")
    assert "read_payment_events" not in spy.names()[spy.names().index("read_labels") :]


def test_the_tables_a_role_needs_fetched(pipeline: Pipeline) -> None:
    assert cli.role_tables(frozenset(), "whole") == []
    assert cli.role_tables(frozenset({"train", "select", "test"}), "whole") == [
        "training.parquet",
        "selection.parquet",
        "test.parquet",
    ]
    # Under the fallback the test world's keys are mule audit's, not mule fetch's.
    assert cli.role_tables(frozenset({"test"}), "fallback") == []


# mule train.


def _records(pipeline: Pipeline, changes: dict[str, dict[str, bool]]) -> None:
    for name, checks in changes.items():
        store = Store(pipeline.identity(name))
        held = store.record()
        assert held is not None
        store.write_record(replace(held, checks={**held.checks, **checks}))


def test_train_stops_on_a_failed_feature_audit_and_names_it(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _records(pipeline, {"D2": {FEATURE_AUDIT: False}})
    try:
        with pytest.raises(cli.CommandError, match="training stops: D2: feature_audit"):
            cli.train_command(CONFIG, cli.connect)
    finally:
        _records(pipeline, {"D2": {FEATURE_AUDIT: True}})


def test_train_stops_on_a_failed_label_check_and_a_missing_record(pipeline: Pipeline) -> None:
    label = sorted(LABEL_CHECKS)[0]
    _records(pipeline, {"D1": {label: False}})
    try:
        with pytest.raises(cli.CommandError, match=f"D1: {label}"):
            cli.train_command(CONFIG, cli.connect)
    finally:
        _records(pipeline, {"D1": {label: True}})
    # A failed clock gain test of a development world does not stop it (D1's, D2's and D3's all
    # fail it): a world with no record does.
    record = Store(pipeline.identity("D3")).folder / "audit.toml"
    kept = record.read_text()
    record.unlink()
    try:
        with pytest.raises(cli.CommandError, match="D3: no audit record"):
            cli.train_command(CONFIG, cli.connect)
    finally:
        record.write_text(kept)


def test_train_reports_a_development_world_s_failed_audit_and_goes_on_in_the_proof_of_concept(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, shown: list[str]
) -> None:
    trained_rotations: list[str] = []

    def trains(worlds: cli.RotationWorlds, config: RunConfig, device: object) -> None:
        trained_rotations.append(worlds.name)

    monkeypatch.setattr(cli, "train_rotation_command", trains)
    # The events are collected here, not appended to the rotation's results folder.
    events: list[tuple[str, str]] = []

    def collects(
        folder: Path, command: str, event: str, error: BaseException | None = None
    ) -> None:
        events.append((command, event))

    monkeypatch.setattr(cli, "append_event", collects)
    config = replace(CONFIG, training=replace(CONFIG.training, development_audit_stops=False))
    _records(pipeline, {"D2": {FEATURE_AUDIT: False}})
    try:
        cli.train_command(config, cli.connect)
    finally:
        _records(pipeline, {"D2": {FEATURE_AUDIT: True}})
    assert trained_rotations == [cli.run_name(ROTATION)]
    assert "D1-D2-D3: audit not passed, and training goes on: D2: feature_audit" in shown
    assert ("train", "audit not passed: D2: feature_audit") in events


def test_train_names_a_world_that_was_never_prepared(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(CONFIG, worlds=WorldsConfig(rotations=(Rotation("A", "B", "C"),)))
    with pytest.raises(cli.NotReady, match=r"^the world's identity: No world is registered as A"):
        cli.train_command(config, cli.connect)


def test_train_names_each_world_of_a_failed_audit_once(pipeline: Pipeline) -> None:
    # D2 is in both rotations, and its failed audit is named once.
    _records(pipeline, {"D2": {FEATURE_AUDIT: False}})
    try:
        with pytest.raises(cli.CommandError, match=r"^training stops: D2: feature_audit\. Run"):
            cli.train_command(TWO_ROTATIONS, cli.connect)
    finally:
        _records(pipeline, {"D2": {FEATURE_AUDIT: True}})


def test_train_checks_every_rotation_before_the_first_one_trains(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, shown: list[str]
) -> None:
    trained_rotations: list[str] = []

    def trains(worlds: cli.RotationWorlds, config: RunConfig, device: object) -> None:
        trained_rotations.append(worlds.name)

    monkeypatch.setattr(cli, "train_rotation_command", trains)
    # The second rotation trains on D2, which was prepared to select and has no training examples.
    with pytest.raises(cli.NotReady, match=r"^one example read: the cache and examples of D2 are"):
        cli.train_command(TWO_ROTATIONS, cli.connect)
    # The first rotation's own check passed, and nothing trained.
    assert any(line.startswith("ready: read the example of") and "of D1" in line for line in shown)
    assert trained_rotations == []


def test_train_finds_a_selection_world_that_was_never_fetched_before_it_trains(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A copy of the data folder in which D2, the selection world, was prepared and never fetched:
    # its selection pass would fail with a missing hub after the first epoch.
    copy = tmp_path / "data"
    shutil.copytree(paths.DATA_DIR, copy)
    monkeypatch.setattr(paths, "DATA_DIR", copy)
    shutil.rmtree(paths.cache_dir(pipeline.identity("D2")))
    worlds = cli.rotation_worlds(ROTATION, CONFIG)
    with pytest.raises(cli.NotReady, match=r"^one example read: the cache and examples of D2 are"):
        cli.check_trainable(worlds, CONFIG)
    with pytest.raises(cli.NotReady, match=r"^one example read: the cache and examples of D2 are"):
        cli.train_command(CONFIG, cli.connect)


@pytest.fixture(scope="module")
def trained(pipeline: Pipeline) -> Path:
    """mule train, run on the rotation once."""
    cli.train_command(CONFIG, cli.connect)
    return paths.results_dir(cli.run_name(ROTATION))


def test_train_writes_every_model_file_of_the_rotation(trained: Path) -> None:
    names = {path.name for path in trained.glob("*.pt")}
    # One seed of the full model and of each of the four ablations, and each XGBoost comparison
    # once, with no seed.
    assert names == {
        "gnn_seed0.pt",
        "gnn_scored_alone_seed0.pt",
        "gnn_without_devices_and_ips_seed0.pt",
        "gnn_without_community_seed0.pt",
        "gnn_without_payment_timing_seed0.pt",
        "xgboost_own.pt",
        "xgboost_graph.pt",
    }
    # The development rotation keeps every epoch of seed 0, and decides on the prior grid.
    assert len(list((trained / "epochs").glob("*.pt"))) == 2 * 2
    decision = (trained / "prior_decision.txt").read_text()
    assert "prior 1x against 3x" in decision and "reported runs" in decision
    events = [json.loads(line) for line in (trained / "events.jsonl").read_text().splitlines()]
    assert [(event["command"], event["event"]) for event in events] == [
        ("train", "started"),
        ("train", "finished"),
    ]


# mule evaluate.


@pytest.fixture(scope="module")
def evaluated(trained: Path) -> Path:
    cli.evaluate_command(CONFIG, cli.connect)
    return trained


def test_evaluate_writes_the_tables_marked_development_only(evaluated: Path) -> None:
    for name in (
        "metrics",
        "per_week",
        "groups",
        "spread",
        "comparisons",
        "mules",
        "victim_origin",
        "selection_epochs",
        "selection_correlation",
        CLOCK_INVARIANCE,
    ):
        table = pd.read_parquet(evaluated / f"{name}.parquet")
        assert table["development_only"].all() and (table["rotation"] == "D1-D2-D3").all()
    metrics = pd.read_parquet(evaluated / "metrics.parquet")
    # Seven runs: five seeds' worth is one seed here, of five rankers, and two XGBoost models
    # with no seed.
    assert set(metrics["ranker"]) == {
        "gnn",
        "gnn_scored_alone",
        "gnn_without_devices_and_ips",
        "gnn_without_community",
        "gnn_without_payment_timing",
        "xgboost_own",
        "xgboost_graph",
    }
    assert metrics.loc[metrics["ranker"].str.startswith("xgboost"), "seed"].isna().all()
    manifest = json.loads((evaluated / "manifest.json").read_text())
    assert manifest["reported"] is False and manifest["world_identities"].keys() == {
        "D1",
        "D2",
        "D3",
    }
    assert (evaluated / "missed" / "mules.parquet").is_file()
    events = [json.loads(line) for line in (evaluated / "events.jsonl").read_text().splitlines()]
    assert ("evaluate", "finished") in [(e["command"], e["event"]) for e in events]


def test_evaluate_runs_the_clock_invariance_test_of_the_selected_gnn_on_the_stored_sample(
    evaluated: Path, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mule evaluate hands the invariance test the sample mule audit stored for the test world,
    its scoring times and the week starts of the calendar, and a scorer that gives the selected
    GNN's logits; the table is written, and the events say how it ended."""
    seen: dict[str, Any] = {}
    real = cli.invariance

    def watching(
        stored: pd.DataFrame,
        key: Any,
        scoring_times: Any,
        week_starts: Any,
        score: Callable[[pd.DataFrame], pd.DataFrame],
        audit: Any,
        seed: int,
    ) -> Any:
        seen.update(stored=stored, times=scoring_times, starts=week_starts, seed=seed)
        seen["scored"] = score(stored.loc[:, ["t_ms", "account_id"]])
        return real(stored, key, scoring_times, week_starts, score, audit, seed)

    monkeypatch.setattr(cli, "invariance", watching)
    worlds = cli.rotation_worlds(ROTATION, CONFIG)
    cli.evaluate_rotation(worlds, CONFIG, torch.device("cpu"))
    identity = pipeline.identity("D3")
    stored = Store(identity).clock_sample()
    assert stored is not None
    pd.testing.assert_frame_equal(seen["stored"], stored)
    assert seen["times"] == CONFIG.calendar.scoring_times(pipeline.seeds["D3"])
    assert seen["starts"] == CONFIG.calendar.week_starts()
    assert seen["seed"] == CONFIG.training.bootstrap_seed
    # The scorer is the selected GNN's: the one seed's model file, under the current lock.
    keys = stored.loc[:, ["t_ms", "account_id"]]
    direct = cli.score_gnn(
        evaluated / "gnn_seed0.pt",
        cli.gnn_lock(None, CONFIG.model, CONFIG.subgraph, CONFIG.training),
        ScoringBatches(
            Cache.of(identity, CONFIG.subgraph), identity, keys, CONFIG.training.batch_size
        ),
    )
    assert seen["scored"].equals(direct) and len(direct) == len(keys)
    table = pd.read_parquet(evaluated / "clock_invariance.parquet")
    assert len(table) == 1 and bool(table["passed"].item()) in (True, False)
    events = [json.loads(line) for line in (evaluated / "events.jsonl").read_text().splitlines()]
    ended = [e["event"] for e in events if e["event"].startswith("clock invariance")]
    assert ended and set(ended) <= {"clock invariance passed", "clock invariance not passed"}


def test_evaluate_without_a_stored_clock_sample_runs_no_invariance_test_and_says_so(
    evaluated: Path, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "results" / cli.run_name(ROTATION)
    shutil.copytree(evaluated, folder)
    (folder / "clock_invariance.parquet").unlink()
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    stored = Store(pipeline.identity("D3")).folder / "clock_sample.parquet"
    kept = stored.read_bytes()
    stored.unlink()
    try:
        cli.evaluate_rotation(cli.rotation_worlds(ROTATION, CONFIG), CONFIG, torch.device("cpu"))
    finally:
        stored.write_bytes(kept)
    assert not (folder / "clock_invariance.parquet").exists()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert f"{CLOCK_INVARIANCE} was not run" in manifest["not_reported_because"]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert "clock invariance not run" in [event["event"] for event in events]


def test_a_rotations_verdict_holds_the_clock_invariance_test(pipeline: Pipeline) -> None:
    worlds = cli.rotation_worlds(ROTATION, CONFIG)
    verdict = cli._verdict  # pyright: ignore[reportPrivateUsage]

    def finding(passed: bool) -> Invariance:
        nothing = Compared(0.5, 0.5, 0.0, (-0.01, 0.01))
        return Invariance(
            nothing,
            nothing,
            0.05,
            passed,
            rows=8,
            mule_rows=4,
            clusters=8,
            mule_clusters=4,
            mule_rows_left_out=0,
            nonmule_rows_left_out=0,
        )

    assert f"{CLOCK_INVARIANCE} was not run" in verdict(worlds, CONFIG, None).reasons
    assert f"{CLOCK_INVARIANCE} did not pass" in verdict(worlds, CONFIG, finding(False)).reasons
    reasons = verdict(worlds, CONFIG, finding(True)).reasons
    assert not any(reason.startswith(CLOCK_INVARIANCE) for reason in reasons)


def test_evaluate_builds_each_test_example_once_when_the_examples_are_kept(
    evaluated: Path, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof of concept (docs/design.md, "Proof-of-concept settings (25 October 2026)"): the test
    world's examples are built once for all seven runs, and once per run in the full design."""
    made: list[ScoringBatches] = []

    class Recording(ScoringBatches):
        def __init__(self, *args: Any, **keywords: Any) -> None:
            super().__init__(*args, **keywords)
            made.append(self)

    monkeypatch.setattr(cli, "ScoringBatches", Recording)
    worlds = cli.rotation_worlds(ROTATION, CONFIG)
    cli.evaluate_rotation(worlds, CONFIG, torch.device("cpu"))
    test_world = made[0]
    assert test_world.built == len(test_world)
    made.clear()
    anew = replace(CONFIG, training=replace(CONFIG.training, reuse_built_examples=False))
    cli.evaluate_rotation(worlds, anew, torch.device("cpu"))
    assert made[0].built == 7 * len(made[0])


def test_evaluate_checks_every_rotation_before_the_first_one_is_evaluated(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluated_rotations: list[str] = []

    def evaluates(worlds: cli.RotationWorlds, config: RunConfig, device: object) -> pd.DataFrame:
        evaluated_rotations.append(worlds.name)
        return pd.DataFrame()

    monkeypatch.setattr(cli, "evaluate_rotation", evaluates)
    # The second rotation tests on D1, which was prepared to train and has no test examples.
    with pytest.raises(cli.NotReady, match=r"^one example read: the cache and examples of D1 are"):
        cli.evaluate_command(TWO_ROTATIONS, cli.connect)
    assert evaluated_rotations == []


def _metrics(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ranker": ["gnn"],
            "seed": [0],
            "group": ["all"],
            "metric": ["roc_auc"],
            "value": [value],
            "selected": [True],
        }
    )


def test_the_spread_over_rotations_pools_only_rotations_of_one_kind() -> None:
    metrics = {
        "A-B-C": _metrics(0.6),
        "B-C-A": _metrics(0.8),
        "C-A-B": _metrics(0.7),
        "D1-D2-D3": _metrics(0.1),
    }
    kinds = dict.fromkeys(("A-B-C", "B-C-A", "C-A-B"), "reported") | {"D1-D2-D3": "development"}
    found = cli.rotation_spreads(metrics, kinds)
    # The development rotation, alone of its kind, has no spread, and is not pooled with the
    # three reported ones: their mean is 0.7 over 3 rotations, not 0.55 over 4.
    assert set(found) == {"reported"}
    mean = found["reported"][found["reported"]["of"] == "seed_mean"]
    assert mean["rotations"].tolist() == [3] and mean["mean"].item() == pytest.approx(0.7)
    # Two development rotations make a spread of their own.
    both = cli.rotation_spreads(
        {**metrics, "D2-D3-D1": _metrics(0.3)}, {**kinds, "D2-D3-D1": "development"}
    )
    assert set(both) == {"reported", "development"}
    assert both["development"]["rotations"].tolist() == [2, 2]
    assert cli.rotation_spreads({}, {}) == {}


def test_evaluate_writes_one_spread_for_each_kind_with_two_rotations(
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    shown: list[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")

    def ready(worlds: cli.RotationWorlds, config: RunConfig) -> None:
        return None

    monkeypatch.setattr(cli, "check_evaluable", ready)

    def evaluates(worlds: cli.RotationWorlds, config: RunConfig, device: object) -> pd.DataFrame:
        return _metrics(0.6 if worlds.name == "D1-D2-D3" else 0.8)

    monkeypatch.setattr(cli, "evaluate_rotation", evaluates)
    cli.evaluate_command(TWO_ROTATIONS, cli.connect)
    written = sorted(path.name for path in paths.results_dir(cli.ROTATIONS_RUN).glob("*"))
    assert written == ["rotation_spread_development.parquet"]
    spread = pd.read_parquet(paths.results_dir(cli.ROTATIONS_RUN) / written[0])
    assert set(spread["rotations"]) == {2}
    assert shown[-1] == (
        "the spread over 2 development rotations is in "
        "results/rotations/rotation_spread_development.parquet"
    )


# The proof of concept (docs/design.md, "Proof-of-concept settings (25 October 2026)"): the sampled
# worlds and the trimmed training, on the pipeline's worlds with a weight of 10 for the reference
# worlds' few hundred accounts.


def test_the_proof_of_concept_trains_one_seed_and_evaluates_on_sampled_worlds(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    poc = replace(
        CONFIG,
        data=replace(
            CONFIG.data, scoring="fallback", fallback_weight=10, per_positive=1, min_unlabelled=8
        ),
        training=replace(CONFIG.training, ablations=False, fixed_prior=1.0),
    )
    for name in ("D1", "D2", "D3"):
        cli.prepare_command(poc, pipeline.connect(name))
    # The test world's fallback list is mule audit's: the true mules and a sample of the rest.
    graph = pipeline.graphs["D3"]
    times = poc.calendar.scoring_times(
        {world.name: world.seed for world in poc.worlds.worlds}["D3"]
    )
    populations = {t_ms: [row.account_id for row in list_population(graph, t_ms)] for t_ms in times}
    listed = fallback_keys(read_answer_key(Spy(graph)), populations, poc.data, times)
    Store(pipeline.identity("D3")).write_fallback(listed)
    assert 0 < len(listed) < sum(len(accounts) for accounts in populations.values())
    cli.train_command(poc, cli.connect)
    folder = paths.results_dir(cli.run_name(ROTATION))
    # The full model of the one seed and each XGBoost comparison, no ablation and no grid.
    assert {path.name for path in folder.glob("*.pt")} == {
        "gnn_seed0.pt",
        "xgboost_own.pt",
        "xgboost_graph.pt",
    }
    assert not (folder / "prior_decision.txt").exists()
    cli.evaluate_command(poc, cli.connect)
    metrics = pd.read_parquet(folder / "metrics.parquet")
    assert set(metrics["ranker"]) == {"gnn", "xgboost_own", "xgboost_graph"}
    assert metrics["development_only"].all()
    per_week = pd.read_parquet(folder / "per_week.parquet")
    assert set(per_week["ranker"]) == {"gnn", "xgboost_own", "xgboost_graph"}


# mule score.


def test_score_ranks_the_accounts_asked_with_the_best_selection_seed(
    pipeline: Pipeline, evaluated: Path, shown: list[str]
) -> None:
    config = replace(
        CONFIG,
        score=ScoreConfig(("a_mule", "a_h1", "a_collector"), (date(2024, 7, 1),), "D1-D2-D3"),
    )
    cli.score_command(config, pipeline.connect("D3"))
    scored = pd.read_parquet(evaluated / "scores.parquet")
    assert sorted(scored["account_id"]) == ["a_collector", "a_h1", "a_mule"]
    assert set(scored["date"].astype(str)) == {"2024-07-01"} and set(scored["t_ms"]) == {T}
    assert sorted(scored["rank"]) == [1, 2, 3]
    assert any("scored with the seed 0 model" in line for line in shown)


def test_score_takes_any_date_at_midnight_not_only_a_monday(
    pipeline: Pipeline, evaluated: Path
) -> None:
    # A Wednesday and a Sunday, each scored at 00:00 UTC.
    days = (date(2024, 7, 3), date(2024, 7, 7))
    config = replace(CONFIG, score=ScoreConfig(("a_mule", "a_h1"), days, "D1-D2-D3"))
    cli.score_command(config, pipeline.connect("D3"))
    scored = pd.read_parquet(evaluated / "scores.parquet")
    assert set(scored["date"].astype(str)) == {"2024-07-03", "2024-07-07"}
    assert set(scored["t_ms"]) == {T + 2 * DAY_MS, T + 6 * DAY_MS}
    assert sorted(scored["account_id"]) == ["a_h1", "a_h1", "a_mule", "a_mule"]


def test_score_matches_no_world_so_a_graph_without_an_identity_file_is_scored(
    pipeline: Pipeline, evaluated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A bank's graph has no PhantomLedger identity file: the records folder is empty, and
    # read_world never runs.
    monkeypatch.setenv("MULE_PL_RECORDS_DIR", str(tmp_path))
    spy = Live(pipeline.graphs["D3"])
    config = replace(
        CONFIG, score=ScoreConfig(("a_mule", "a_h1"), (date(2024, 7, 1),), "D1-D2-D3")
    )
    cli.score_command(config, lambda: spy)
    assert "read_world" not in set(spy.names())
    scored = pd.read_parquet(evaluated / "scores.parquet")
    assert sorted(scored["account_id"]) == ["a_h1", "a_mule"]
    assert sorted(scored["rank"]) == [1, 2]


def test_score_needs_its_accounts_dates_and_run_and_a_trained_run(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(cli.CommandError, match="set score.accounts, score.dates and score.run"):
        cli.score_command(CONFIG, pipeline.connect("D3"))
    config = replace(CONFIG, score=ScoreConfig(("a_mule",), (date(2024, 7, 1),), "A-B-C"))
    with pytest.raises(cli.CommandError, match="holds no model file of the full model"):
        cli.score_command(config, pipeline.connect("D3"))


# The other commands, on what they call.


def test_install_says_what_it_installed(monkeypatch: pytest.MonkeyPatch, shown: list[str]) -> None:
    def installs(executor: object) -> list[str]:
        return ["read_hubs", "read_world"]

    def installs_nothing(executor: object) -> list[str]:
        return []

    monkeypatch.setattr(installer, "install", installs)
    cli.install_command(CONFIG, lambda: Live(ReferenceGraph(WorldBuilder().tables())))
    assert "installed 2 queries in" in shown[0] and shown[0].endswith("read_hubs, read_world")
    monkeypatch.setattr(installer, "install", installs_nothing)
    cli.install_command(CONFIG, lambda: Live(ReferenceGraph(WorldBuilder().tables())))
    assert shown[-1] == "every query is current: nothing to install"


def _parity(failed: tuple[str, ...]) -> Parity:
    queries = {
        name: QueryParity(name, VERSIONS[name], checks=3, mismatched=int(name in failed))
        for name in QUERY_FILES
    }
    return Parity("0" * 32, "4.2.5", Path("/exports/x"), (T,), queries, Communities(10, 2, 1))


def test_parity_reports_a_query_that_did_not_match(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, shown: list[str]
) -> None:

    seeds: list[int] = []

    def matches(executor: object, config: object, *, seed: int) -> Parity:
        seeds.append(seed)
        return _parity(())

    def differs(executor: object, config: object, *, seed: int) -> Parity:
        return _parity(("read_hubs",))

    monkeypatch.setattr(cli.parity, "run", matches)
    cli.parity_command(CONFIG, pipeline.connect("D1"))
    assert shown[-1].startswith("every query matched the reference graph")
    # The run is given the configured seed of the loaded world, D1's, which names its times.
    assert seeds == [{world.name: world.seed for world in CONFIG.worlds.worlds}["D1"]]
    monkeypatch.setattr(cli.parity, "run", differs)
    with pytest.raises(cli.CommandError, match=r"1 query did not match .*: read_hubs; .* the 11"):
        cli.parity_command(CONFIG, pipeline.connect("D1"))
    assert any(line.startswith("read_hubs: FAILED") for line in shown)


def test_audit_fails_the_command_with_the_checks_that_failed(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, shown: list[str]
) -> None:
    record = AuditRecord(
        "D1", "development", "0" * 32, "c" * 32, "s" * 32, {}, {"a": True, "b": False}
    )

    def fails(executor: object, config: object) -> auditing.Audited:
        return auditing.Audited(record, {"b": ("why",)})

    def passes(executor: object, config: object) -> auditing.Audited:
        return auditing.Audited(replace(record, checks={"a": True}), {})

    monkeypatch.setattr(cli.auditing, "audit", fails)
    with pytest.raises(cli.CommandError, match=r"1 check failed: b; the pass record was written"):
        cli.audit_command(CONFIG, pipeline.connect("D1"))
    assert "  FAILED  b" in shown and "  b: why" in shown
    monkeypatch.setattr(cli.auditing, "audit", passes)
    cli.audit_command(CONFIG, pipeline.connect("D1"))
    assert "  passed  a" in shown


def test_bench_prints_the_headline_numbers_and_where_its_table_is(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, shown: list[str], tmp_path: Path
) -> None:
    table = pd.DataFrame(
        [("bootstrap", "projected_seconds", "people=30000", 12.5, "s", "")],
        columns=["section", "name", "variant", "value", "unit", "note"],
    )
    loaded = cli.check_ready(Live(pipeline.graphs["D1"]), CONFIG)

    def benches(executor: object, config: object) -> tuple[LoadedWorld, pd.DataFrame, Path]:
        return loaded, table, tmp_path / "D1.parquet"

    monkeypatch.setattr(cli.bench, "run", benches)
    cli.bench_command(CONFIG, pipeline.connect("D1"))
    assert "bootstrap, seconds, people=30000: 12.5" in shown
    assert shown[-1] == f"the bench table of D1 is in {tmp_path / 'D1.parquet'}"


def test_a_prepared_world_is_registered_once_more_when_prepared_again(pipeline: Pipeline) -> None:
    before = registered("D1")
    cli.prepare_command(CONFIG, pipeline.connect("D1"))
    assert registered("D1") == before
    # The registry holds all three worlds, each its own.
    assert len({registered(name).identity for name in ("D1", "D2", "D3")}) == 3
