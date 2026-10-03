"""The command line: one entry point, the cuBLAS workspace, seven commands without options."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

from mule_pattern_learner import cli
from mule_pattern_learner.artifacts import (
    EPOCH_COLUMNS,
    read_epochs,
    read_events,
    read_history,
    read_json,
)
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig, TransportConfig
from mule_pattern_learner.paths import (
    DATA_DIR,
    REPOSITORY_ROOT,
    DatasetPaths,
    RunPaths,
    command_events,
)
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import prepare as pipeline_prepare
from mule_pattern_learner.pipeline import train as pipeline_train
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.runtime.progress import emit, recording
from mule_pattern_learner.testing.builders import (
    neighbourhood,
    recorded_events,
    scope_population,
)
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.executor import (
    TigerGraphUnavailableError,
    TransientQueryError,
)

COMMANDS = ("install", "train", "evaluate", "score", "report", "diagnose", "check")


@pytest.fixture(autouse=True)
def command_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The results directory of the commands these tests run: tmp_path's, never results/.

    A command records there, in events.jsonl, what it emits before a run, a dataset or a
    study records its events.
    """
    results = tmp_path / "command_results"
    monkeypatch.setattr(cli, "RESULTS_DIR", results)
    return results


def test_python_m_runs_the_command_line() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mule_pattern_learner", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(command in result.stdout for command in COMMANDS)


def test_mule_is_the_one_console_script() -> None:
    scripts = {
        e.name: e.value
        for e in distribution("mule-pattern-learner").entry_points
        if e.group == "console_scripts"
    }
    # Another name means the installed metadata is stale: rerun pip install -e '.[dev]'.
    assert scripts == {"mule": "mule_pattern_learner.cli:main"}


def test_cli_reserves_the_cublas_workspace_first_and_keeps_user_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Importing the command line leaves the environment alone.
    code = (
        "import os\nimport mule_pattern_learner.cli\nprint('CUBLAS_WORKSPACE_CONFIG' in os.environ)"
    )
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    env["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
    # main reserves it before it parses anything, and an explicit user value wins.
    monkeypatch.setattr(sys, "argv", ["mule", "--help"])
    for value, expected in ((None, ":4096:8"), (":16:8", ":16:8")):
        if value is None:
            monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        else:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
        with pytest.raises(SystemExit):
            cli.main()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == expected


def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    (action,) = (a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # pyright: ignore[reportPrivateUsage]
    return dict(action.choices)  # pyright: ignore[reportUnknownArgumentType]


def test_the_commands_take_no_options_and_default_to_the_baseline_run() -> None:
    parser = cli.build_parser()
    commands = subcommands(parser)
    assert tuple(commands) == COMMANDS
    for name, command in commands.items():
        # Only --help: every input is a built-in setting or a positional argument.
        options = [o for a in command._actions for o in a.option_strings]  # pyright: ignore[reportPrivateUsage]
        assert options == ["-h", "--help"], name
    assert vars(parser.parse_args(["train"])) == {"command": "train"}
    for retired in (["train", "--config", "x.toml"], ["prepare"], ["evaluate-final"]):
        with pytest.raises(SystemExit):
            parser.parse_args(retired)
    # RUN is the baseline run unless a run directory is named.
    assert parser.parse_args(["evaluate"]).run == pipeline_train.BASELINE_RUN
    named = parser.parse_args(["evaluate", "results/x/seed-1"])
    assert named.run == RunPaths(Path("results/x/seed-1"))
    scoring = parser.parse_args(["score", "new.txt"])
    assert (scoring.accounts, scoring.date) == (Path("new.txt"), None)
    assert parser.parse_args(["score", "new.txt", "2025-02-01"]).date == "2025-02-01"
    # report takes a run's directory, a suite's or a study's.
    assert parser.parse_args(["report"]).directory == pipeline_train.BASELINE_RUN.root
    reported = parser.parse_args(["report", "results/experiments/controls"]).directory
    assert reported == Path("results/experiments/controls")
    # diagnose runs every analysis, or the one named.
    assert parser.parse_args(["diagnose"]).analysis is None
    assert parser.parse_args(["diagnose", "learning-curve"]).analysis == "learning-curve"
    with pytest.raises(SystemExit):
        parser.parse_args(["diagnose", "no_graph"])


# Results as the use cases return them, small but complete enough for each summary.
RUN = pipeline_train.BASELINE_RUN
METRICS: dict[str, Any] = {
    "status": "complete",
    "best_epoch": 5,
    "elapsed_seconds": 8640.0,
    "observed_label_proxy": {
        split: {
            "n": 2000 + positives,
            "positives": positives,
            "average_precision": ap,
            "roc_auc": auc,
            "recall_at_1pct": recall,
        }
        for split, positives, ap, auc, recall in (
            ("validation", 11, 0.5661, 0.9592, 0.4545),
            ("test", 20, 0.512, 0.948, 0.4),
        )
    },
}


def audit_report(split: str, purpose: str, mules: int, ap: float) -> dict[str, Any]:
    ranking = {"average_precision": ap, "roc_auc": 0.941}
    for kind, values in (("recall", (0.196, 0.451, 0.608)), ("precision", (0.352, 0.181, 0.122))):
        ranking |= {f"{kind}_at_{n}pct": v for n, v in zip((1, 5, 10), values, strict=True)}
    return {
        "split": split,
        "purpose": purpose,
        "metrics": {"sample_positives": mules, **ranking, "threshold": 0.5},
        "intervals": {name: [value - 0.05, value + 0.05] for name, value in ranking.items()},
        "constants": {"interval": 0.9},
    }


AUDITS = {
    "validation": audit_report("validation", "decisions", 51, 0.3121),
    "test": audit_report("test", "reporting", 62, 0.2984),
}
CHECKED: dict[str, Any] = {
    "graph": "Mule_Pattern_Learner",
    "scope_schema": "present",
    "queries": {"up_to_date": ["q1", "q2", "q3"], "stale": {}, "retired": ["old_q1"]},
    "cugraph": {"status": "passed", "device": "cuda:0", "reason": None},
    "dataset": "1a2b3c4d5e6f7a8b9c",
    "first_step": {
        "status": "passed",
        "device": "cuda",
        "roots": 64,
        "accepted_roots": 64,
        "context_requests": 128,
        "batch_seconds": 2.1,
        "loss": 0.69314718,
        "objective": 0.69314718,
        "tensor_digests": {"x": {"sha256": "0" * 64}},
        "batch_digest": "9f3c2a1b7e4d" + "0" * 52,
    },
    "problems": [],
    "status": "ready",
    "graph_writes": 0,
}
NOT_READY: dict[str, Any] = {
    "graph": "Mule_Pattern_Learner",
    "scope_schema": "missing",
    "queries": {
        "up_to_date": ["q1", "q2"],
        "stale": {"q3": ["differs from repository source"]},
        "retired": [],
    },
    "cugraph": {"status": "no_cuda"},
    "dataset": None,
    "problems": [
        "the scope vertex type is missing; `mule install` adds it",
        "training queries differ from the repository; `mule install` installs them",
        "found 0 datasets of the built-in run in data; `mule train` prepares one",
    ],
    "status": "not_ready",
    "graph_writes": 0,
}
STUDY: dict[str, Any] = {
    "status": "complete",
    "directory": str(REPOSITORY_ROOT / "results" / "diagnostics" / "1a2b3c4d5e6f7a8b9c"),
    "dataset_id": "1a2b3c4d5e6f7a8b9c",
    "run": "baseline/seed-42",
    "analyses": {},
}
INSTALLED: dict[str, Any] = {
    "installed": ["q1", "q2"],
    "up_to_date": ["q3"],
    "verified": ["q1", "q2", "q3"],
    "dropped": ["old_q1"],
    "not_defined": ["someone_elses"],
}
SCORED: dict[str, Any] = {
    "accounts": 1234,
    "rejected": 3,
    "rejected_roots_by_status": {"missing_entity": 2, "history_capacity_exceeded": 1},
    "rejected_children": 7,
    "output": str(RUN.scores("new", "2025-02-01")),
    "rejected_output": str(RUN.scores_rejected("new", "2025-02-01")),
}
DRAWN: dict[str, Any] = {
    "report": str(RUN.report),
    "figures": [str(RUN.figure(name)) for name in ("training_objective", "audit_roc")],
}
# What each command shows, its summary only: the use cases are replaced.
SUMMARIES = {
    "install": """\
3 training queries installed with the repository's text, 2 of them now; 1 retired query dropped
Left in place, since no repository file defines them: someone_elses
""",
    "train": """\
Trained in 2.4 h; model.pt holds the weights of the best epoch, 5.
Proxy metrics, on the revealed labels at the selected epoch:
              known mules     AP  ROC AUC  recall at 1%
  validation           11  0.566    0.959         0.455
  test                 20  0.512    0.948         0.400
These count unlabelled accounts as negatives; `mule evaluate` gives the ground-truth audit.
Run directory: results/baseline/seed-42/
  model.pt, metrics.json, history.csv, epochs.csv, events.jsonl, predictions/, plots/, report.md
""",
}


def test_each_command_runs_its_use_case_and_shows_a_short_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def use_case(name: str, result: dict[str, Any]) -> Any:
        def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((name, args))
            return result

        return run

    monkeypatch.setattr(cli, "install_queries", use_case("install", INSTALLED))
    monkeypatch.setattr(cli, "train_run", use_case("train", METRICS))
    monkeypatch.setattr(cli, "evaluate_run", use_case("evaluate", AUDITS))
    monkeypatch.setattr(cli, "score_accounts", use_case("score", SCORED))
    monkeypatch.setattr(cli, "report_directory", use_case("report", DRAWN))
    monkeypatch.setattr(cli, "check", use_case("check", CHECKED))
    monkeypatch.setattr(cli, "diagnose_built_in", use_case("diagnose", STUDY))
    shown: dict[str, str] = {}
    commands = (["install"], ["train"], ["evaluate"], ["score", "new.txt"], ["report"])
    for argv in (*commands, ["check"], ["diagnose"], ["diagnose", "drift"]):
        monkeypatch.setattr(sys, "argv", ["mule", *argv])
        cli.main()
        shown[argv[0]] = capsys.readouterr().out
    assert calls == [
        ("install", ()),
        ("train", ()),
        ("evaluate", (RUN,)),
        ("score", (RUN, Path("new.txt"), None)),
        ("report", (RUN.root,)),
        ("check", ()),
        ("diagnose", (None,)),
        ("diagnose", ("drift",)),
    ]
    assert shown["install"] == SUMMARIES["install"]
    assert shown["train"] == SUMMARIES["train"]
    assert shown["evaluate"] == EVALUATED
    assert shown["score"] == SCORE_SUMMARY
    assert shown["report"] == (
        "Drew 2 figures into results/baseline/seed-42/plots/ and wrote "
        "results/baseline/seed-42/report.md\n"
    )
    assert shown["check"] == CHECK_SUMMARY
    assert shown["diagnose"] == (
        "Study complete, of dataset 1a2b3c4d5e6f and run baseline/seed-42\n"
        "Study directory: results/diagnostics/1a2b3c4d5e6f7a8b9c/\n"
        "  study.json, features.parquet, <analysis>.csv, events.jsonl, plots/, report.md\n"
    )
    # No command prints a record any more.
    assert not any(line.startswith("{") for out in shown.values() for line in out.splitlines())


EVALUATED = """\
Ground-truth audit of results/baseline/seed-42, with 90% intervals:
                    validation (decisions)      test (reporting)
  mules                                 51                    62
  AP                  0.312 [0.262, 0.362]  0.298 [0.248, 0.348]
  ROC AUC             0.941 [0.891, 0.991]  0.941 [0.891, 0.991]
  recall at 1%        0.196 [0.146, 0.246]  0.196 [0.146, 0.246]
  recall at 5%        0.451 [0.401, 0.501]  0.451 [0.401, 0.501]
  recall at 10%       0.608 [0.558, 0.658]  0.608 [0.558, 0.658]
  precision at 1%     0.352 [0.302, 0.402]  0.352 [0.302, 0.402]
  precision at 5%     0.181 [0.131, 0.231]  0.181 [0.131, 0.231]
  precision at 10%    0.122 [0.072, 0.172]  0.122 [0.072, 0.172]
Decide on validation; the test audit is for reporting only.
Files: results/baseline/seed-42/audit/ (reports and scored samples), plots/ and report.md
"""
SCORE_SUMMARY = """\
Scored 1,234 accounts into results/baseline/seed-42/scores/new_2025-02-01.parquet
TigerGraph rejected 3 accounts (missing_entity 2, history_capacity_exceeded 1), listed in \
results/baseline/seed-42/scores/new_2025-02-01_rejected.txt
7 child contexts TigerGraph rejected were left out of the scored accounts' neighbourhoods
"""
CHECK_SUMMARY = """\
Readiness of graph Mule_Pattern_Learner, which mule check only reads:
  [x] scope vertex type present
  [x] 3 training queries installed with the repository's text
  [-] retired queries still installed, which `mule install` drops: old_q1
  [x] cuGraph probe passed on cuda:0
  [x] dataset 1a2b3c4d5e6f prepared
  [x] first training batch on cuda: 64 of 64 roots, 128 context requests, 2 s; one step: \
loss 0.693147, objective 0.693147
      batch digest 9f3c2a1b7e4d
Ready to train.
The full report, with every tensor's digest, is in results/check.json
"""
NOT_READY_SUMMARY = """\
Readiness of graph Mule_Pattern_Learner, which mule check only reads:
  [ ] scope vertex type missing
  [ ] 1 of 3 training queries are stale: q3 (differs from repository source)
  [-] no CUDA device: training samples with the torch sampler
  [ ] no prepared dataset of the built-in run; `mule train` prepares one
  [ ] first training batch: built once everything above is ready
Not ready: run `mule install`, then `mule train`.
The full report, with every tensor's digest, is in results/check.json
"""


def test_a_graph_not_ready_or_a_study_incomplete_fails_after_its_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    incomplete = STUDY | {
        "status": "incomplete",
        "skipped": {"subgroups": "no audit", "proxy-validity": "no metrics.json"},
    }

    def not_ready() -> dict[str, Any]:
        return NOT_READY

    def diagnosed(analysis: str | None) -> dict[str, Any]:
        return incomplete

    monkeypatch.setattr(cli, "check", not_ready)
    monkeypatch.setattr(cli, "diagnose_built_in", diagnosed)
    shown: dict[str, str] = {}
    for command in ("check", "diagnose"):
        monkeypatch.setattr(sys, "argv", ["mule", command])
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 1
        shown[command] = capsys.readouterr().out
    assert shown["check"] == NOT_READY_SUMMARY
    assert shown["diagnose"].startswith(
        "Study incomplete: 2 analyses skipped (subgroups, proxy-validity); the reasons are "
        "above and in study.json\n"
    )


def test_retries_that_run_out_end_the_command_with_one_clear_line_on_stderr(
    monkeypatch: pytest.MonkeyPatch, command_records: Path
) -> None:
    screen = Terminal()
    monkeypatch.setattr(sys, "stdout", screen)
    error = TigerGraphUnavailableError(
        "connect failed after 31 attempt(s) (TigerGraph unavailable for 1800s (max_outage_s "
        "= 1800)): TigerGraphException: starting workspace"
    )

    def unavailable(**kwargs: Any) -> dict[str, Any]:
        # A step's progress is on the screen when the outage ends the run.
        emit({"event": "train", "epoch": 1, "step": 3, "steps": 9, "loss": 0.5} | STEP_TIME)
        raise error

    monkeypatch.setattr(cli, "train_run", unavailable)
    monkeypatch.setattr(sys, "argv", ["mule", "train"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    # Python writes a SystemExit's text to stderr and exits 1.
    assert stopped.value.code == (
        f"mule train stopped: TigerGraph stayed unavailable. {error}. Run it again once "
        "TigerGraph answers; an interrupted run resumes where it stopped."
    )
    # The progress line ends first, so the message starts a line of its own.
    assert screen.getvalue() == "\repoch 1  step 3/9  loss 0.500  2.0 s/step\n"
    # No run recorded the step here, so the command's own events.jsonl did, and then
    # the error that stopped the command, which stderr shows.
    recorded = read_events(command_events(command_records))
    assert [e["event"] for e in recorded] == ["train", "command_stopped"]
    assert recorded[-1]["command"] == "train"
    assert recorded[-1]["error"] == f"TigerGraphUnavailableError: {error}"


def test_the_error_that_stops_a_command_is_recorded_where_its_records_were_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command_records: Path
) -> None:
    run = RunPaths(tmp_path / "run")
    run.root.mkdir()
    stop = TransientQueryError(
        "fetch_training_context (512 keys) failed after 2 attempts (suspected deterministic "
        "failure, retried once): TigerGraphException: out of memory"
    )
    bug = ValueError("a bug's message, longer than an error's summary keeps: " + "x" * 300)
    raising: list[Exception] = [stop]

    def evaluate(audited: RunPaths) -> dict[str, Any]:
        # The run's events.jsonl is recording when the error is raised.
        with recording(audited.events):
            raise raising[0]

    monkeypatch.setattr(cli, "evaluate_run", evaluate)
    monkeypatch.setattr(sys, "argv", ["mule", "evaluate", str(run.root)])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert str(stopped.value.code).startswith("mule evaluate stopped: a TigerGraph request")
    # A bug is raised as it was, so Python shows its traceback.
    raising[0] = bug
    with pytest.raises(ValueError) as raised:
        cli.main()
    assert raised.value is bug
    # Both are in the run's events.jsonl, after the time and the command, beside what led
    # to them; results/events.jsonl has nothing, since the run was recording.
    assert recorded_events(run.events) == [
        {
            "command": "evaluate",
            "event": "command_stopped",
            "error": f"TransientQueryError: {stop}",
        },
        {
            "command": "evaluate",
            "event": "command_failed",
            "error": f"ValueError: {str(bug)[:200]}",
            "type": "ValueError",
            "message": str(bug),
        },
    ]
    assert not command_events(command_records).exists()
    # A type outside the builtins is named with its module, as a traceback names it.
    undecodable = json.JSONDecodeError("Expecting value", "<html>", 0)
    assert cli.error_record(undecodable)["type"] == "json.decoder.JSONDecodeError"


STEP_TIME = {"seconds_per_step": 2.0}


def test_the_line_of_a_stopped_command_ends_in_one_full_stop() -> None:
    # TigerGraph's words may end in a full stop of their own, or be cut with an ellipsis.
    ended = TransientQueryError("q failed after 2 attempt(s): TigerGraphException: Halted.")
    assert cli.stopped("mule train", ended) == (
        "mule train stopped: a TigerGraph request kept failing. q failed after 2 attempt(s): "
        "TigerGraphException: Halted."
    )
    cut = TigerGraphUnavailableError("connect failed after 3 attempt(s): ReadTimeout: word wo...")
    assert cli.stopped("mule check", cut) == (
        "mule check stopped: TigerGraph stayed unavailable. connect failed after 3 "
        "attempt(s): ReadTimeout: word wo... Run it again once TigerGraph answers; an "
        "interrupted run resumes where it stopped."
    )


class Terminal(io.StringIO):
    """A stdout that says it is a terminal."""

    def isatty(self) -> bool:
        return True


def test_train_prepares_then_trains_or_resumes_the_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    prepared: list[tuple[RunConfig, Path]] = []
    trained: list[tuple[RunConfig, DatasetPaths, RunPaths, dict[str, Any]]] = []
    dataset = DatasetPaths.of("id", tmp_path / "data")

    def prepare(c: RunConfig, data: Path, *, session: object) -> DatasetPaths:
        # `mule train` connects on its own.
        assert session is None
        prepared.append((c, data))
        return dataset

    # `mule train` is pipeline.train.train_run with resume: patch the pipeline's steps,
    # and its checks of a complete or interrupted run, so a local results/baseline/seed-42
    # is not read.
    recorded: list[dict[str, Any] | None] = [None]
    checked: list[RunPaths] = []

    def completed_run(c: RunConfig, run: RunPaths) -> dict[str, Any] | None:
        assert c is DEFAULT_CONFIG and run == pipeline_train.BASELINE_RUN
        return recorded[0]

    def check_resumable(c: RunConfig, run: RunPaths) -> None:
        assert c is DEFAULT_CONFIG
        checked.append(run)

    monkeypatch.setattr(pipeline_train, "completed_run", completed_run)
    monkeypatch.setattr(pipeline_train, "check_resumable", check_resumable)
    monkeypatch.setattr(pipeline_train, "prepare_dataset", prepare)

    def train(c: RunConfig, d: DatasetPaths, o: RunPaths, **kwargs: Any) -> dict[str, Any]:
        trained.append((c, d, o, kwargs))
        return METRICS

    monkeypatch.setattr(pipeline_train, "train", train)
    # The training figures come after every other file of the run.
    reported: list[RunPaths] = []

    def write_training_report(run: RunPaths) -> dict[str, Any]:
        assert len(trained) == 1
        reported.append(run)
        return {}

    monkeypatch.setattr(pipeline_train, "write_training_report", write_training_report)
    monkeypatch.setattr(sys, "argv", ["mule", "train"])
    cli.main()
    assert capsys.readouterr().out == SUMMARIES["train"]
    assert reported == [pipeline_train.BASELINE_RUN]
    # One command checks that an interrupted run has the built-in settings, prepares the
    # built-in run's dataset in data/, then trains it into results/baseline/seed-42/
    # (resuming if interrupted).
    assert checked == [pipeline_train.BASELINE_RUN]
    assert prepared[-1] == (DEFAULT_CONFIG, DATA_DIR)
    c, d, o, kwargs = trained[-1]
    assert c is DEFAULT_CONFIG and d == dataset and o == pipeline_train.BASELINE_RUN
    # The trainer opens the source through the pipeline once its checks passed.
    assert kwargs == {"open_contexts": open_context_source, "resume": True}
    # A complete run is summarised from its metrics.json, saying so, and nothing is
    # prepared, trained or drawn.
    recorded[0] = METRICS
    cli.main()
    assert capsys.readouterr().out == (
        "results/baseline/seed-42 is complete; its metrics.json is summarised below\n"
        + SUMMARIES["train"]
    )
    assert len(checked) == len(prepared) == len(trained) == len(reported) == 1


def train_on_fakes(home: Path, monkeypatch: pytest.MonkeyPatch) -> RunPaths:
    """`mule train` of a small run on the fake graph, from home into home/results/.

    The fake graph's queries are installed and it holds the built-in run's scope with its
    known mules revealed, so preparation only reads. Every step is logged.
    """
    scope = DEFAULT_CONFIG.scope.id
    graph = FakeTigerGraph(
        factory=neighbourhood,
        hubs=[("N3", cutoff) for cutoff in (101, 102, 103)],
        population=scope_population(200),
        scopes={scope: {"ready": True, "source_id": "console_fixture", "split_seed": 42}},
    )

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        return graph

    def revealed(*args: Any) -> None:
        return None

    for module in (pipeline_prepare, pipeline_connect):
        monkeypatch.setattr(module, "connect", connect)
    monkeypatch.setattr(pipeline_prepare, "ensure_revealed_labels", revealed)
    config = DEFAULT_CONFIG.with_changes(
        {
            "dataset": {"seed_limits": {"train": 64, "validation": 24, "test": 24}},
            "training": {"epochs": 2, "steps_per_epoch": 3, "batch_size": 16},
            "runtime": {"device": "cpu", "threads": 1, "log_every_steps": 1},
        }
    )
    run = RunPaths.of("baseline", 42, home / "results")

    def train(resume: bool) -> dict[str, Any]:
        return pipeline_train.train_run(run, config=config, data=home / "data", resume=resume)

    # The command's run, results and repository are home's, so nothing outside tmp_path is
    # touched.
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", home)
    monkeypatch.setattr(cli, "RESULTS_DIR", home / "results")
    monkeypatch.setattr(cli, "BASELINE_RUN", run)
    monkeypatch.setattr(cli, "train_run", train)
    home.mkdir()
    monkeypatch.chdir(home)
    monkeypatch.setattr(sys, "argv", ["mule", "train"])
    cli.main()
    return run


def normalised(text: str) -> str:
    """Console text with the digits of its decimals as # and its seconds as "# s".

    Losses, metrics and timings vary between machines; their layout does not.
    """
    text = re.sub(r"\d+\.\d+", lambda found: re.sub(r"\d", "#", found.group()), text)
    return re.sub(r" +\d+ s\b", " # s", text)


def screen(raw: str) -> str:
    """What a terminal shows after raw: a carriage return goes back to the line's start."""
    lines: list[str] = []
    line: list[str] = []
    at = 0
    for char in raw:
        if char == "\r":
            at = 0
        elif char == "\n":
            lines.append("".join(line).rstrip())
            line, at = [], 0
        else:
            line[at : at + 1] = [char]
            at += 1
    return "".join(f"{text}\n" for text in [*lines, "".join(line).rstrip()] if text)


# `mule train` of the fakes' small run, written to a file or a pipe: no step lines.
TRAINED_ON_FAKES = """\
Dataset DATASET: 18 / 5 / 6 known mules in train / validation / test
Training on cpu (torch sampler) into results/baseline/seed-42: 3 steps per epoch, at most 2 \
epochs, early stop after 6 epochs without gain
epoch  1  loss #.###  validation AP #.###  ROC AUC #.### # s  best so far
epoch  2  loss #.###  validation AP #.###  ROC AUC #.### # s  best so far
Trained in # s; model.pt holds the weights of the best epoch, 2.
Proxy metrics, on the revealed labels at the selected epoch:
              known mules     AP  ROC AUC  recall at 1%
  validation            5  #.###    #.###         #.###
  test                  6  #.###    #.###         #.###
These count unlabelled accounts as negatives; `mule evaluate` gives the ground-truth audit.
Run directory: results/baseline/seed-42/
  model.pt, metrics.json, history.csv, epochs.csv, events.jsonl, predictions/, plots/, report.md
"""
# The fields every record of these events had before the console printed short lines; the
# run's events.jsonl keeps them all.
TOTALS = {
    "database_calls",
    "contexts_requested",
    "contexts_distinct",
    "memory_hits",
    "disk_hits",
    "rejections",
    "stub_children",
    "rejected_children",
    "sampler_backend",
    "elapsed_seconds",
}
RECORDED_FIELDS = {
    "start": {"event", "device", "threads", "deterministic", "known_mules", "loss", "run"}
    | {"epoch", "step", "steps", "prefetch_batches", "max_rejected_root_fraction"}
    | TOTALS,
    "train": {"event", "epoch", "step", "steps", "date", "loss", "objective", "corrected_steps"}
    | {"seconds_per_step", "batch_wait_seconds", "rejected_roots", "batch"}
    | TOTALS,
    "score": {"event", "split", "accounts", "total"} | TOTALS,
    "epoch": {"event", *EPOCH_COLUMNS} | TOTALS,
    "complete": {"event", "best_epoch"} | TOTALS,
}


def test_mule_train_shows_progress_and_a_summary_and_keeps_every_record_in_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = train_on_fakes(tmp_path / "plain", monkeypatch)
    plain = capsys.readouterr().out
    dataset = read_json(run.metrics)["dataset_id"][:12]
    assert normalised(plain) == TRAINED_ON_FAKES.replace("DATASET", dataset)
    # The run's events.jsonl has every record whole, scoring's progress included.
    recorded = read_events(run.events)
    assert [e["event"] for e in recorded if e["event"] != "score"] == [
        "start",
        *(["train"] * 3 + ["epoch"]) * 2,
        "complete",
    ]
    assert {e["event"] for e in recorded} == set(RECORDED_FIELDS)
    for record in recorded:
        assert RECORDED_FIELDS[record["event"]] <= set(record), record["event"]
    assert len(read_history(run.history)) == 6 and len(read_epochs(run.epochs)) == 2
    # The dataset's preparation recorded its own events; what came before it, the
    # install that found every query up to date, is in results/events.jsonl.
    prepared = DatasetPaths.of(read_json(run.metrics)["dataset_id"], tmp_path / "plain" / "data")
    assert [e["event"] for e in read_events(prepared.events)] == ["scope", "hubs", "dataset"]
    (install,) = read_events(command_events(tmp_path / "plain" / "results"))
    assert install["event"] == "install" and install["stale"] == [] and install["up_to_date"]
    # It names the command that wrote it, after the time.
    assert list(install)[:3] == ["time", "command", "event"] and install["command"] == "train"
    # On a terminal the same lines remain, and each step was shown in place before them.
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    train_on_fakes(tmp_path / "terminal", monkeypatch)
    raw = terminal.getvalue()
    monkeypatch.undo()
    assert normalised(screen(raw)) == normalised(plain)
    steps = [part.rstrip() for part in raw.split("\r") if re.match(r"epoch \d+  step", part)]
    assert [normalised(step) for step in steps] == [
        f"epoch {epoch}  step {step}/3  loss #.###  #.# s/step"
        for epoch in (1, 2)
        for step in (1, 2, 3)
    ]
    # Then, in place too, the scoring of validation after each epoch and of test at the
    # end, which the next line clears; a log has none of it.
    scoring = [part.rstrip() for part in raw.split("\r") if part.startswith("scoring")]
    assert scoring == [
        *["scoring validation 16/24", "scoring validation 24/24"] * 2,
        "scoring test 16/25",
        "scoring test 25/25",
    ]
    assert "scoring" not in plain
