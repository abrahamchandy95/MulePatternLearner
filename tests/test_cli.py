"""The command line: one entry point, the cuBLAS workspace, seven commands without options."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from mule_pattern_learner import cli
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.paths import DATA_DIR, REPOSITORY_ROOT, DatasetPaths, RunPaths
from mule_pattern_learner.pipeline import train as pipeline_train
from mule_pattern_learner.pipeline.connect import open_context_source

COMMANDS = ("install", "train", "evaluate", "score", "report", "diagnose", "check")


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


def test_each_command_runs_its_use_case_and_prints_one_json_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def use_case(name: str, result: dict[str, Any]) -> Any:
        def run(*args: Any) -> dict[str, Any]:
            calls.append((name, args))
            return result

        return run

    monkeypatch.setattr(cli, "install_queries", use_case("install", {"installed": []}))
    monkeypatch.setattr(cli, "evaluate_run", use_case("evaluate", {"metrics": {}}))
    monkeypatch.setattr(cli, "score_accounts", use_case("score", {"accounts": 2}))
    monkeypatch.setattr(cli, "report_directory", use_case("report", {"figures": []}))
    monkeypatch.setattr(cli, "check", use_case("check", {"status": "ready"}))
    monkeypatch.setattr(cli, "diagnose_built_in", use_case("diagnose", {"status": "complete"}))
    commands = (["install"], ["evaluate"], ["score", "new.txt"], ["report"], ["check"])
    for argv in (*commands, ["diagnose"], ["diagnose", "drift"]):
        monkeypatch.setattr(sys, "argv", ["mule", *argv])
        cli.main()
        json.loads(capsys.readouterr().out)
    assert calls == [
        ("install", ()),
        ("evaluate", (pipeline_train.BASELINE_RUN,)),
        ("score", (pipeline_train.BASELINE_RUN, Path("new.txt"), None)),
        ("report", (pipeline_train.BASELINE_RUN.root,)),
        ("check", ()),
        ("diagnose", (None,)),
        ("diagnose", ("drift",)),
    ]
    # A graph that is not ready, or a study missing an analysis' inputs, is a failure,
    # after the result is printed.
    monkeypatch.setattr(cli, "check", use_case("check", {"status": "not_ready"}))
    monkeypatch.setattr(cli, "diagnose_built_in", use_case("diagnose", {"status": "incomplete"}))
    for command, status in (("check", "not_ready"), ("diagnose", "incomplete")):
        monkeypatch.setattr(sys, "argv", ["mule", command])
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 1
        assert json.loads(capsys.readouterr().out) == {"status": status}


def test_train_prepares_then_trains_or_resumes_the_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
        return {"status": "complete"}

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
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}
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
    # A complete run prints its recorded result, and nothing is prepared, trained or drawn.
    recorded[0] = {"status": "complete", "best_epoch": 3}
    cli.main()
    assert json.loads(capsys.readouterr().out) == recorded[0]
    assert len(checked) == len(prepared) == len(trained) == len(reported) == 1
