"""The command line: entry points, the cuBLAS workspace, subcommands and defaults."""

from __future__ import annotations

from importlib.metadata import distribution
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


def test_python_m_runs_the_command_line() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mule_pattern_learner", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "train" in result.stdout


def test_mule_and_mule_temporal_run_the_same_main() -> None:
    scripts = {
        e.name: e.value
        for e in distribution("mule-pattern-learner").entry_points
        if e.group == "console_scripts"
    }
    # A missing name means the installed metadata is stale: rerun pip install -e '.[dev]'.
    assert scripts["mule"] == scripts["mule-temporal"] == "mule_pattern_learner.cli:main"


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


def test_cli_needs_no_config_truth_or_dataset() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train"])
    # The dataset is the built-in run's own, in data/.
    assert not hasattr(args, "dataset")
    # The settings are built in: no command reads a configuration file.
    for command in (["train"], ["prepare"]):
        with pytest.raises(SystemExit):
            parser.parse_args([*command, "--config", "overrides.toml"])
    final = parser.parse_args(["evaluate-final"])
    assert final.dataset is None and final.truth is None
    # The audit goes into the baseline run, or the run directory named.
    assert final.run == pipeline_train.BASELINE_RUN.root
    assert parser.parse_args(["evaluate-final", "results/x/seed-1"]).run == Path("results/x/seed-1")
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
    connected: list[object] = []
    monkeypatch.setattr(cli, "connect", lambda transport: connected.append(transport))
    for argv, expected in (
        (["install"], {"include_optional": False, "force": False}),
        (["install", "--force", "--include-optional"], {"include_optional": True, "force": True}),
    ):
        monkeypatch.setattr(sys, "argv", ["mule-temporal", *argv])
        cli.main()
        assert calls[-1] == expected
    assert capsys.readouterr().out.count('"installed": []') == 2
    # The built-in run's retry budgets.
    assert connected == [DEFAULT_CONFIG.transport] * 2


def test_train_command_prepares_then_trains_or_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared: list[tuple[RunConfig, Path]] = []
    trained: list[tuple[RunConfig, DatasetPaths, RunPaths, dict[str, Any]]] = []
    dataset = DatasetPaths.of("id", tmp_path / "data")

    def prepare(c: RunConfig, data: Path) -> DatasetPaths:
        prepared.append((c, data))
        return dataset

    # `mule-temporal train` is pipeline.run with resume: patch the pipeline's steps.
    monkeypatch.setattr(pipeline_train, "prepare_dataset", prepare)

    def train(c: RunConfig, d: DatasetPaths, o: RunPaths, **kwargs: Any) -> dict[str, Any]:
        trained.append((c, d, o, kwargs))
        return {}

    monkeypatch.setattr(pipeline_train, "train", train)
    cli.train_command()
    # One command prepares the built-in run's dataset in data/, then trains it into
    # results/baseline/seed-42/ (resuming if interrupted).
    assert prepared[-1] == (DEFAULT_CONFIG, DATA_DIR)
    c, d, o, kwargs = trained[-1]
    assert c is DEFAULT_CONFIG and d == dataset and o == pipeline_train.BASELINE_RUN
    # The trainer opens the live source through the pipeline once its checks passed.
    assert kwargs == {"open_contexts": open_context_source, "resume": True}
