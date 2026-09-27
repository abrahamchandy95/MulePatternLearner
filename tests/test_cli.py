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
from mule_pattern_learner.pipeline import train as pipeline_train
from mule_pattern_learner.testing.builders import base_config

ROOT = Path(__file__).resolve().parents[1]


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


def test_cli_sets_the_cublas_workspace_at_import_and_keeps_user_values() -> None:
    code = (
        "import os, sys\n"
        "import mule_pattern_learner.cli\n"
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
    source = (ROOT / "src/mule_pattern_learner/cli.py").read_text()
    assert source.index("CUBLAS_WORKSPACE_CONFIG") < source.index("\nimport argparse")


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
    monkeypatch.setattr(pipeline_train, "run_config", lambda path: dict(config))
    monkeypatch.setattr(pipeline_train, "prepare_live", prepare)
    monkeypatch.setattr(
        pipeline_train, "train", lambda c, d, o, *, resume: trained.append((c, d, o, resume)) or {}
    )
    output = tmp_path / "model.pt"
    cli.train_command(cli.build_parser().parse_args(["train", "--output", str(output)]))
    # One command prepares into the run directory, then trains (resuming if interrupted).
    assert prepared[-1][1] == tmp_path / "model_run" / "prepared"
    c, d, o, resume = trained[-1]
    assert c["dataset_id"] == "derived" and d == prepared[-1][1] and o == output and resume
