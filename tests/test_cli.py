"""mule has the design's nine commands, and each is a stub that exits with status 1."""

from __future__ import annotations

import subprocess
import sys
import tomllib

import pytest

from mule_pattern_learner import cli
from mule_pattern_learner.paths import REPOSITORY_ROOT


def test_mule_is_the_command_line_with_the_designs_commands() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
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
    assert tuple(cli.COMMANDS) == names


@pytest.mark.parametrize("command", list(cli.COMMANDS))
def test_each_command_says_it_is_not_built_yet(command: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main([command])
    assert stopped.value.code == (
        f"mule {command} is not built yet: the code is being rebuilt to docs/design.md"
    )


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main([])
    assert stopped.value.code == 2 and "COMMAND" in capsys.readouterr().err


def test_python_dash_m_runs_the_same_command_line() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mule_pattern_learner", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(command in result.stdout for command in cli.COMMANDS)
    stub = subprocess.run(
        [sys.executable, "-m", "mule_pattern_learner", "train"], capture_output=True, text=True
    )
    assert stub.returncode == 1 and "mule train is not built yet" in stub.stderr
