"""The import contracts of pyproject.toml hold, so the layers of the package run one way."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from mule_pattern_learner.paths import REPOSITORY_ROOT


def test_the_import_contracts_hold() -> None:
    # lint-imports sits beside the interpreter that runs the tests, in .venv/bin.
    lint = Path(sys.executable).parent / "lint-imports"
    result = subprocess.run(
        [str(lint), "--no-cache"], cwd=REPOSITORY_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
