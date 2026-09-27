"""The installed package runs its command line as `mule` and `python -m mule_pattern_learner`."""

from __future__ import annotations

from importlib.metadata import distribution
import subprocess
import sys


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
    assert (
        scripts["mule"] == scripts["mule-temporal"] == "mule_pattern_learner.temporal.live.cli:main"
    )
