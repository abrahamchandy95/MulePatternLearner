"""The scripts import, show help without side effects, and run offline.

The live checks are integration tests now (tests/integration), and the reveal's
simulation is `mule diagnose reveal-spread`; two scripts remain: render_queries.py and
run_experiments.py.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from mule_pattern_learner.experiments import runner
from mule_pattern_learner.experiments.variants import SUITES, VARIANTS
from mule_pattern_learner.paths import REPOSITORY_ROOT

SCRIPTS = REPOSITORY_ROOT / "scripts"
# Every script; each must parse --help before connecting.
SCRIPT_NAMES = ("render_queries", "run_experiments")


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment_at_import(path: Path) -> list[int]:
    """Lines outside every function and class that read or change os.environ."""
    body = ast.parse(path.read_text()).body
    return [
        node.lineno
        for statement in body
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    ]


def test_no_module_or_script_reads_the_environment_when_imported() -> None:
    package = REPOSITORY_ROOT / "src/mule_pattern_learner"
    modules = [*package.rglob("*.py"), *SCRIPTS.glob("*.py")]
    assert {str(path): environment_at_import(path) for path in modules} == {
        str(path): [] for path in modules
    }
    assert {path.stem for path in SCRIPTS.glob("*.py")} == set(SCRIPT_NAMES)


def test_the_command_line_reserves_the_cublas_workspace_first() -> None:
    cli = REPOSITORY_ROOT / "src/mule_pattern_learner/cli.py"
    (main,) = (
        node
        for node in ast.parse(cli.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first = main.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert ast.unparse(first.value.func) == "reserve_deterministic_cublas"
    # The scripts never load torch, so they have no CUDA work to prepare for.
    paths = [str(SCRIPTS / f"{name}.py") for name in SCRIPT_NAMES]
    code = (
        "import importlib.util, sys\n"
        f"for path in {paths!r}:\n"
        "    spec = importlib.util.spec_from_file_location('script', path)\n"
        "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print('torch' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_scripts_import_and_print_help_without_connecting(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("--help must not connect to TigerGraph")

    monkeypatch.setattr(TigerGraphExecutor, "__init__", refuse)
    module = load(name)
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_the_experiments_script_lists_the_variants_and_runs_the_names_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load("run_experiments")
    monkeypatch.setattr(sys, "argv", ["run_experiments", "--help"])
    with pytest.raises(SystemExit):
        script.main()
    shown = capsys.readouterr().out
    assert all(name in shown for name in [*SUITES, *VARIANTS])
    assert "(loss.positive_weight = prior)" in shown
    # The names go to run_suite as given; the exit status says whether all completed.
    ran: list[tuple[str, ...]] = []

    def run_suite(names: list[str]) -> dict[str, Any]:
        ran.append(tuple(names))
        return {"status": "complete" if names else "failed"}

    monkeypatch.setattr(runner, "run_suite", run_suite)
    for argv, status in ((["no_attention", "controls"], 0), ([], 1)):
        monkeypatch.setattr(sys, "argv", ["run_experiments", *argv])
        assert script.main() == status
        assert json.loads(capsys.readouterr().out)["status"] in ("complete", "failed")
    assert ran == [("no_attention", "controls"), ()]
    # An unknown name is refused before anything runs.
    monkeypatch.setattr(sys, "argv", ["run_experiments", "no_graph"])
    with pytest.raises(SystemExit) as refused:
        script.main()
    assert refused.value.code == 2 and len(ran) == 2
    assert "Unknown suites or variants ['no_graph']" in capsys.readouterr().err
