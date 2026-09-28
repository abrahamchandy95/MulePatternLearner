"""The import contracts that list every package but a few leave none out, and no pyplot.

import-linter checks only the source modules a contract names, so a package missing
from such a list goes unchecked. It sees an external package only as a whole, so an AST
check keeps matplotlib.pyplot out of every module.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

import mule_pattern_learner
from mule_pattern_learner.paths import REPOSITORY_ROOT

PACKAGE = "mule_pattern_learner"
# The contracts that list every top-level module but those they allow.
ENUMERATED = {
    "Only reporting draws": {"reporting"},
    "Fakes stay out of the package": {"testing"},
    "Only diagnostics uses the verification mirrors": {"diagnostics", "reference"},
}


def top_level_modules() -> set[str]:
    root = Path(mule_pattern_learner.__file__).parent
    modules = {path.stem for path in root.glob("*.py") if path.stem != "__init__"}
    return modules | {path.name for path in root.iterdir() if (path / "__init__.py").exists()}


def contracts() -> dict[str, dict[str, object]]:
    config = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    listed = config["tool"]["importlinter"]["contracts"]
    return {contract["name"]: contract for contract in listed}


def test_every_enumerated_contract_lists_every_package_but_those_it_allows() -> None:
    known = contracts()
    for name, allowed in ENUMERATED.items():
        sources = known[name]["source_modules"]
        assert isinstance(sources, list)
        listed = {str(module).removeprefix(PACKAGE + ".") for module in sources}
        assert listed == top_level_modules() - allowed, name


def pyplot_imports(tree: ast.AST) -> list[str]:
    """The imports of pyplot (or pylab, its alias) in a module."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                a.name for a in node.names if a.name.startswith(("matplotlib.pyplot", "pylab"))
            ]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith(("matplotlib.pyplot", "pylab")):
                found.append(node.module)
            elif node.module == "matplotlib":
                found += [f"matplotlib.{a.name}" for a in node.names if a.name == "pyplot"]
    return found


def test_nothing_imports_pyplot() -> None:
    # Figures are matplotlib.figure.Figure objects saved through the Agg canvas.
    problems = []
    for folder in ("src", "tests", "scripts"):
        for path in sorted((REPOSITORY_ROOT / folder).rglob("*.py")):
            names = pyplot_imports(ast.parse(path.read_text()))
            problems += [f"{path.relative_to(REPOSITORY_ROOT)}: {name}" for name in names]
    assert problems == []
