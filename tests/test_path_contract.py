"""The path contract of the Files that carry labels guard (docs/design.md, "Leak rules and
how tests enforce them"): only evaluation/ and testing/ reach the locations that hold
labels or values computed from the truth.

The evaluation store, the loader's export folder and PhantomLedger's gate records each have
one accessor in paths.py, and no module outside evaluation/ and testing/ may name one: not
import it, call it or reach it as an attribute. Nor may such a module reach them around their
accessors, by the settings that name their folders or a gate record's file name. Only
paths.py names its own internals (the reader of the folders in .env and the roots of what
the code writes), and testing/ the repository's root, which it runs the tests from. The
loader's manifest and verification report, and PhantomLedger's run summary, which no
MulePatternLearner code opens, are named by no string outside testing/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mule_pattern_learner

PACKAGE = Path(mule_pattern_learner.__file__).parent
# The accessors of the locations that carry labels, and where they may be named.
GUARDED = frozenset({"evaluation_store_dir", "export_root", "pl_gate_record"})
ACCESSOR_PACKAGES = ("evaluation/", "testing/")
# The settings and file names that reach those locations without their accessors.
LOCATIONS = ("MULE_EXPORT_DIR", "MULE_PL_RECORDS_DIR", ".gates.toml")
# What paths.py alone names: the reader of the folders in .env and the roots of what the
# code writes, and the repository's root, which testing/ also names.
INTERNALS = frozenset({"_folder", "_Folders", "DATA_DIR", "RESULTS_DIR"})
ROOT = "REPOSITORY_ROOT"
ROOT_PACKAGES = ("testing/",)
# The files that carry labels and no accessor opens, and where their names may appear.
MARKERS = ("export_manifest", "tigergraph_verification", "run_summary")
MARKER_PACKAGES = ("testing/",)


def violations(source: str, module: str) -> list[str]:
    """What breaks the contract in a module's source; module is its path under the
    package, such as `data/labels.py`."""
    in_paths = module == "paths.py"
    may_name = in_paths or module.startswith(ACCESSOR_PACKAGES)
    may_mark = module.startswith(MARKER_PACKAGES)
    refused = set() if may_name else set(GUARDED)
    if not in_paths:
        refused |= INTERNALS
        if not module.startswith(ROOT_PACKAGES):
            refused.add(ROOT)
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
            marked = [] if may_mark else list(MARKERS)
            marked += [] if may_name else list(LOCATIONS)
            found += [
                f"{module}:{node.lineno} names {marker}"
                for marker in marked
                if marker in node.value
            ]
        found += [
            f"{module}:{getattr(node, 'lineno', 0)} names {name}"
            for name in names
            if name in refused
        ]
    return found


def tree_violations(package: Path) -> list[str]:
    """What breaks the contract in every module of a package folder."""
    return [
        found
        for path in sorted(package.rglob("*.py"))
        for found in violations(path.read_text(), path.relative_to(package).as_posix())
    ]


def test_the_package_keeps_the_path_contract() -> None:
    modules = list(PACKAGE.rglob("*.py"))
    assert PACKAGE / "paths.py" in modules and len(modules) > 10
    assert tree_violations(PACKAGE) == []


# A scratch module that reaches every guarded location and names every marked file.
PLANTED = """
from ..paths import evaluation_store_dir
from .. import paths
import mule_pattern_learner.paths as where

store = evaluation_store_dir("0" * 32)
shards = paths.export_root()
gates = where.pl_gate_record
reach = getattr(paths, "export_root")
manifest = "export_manifest.json"
report = f"{shards}/tigergraph_verification.json"
summary = "run_summary"
"""


# A scratch module that reaches the same locations around their accessors.
AROUND = """
from ..paths import _folder, DATA_DIR, REPOSITORY_ROOT, pl_identity_files
from .. import paths

shards = _folder("MULE_EXPORT_DIR")
store = DATA_DIR / "evaluation"
gates = pl_identity_files()[0].with_name("world_2001.gates.toml")
records = paths._Folders().mule_pl_records_dir or "MULE_PL_RECORDS_DIR"
results = getattr(paths, "RESULTS_DIR")
root = REPOSITORY_ROOT / "data" / "evaluation"
"""


def test_a_planted_reach_is_found_outside_evaluation_and_testing() -> None:
    found = violations(PLANTED, "data/scratch.py")
    for name in (*GUARDED, *MARKERS):
        assert any(line.endswith(f"names {name}") for line in found), name
    # evaluation/ may reach the locations, but names no marked file.
    assert all(line.split()[-1] in MARKERS for line in violations(PLANTED, "evaluation/scratch.py"))
    assert violations(PLANTED, "evaluation/scratch.py")
    assert violations(PLANTED, "testing/scratch.py") == []


def test_a_planted_reach_around_the_accessors_is_found() -> None:
    found = violations(AROUND, "data/scratch.py")
    for name in (*INTERNALS, ROOT, *LOCATIONS):
        assert any(line.endswith(f"names {name}") for line in found), name
    # evaluation/ may name the locations' settings, but no internal of paths.py; testing/
    # may also name the repository's root.
    assert {line.split()[-1] for line in violations(AROUND, "evaluation/scratch.py")} == {
        *INTERNALS,
        ROOT,
    }
    assert {line.split()[-1] for line in violations(AROUND, "testing/scratch.py")} == INTERNALS
    assert violations(AROUND, "paths.py") == []


def test_a_planted_module_fails_the_tree(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "scratch.py").write_text("from ..paths import export_root\n")
    (tmp_path / "testing").mkdir()
    (tmp_path / "testing" / "parity.py").write_text("from ..paths import export_root\n")
    assert tree_violations(tmp_path) == ["data/scratch.py:1 names export_root"]
