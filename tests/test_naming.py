"""Names follow the naming rules of docs/architecture.md (its Naming).

Checked: file and folder names (the guides' in kebab-case), the identifiers the code
defines (read from the AST), string literals that are names rather than prose, the
command line, the run paths and the GSQL query names. Prose (docstrings, comments,
messages and the guides' text) is not.

A name is split into words at underscores, hyphens, dots and case changes, and no
word may be one of FORBIDDEN. The values the graph and the seeded draws hold keep their
names; ALLOWED lists them, each with the reason it stays. The names the queries were
installed under before they were renamed are contract.server.RETIRED_QUERIES, which
`mule install` drops, so they are allowed from there. Names another library defines
are that library's: the code only reads them (pylibcugraph's sampler), and the tests'
imitation of pylibcugraph gives them as keyword arguments.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
import re
import subprocess
import tomllib

from mule_pattern_learner import cli
from mule_pattern_learner.contract import server
from mule_pattern_learner.paths import (
    BASELINE_VARIANT,
    DATA_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    DiagnosticsPaths,
    RunPaths,
    SuitePaths,
    archive_dir,
    archived_run,
)
from mule_pattern_learner.tigergraph.gsql_text import definitions

# Words no name may contain: the old layout's prefixes and grab-bag modules, and the
# words "dataset" and "variant" replaced (docs/architecture.md, Naming).
FORBIDDEN = frozenset(
    {
        "temporal",
        "live",
        "v5",
        "legacy",
        "utils",
        "util",
        "common",
        "helpers",
        "helper",
        "cohort",
        "cohorts",
        "arm",
        "arms",
    }
)
# The persisted values that keep a forbidden word, and why. The scope's edge types
# (Entity_In_Training_Scope and its reverse) and the built-in scope id (strict_mule_v2)
# are persisted too, and hold no forbidden word.
ALLOWED = {
    "Temporal_Training_Scope": "the scope vertex type, part of the graph's schema",
    "temporal_live_step": "the salt of the per-step draws; a new value changes every step",
    "marginal_cohort": "the salt of the reservoir ranks; a new value selects other accounts",
}
# The folders whose file names are checked.
CHECKED_FOLDERS = ("src", "tests", "scripts", "gsql", "docs")
# A guide's or a docs folder's name: lowercase words joined by hyphens.
KEBAB_CASE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
COMMANDS = ("install", "train", "evaluate", "score", "report", "diagnose", "check")
MARKERS = {"graph", "graph_write", "cuda"}
# The verbs a GSQL query name starts with (the graph is dedicated, so there is no prefix).
QUERY_VERBS = frozenset(
    {"fetch", "encode", "create", "finalize", "list", "summarize", "resolve", "reveal"}
    | {"draw", "validate", "read"}
)
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def words(name: str) -> list[str]:
    """The words of a name: split at separators, and where lower case meets upper case."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return [word.lower() for word in re.split(r"[_\-.\s]+", spaced) if word]


def forbidden(name: str) -> bool:
    return name not in ALLOWED and bool(FORBIDDEN & set(words(name)))


def tracked(*folders: str) -> list[Path]:
    """The files git tracks under folders, and the new ones it does not ignore."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *folders],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPOSITORY_ROOT / name for name in listed if (REPOSITORY_ROOT / name).exists()]


def python_files() -> list[Path]:
    return [path for path in tracked("src", "tests", "scripts") if path.suffix == ".py"]


def defined_names(tree: ast.AST) -> Iterator[str]:
    """Every name the code defines, and the pytest markers it uses.

    Import aliases, functions, classes, arguments, assignment targets and the attributes
    it sets. Names it only reads, keyword arguments included, belong to other code.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            yield node.attr
        elif isinstance(node, ast.alias) and node.asname:
            yield node.asname
        elif isinstance(node, ast.Attribute) and ast.unparse(node.value) == "pytest.mark":
            yield node.attr


def name_literals(tree: ast.AST) -> Iterator[str]:
    """String literals that are single names (no spaces), not prose."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if IDENTIFIER.match(node.value):
                yield node.value


def query_names() -> set[str]:
    return {
        name
        for path in (REPOSITORY_ROOT / "gsql").rglob("*.gsql")
        for name in definitions(path.read_text())
    }


def test_file_and_folder_names_are_lowercase_and_free_of_old_words() -> None:
    problems = []
    for path in tracked(*CHECKED_FOLDERS):
        for part in path.relative_to(REPOSITORY_ROOT).parts:
            stem = part.split(".")[0]
            if stem != stem.lower() and part not in ("README.md",):
                problems.append(f"{path}: {part} is not lower case")
            if forbidden(stem):
                problems.append(f"{path}: {part}")
    assert problems == []


def test_guides_and_their_folders_have_kebab_case_names() -> None:
    # Figures keep the names of the plots they are copies of (plots/<topic>_<figure>.png).
    problems = []
    for path in tracked("docs"):
        relative = path.relative_to(REPOSITORY_ROOT / "docs")
        names = [*relative.parts[:-1], path.stem] if path.suffix == ".md" else relative.parts[:-1]
        problems += [f"{relative}: {name}" for name in names if not KEBAB_CASE.fullmatch(name)]
    assert problems == []


def test_no_two_package_modules_share_a_name() -> None:
    package = REPOSITORY_ROOT / "src" / "mule_pattern_learner"
    names = Counter(path.stem for path in package.rglob("*.py") if path.stem != "__init__")
    assert [name for name, count in names.items() if count > 1] == []


def test_the_code_defines_no_name_with_an_old_word() -> None:
    problems = []
    for path in python_files():
        tree = ast.parse(path.read_text())
        for name in defined_names(tree):
            if forbidden(name):
                problems.append(f"{path.relative_to(REPOSITORY_ROOT)}: {name}")
    assert sorted(set(problems)) == []


def test_names_in_strings_are_new_or_persisted() -> None:
    problems = []
    used: set[str] = set()
    # This module's own strings are the lists of words and names, not names in use.
    for path in (path for path in python_files() if path != Path(__file__)):
        tree = ast.parse(path.read_text())
        used.update(defined_names(tree))
        for value in name_literals(tree):
            used.add(value)
            if value not in server.RETIRED_QUERIES and forbidden(value):
                problems.append(f"{path.relative_to(REPOSITORY_ROOT)}: {value!r}")
    assert sorted(set(problems)) == []
    # Every allowed value is still in use, so the list only shrinks.
    assert set(ALLOWED) - used == set()


def good_query_name(name: str) -> bool:
    return (
        re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None
        and words(name)[0] in QUERY_VERBS
        and not forbidden(name)
    )


def test_query_names_start_with_a_verb_and_retired_names_are_gone() -> None:
    names = query_names()
    assert [name for name in sorted(names) if not good_query_name(name)] == []
    # No file defines a retired name, so `mule install` never drops a query it installs.
    retired = server.RETIRED_QUERIES
    assert len(set(retired)) == len(retired) and not set(retired) & names


def test_the_server_contract_names_every_query_once() -> None:
    # Adapters, fakes and tests take the names from contract.server, and every query a
    # GSQL file defines has its name there.
    names = [value for key, value in vars(server).items() if key.endswith("_QUERY")]
    assert len(names) == len(set(names))
    assert set(names) == query_names()


def test_the_command_line_is_mule_with_seven_commands() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    assert project["project"]["scripts"] == {"mule": "mule_pattern_learner.cli:main"}
    parser = cli.build_parser()
    (commands,) = (
        action.choices
        for action in parser._actions  # pyright: ignore[reportPrivateUsage]
        if action.choices is not None and action.dest == "command"
    )
    assert tuple(commands) == COMMANDS
    assert not any(forbidden(command) for command in commands)
    markers = {line.split(":")[0] for line in project["tool"]["pytest"]["ini_options"]["markers"]}
    assert markers == MARKERS


def test_outputs_go_under_results_and_datasets_under_data() -> None:
    assert (DATA_DIR.name, RESULTS_DIR.name) == ("data", "results")
    run = RunPaths.of(BASELINE_VARIANT, 42)
    assert run.root.relative_to(RESULTS_DIR).as_posix() == "baseline/seed-42"
    suite = SuitePaths.of("controls")
    assert suite.root.relative_to(RESULTS_DIR).as_posix() == "experiments/controls"
    study = DiagnosticsPaths.of("id")
    assert study.root.relative_to(RESULTS_DIR).as_posix() == "diagnostics/id"
    for path in (study.features, study.table("learning-curve"), study.figure("drift")):
        relative = path.relative_to(study.root)
        assert relative.as_posix() == relative.as_posix().lower(), relative
    assert study.table("learning-curve").name == "learning_curve.csv"
    assert archive_dir().relative_to(RESULTS_DIR).as_posix() == "archive"
    moved = archived_run(RunPaths.of("prior_weight", 43), "20260928T120000Z")
    assert moved.root.relative_to(RESULTS_DIR).as_posix() == (
        "archive/prior_weight/seed-43/20260928T120000Z"
    )
    for path in (suite.summary, suite.comparison, suite.figure("comparison_ap"), suite.report):
        relative = path.relative_to(suite.root)
        assert relative.as_posix() == relative.as_posix().lower(), relative
    files = [
        run.config,
        run.model,
        run.resume,
        run.history,
        run.epochs,
        run.events,
        run.metrics,
        run.predictions("test"),
        run.audit_report("test"),
        run.audit_scores("test"),
        run.audit_rejected("test"),
        run.scores("accounts", "2025-01-01"),
        run.scores_rejected("accounts", "2025-01-01"),
        run.plots,
        run.report,
    ]
    for path in files:
        relative = path.relative_to(run.root)
        assert not any(forbidden(part.split(".")[0]) for part in relative.parts), relative
        assert relative.as_posix() == relative.as_posix().lower(), relative
