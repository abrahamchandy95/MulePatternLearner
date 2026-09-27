"""Names follow the Naming conventions of the design record (docs/restructure-plan.md).

Checked: file and folder names, the identifiers the code defines (read from the AST),
string literals that are names rather than prose, the command line, the run paths and
the GSQL query names. Prose (docstrings, comments, messages and the guides) is not.

A name is split into words at underscores, hyphens, dots and case changes, and no
word may be one of FORBIDDEN. The values TigerGraph and saved files hold keep their
names; ALLOWED lists them, each with the reason it stays.
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
    RunPaths,
    archive_dir,
    diagnostics_dir,
    suite_dir,
)
from mule_pattern_learner.tigergraph.gsql_text import definitions

# Words no name may contain: the old layout's prefixes and grab-bag modules, and the
# owner's vocabulary ("dataset", never "cohort"; "variant", never "arm").
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
# The persisted values that keep a forbidden word, and why.
ALLOWED = {
    "Temporal_Training_Scope": "the scope vertex type, part of the graph's schema",
    "temporal_live_v5_candidate_pools": (
        "CONTEXT_CONTRACT, which the installed context query prints and saved models "
        "record; it changes in the server step"
    ),
    "temporal_live_step": "the salt of the per-step draws; a new value changes every step",
    "marginal_cohort": "the salt of the reservoir ranks; a new value selects other accounts",
    "cohort_seed": "a key of configurations saved before the typed configuration",
    "temporal": "the variant value of configurations saved before the typed configuration",
    "temporal_training_population": (
        "a query no file defines any more, still installed until the server step retires it"
    ),
    # pylibcugraph's API, which the tests' mock library imitates.
    "heterogeneous_uniform_temporal_neighbor_sample": "a pylibcugraph function",
    "temporal_sampling_comparison": "a parameter of pylibcugraph's samplers",
}
# The queries installed on the server, which keep their names until the server step
# renames them once (the owner decision on query names) to the name given, or retires
# them (None).
INSTALLED_QUERIES: dict[str, str | None] = {
    "temporal_training_context": "fetch_training_context",
    "temporal_fourier64_values": "encode_fourier64",
    "temporal_fourier64": None,
    "temporal_create_training_scope": "create_training_scope",
    "temporal_finalize_training_scope": "finalize_training_scope",
    "temporal_scope_population": "list_scope_accounts",
    "temporal_scope_policy": "summarize_scope_policy",
    "temporal_training_cutoffs": "resolve_split_cutoffs",
    "temporal_hub_registry": "list_hub_accounts",
    "temporal_reveal_mule_labels": "reveal_mule_labels",
    "temporal_reveal_uniforms": "draw_reveal_uniforms",
    "temporal_validate_account_supervision": "validate_label_contract",
    "temporal_get_account_supervision": "read_ground_truth",
    "zelle_pair_time64": "encode_zelle_pair_gaps",
    "payment_pair_time64": "encode_payment_pair_gaps",
}
# The folders whose file names are checked; the guides move and get kebab-case names
# in the docs step.
CHECKED_FOLDERS = ("src", "tests", "scripts", "gsql")
COMMANDS = ("install", "train", "evaluate", "score", "check")
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
            if value not in INSTALLED_QUERIES and forbidden(value):
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


def test_query_names_start_with_a_verb_or_wait_for_the_server_step() -> None:
    names = query_names()
    assert [name for name in names - INSTALLED_QUERIES.keys() if not good_query_name(name)] == []
    # A query renamed on the server leaves the list; a retired one leaves the files.
    assert set(INSTALLED_QUERIES) <= names
    renamed = [new for new in INSTALLED_QUERIES.values() if new is not None]
    assert [name for name in renamed if not good_query_name(name)] == []


def test_the_server_contract_names_every_installed_query_once() -> None:
    # Adapters, fakes and tests take the names from contract.server, so the server step
    # renames each there; a retired query has no name there.
    names = [value for key, value in vars(server).items() if key.endswith("_QUERY")]
    assert len(names) == len(set(names))
    assert set(names) == {name for name, new in INSTALLED_QUERIES.items() if new is not None}


def test_the_command_line_is_mule_with_five_commands() -> None:
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
    assert suite_dir("controls").relative_to(RESULTS_DIR).as_posix() == "experiments/controls"
    assert diagnostics_dir("id").relative_to(RESULTS_DIR).as_posix() == "diagnostics/id"
    assert archive_dir().relative_to(RESULTS_DIR).as_posix() == "archive"
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
