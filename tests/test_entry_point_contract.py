"""The entry points of the executor (docs/design.md, "Query plumbing" and the Labels guard):
the fetch executor's allowlist keeps the answer key out of data and fetch code only if no
module there can reach it another way.

Outside evaluation/ and testing/ (and the registry and the executor, which define them), no
module names the evaluation or parity executor, the evaluation allowlist or the answer key's
query, by name or in a string other than a docstring. Outside tigergraph/ and testing/, no
module runs a query, a connection operation or a GSQL statement on a TigerGraphExecutor
itself: it runs queries through an entry point, which refuses any its role may not run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mule_pattern_learner

PACKAGE = Path(mule_pattern_learner.__file__).parent
# What reaches the answer key, and the modules that may name it.
ANSWER_KEY_NAMES = frozenset(
    {"EvaluationExecutor", "ParityExecutor", "EVALUATION_ONLY", "READ_ANSWER_KEY"}
)
ANSWER_KEY = "read_answer_key"
ANSWER_KEY_MODULES = ("evaluation/", "testing/", "tigergraph/queries/", "tigergraph/executor.py")
# The executor that runs anything installed, its methods that run something, and the
# modules that may call them on it.
EXECUTOR = "TigerGraphExecutor"
RUNS = frozenset({"run", "call", "gsql"})
RUN_MODULES = ("tigergraph/", "testing/")


def _docstrings(tree: ast.Module) -> set[int]:
    """The ids of a module's docstring nodes: its own, its classes' and its functions'."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def _builds(node: ast.expr | None) -> bool:
    """Whether node builds a TigerGraphExecutor."""
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == EXECUTOR)
        or (isinstance(node.func, ast.Attribute) and node.func.attr == EXECUTOR)
    )


def _typed(annotation: ast.expr | None) -> bool:
    """Whether an annotation names a TigerGraphExecutor."""
    return annotation is not None and EXECUTOR in ast.unparse(annotation)


def _executors(tree: ast.Module) -> set[str]:
    """What a module binds to a TigerGraphExecutor, as written (`raw`, `self.executor`):
    assigned from its constructor, or annotated with its class."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _builds(node.value):
            found |= {ast.unparse(target) for target in node.targets}
        elif isinstance(node, ast.AnnAssign) and (_typed(node.annotation) or _builds(node.value)):
            found.add(ast.unparse(node.target))
        elif isinstance(node, ast.NamedExpr) and _builds(node.value):
            found.add(node.target.id)
        elif isinstance(node, ast.arg) and _typed(node.annotation):
            found.add(node.arg)
    return found


def violations(source: str, module: str) -> list[str]:
    """What breaks the contract in a module's source; module is its path under the
    package, such as `data/labels.py`."""
    tree = ast.parse(source)
    may_name = module.startswith(ANSWER_KEY_MODULES)
    may_run = module.startswith(RUN_MODULES)
    docstrings = _docstrings(tree)
    executors = _executors(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if not may_name:
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                names = [ANSWER_KEY] if ANSWER_KEY in node.value else [node.value]
            found += [
                f"{module}:{line} names {name}"
                for name in names
                if name in ANSWER_KEY_NAMES or name == ANSWER_KEY
            ]
        if (
            not may_run
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in RUNS
            and (_builds(node.func.value) or ast.unparse(node.func.value) in executors)
        ):
            found.append(f"{module}:{line} calls {node.func.attr} on a {EXECUTOR}")
    return found


def tree_violations(package: Path) -> list[str]:
    """What breaks the contract in every module of a package folder."""
    return [
        found
        for path in sorted(package.rglob("*.py"))
        for found in violations(path.read_text(), path.relative_to(package).as_posix())
    ]


def test_the_package_keeps_the_entry_point_contract() -> None:
    modules = list(PACKAGE.rglob("*.py"))
    assert PACKAGE / "tigergraph" / "executor.py" in modules and len(modules) > 10
    assert tree_violations(PACKAGE) == []


# A scratch module that reaches the answer key around the fetch executor.
PLANTED = '''
"""Mentions read_answer_key in its docstring, which runs nothing."""
from ..tigergraph.executor import EvaluationExecutor, ParityExecutor, TigerGraphExecutor
from ..tigergraph.queries import EVALUATION_ONLY, READ_ANSWER_KEY
from ..tigergraph import queries


def fetch(executor: TigerGraphExecutor) -> None:
    executor.run("read_" "answer_key", {})
    ParityExecutor(executor).run(queries.READ_ANSWER_KEY, {})
    EvaluationExecutor(executor)


class Fetch:
    def __init__(self) -> None:
        self.executor = TigerGraphExecutor(settings=None)

    def rows(self) -> None:
        self.executor.call(lambda conn: conn, what="rows")


raw: TigerGraphExecutor = TigerGraphExecutor(settings=None)
raw.gsql("RUN QUERY x()")
TigerGraphExecutor(settings=None).run("list_accounts", {})
allowed = EVALUATION_ONLY
'''


def test_a_planted_reach_of_the_answer_key_is_found() -> None:
    found = violations(PLANTED, "data/scratch.py")
    named = {line.split(" names ")[1] for line in found if " names " in line}
    assert named == {*ANSWER_KEY_NAMES, ANSWER_KEY}
    runs = sorted(int(line.split(":")[1].split()[0]) for line in found if " calls " in line)
    # executor.run, self.executor.call, raw.gsql and the run on a new executor.
    assert runs == [9, 19, 23, 24]
    # Evaluation may name the answer key, but runs queries through its entry point too.
    assert all(" calls " in line for line in violations(PLANTED, "evaluation/scratch.py"))
    assert len(violations(PLANTED, "evaluation/scratch.py")) == 4
    # tigergraph/ runs the executor, and names the answer key only where it is defined.
    assert all(" names " in line for line in violations(PLANTED, "tigergraph/fetch.py"))
    assert violations(PLANTED, "tigergraph/executor.py") == []
    assert violations(PLANTED, "testing/parity.py") == []


def test_a_planted_module_fails_the_tree(tmp_path: Path) -> None:
    (tmp_path / "subgraph").mkdir()
    (tmp_path / "subgraph" / "fetch.py").write_text(
        "from ..tigergraph.executor import ParityExecutor\n"
    )
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "audit.py").write_text(
        "from ..tigergraph.executor import EvaluationExecutor\n"
    )
    assert tree_violations(tmp_path) == ["subgraph/fetch.py:1 names ParityExecutor"]
