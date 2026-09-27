"""The integration tests collect cleanly, and none of them runs by default."""

from __future__ import annotations

import ast
import subprocess
import sys

from mule_pattern_learner.paths import REPOSITORY_ROOT

INTEGRATION = REPOSITORY_ROOT / "tests" / "integration"
# The integration test modules and the marker every test in them carries.
MARKERS = {
    "test_context_query.py": "graph",
    "test_cugraph_sampler.py": "cuda",
    "test_feature_parity.py": "graph",
    "test_label_reveal.py": "graph",
    "test_scope_isolation.py": "graph_write",
}


def collect(*options: str) -> list[str]:
    """The node ids pytest collects from tests/integration with these options."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + [*options, str(INTEGRATION)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    # Exit code 5 means nothing was collected, which is what the default selection gives.
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if "::" in line]


def module_marker(name: str) -> str:
    """The marker a module's pytestmark assignment gives all of its tests."""
    tree = ast.parse((INTEGRATION / name).read_text())
    (value,) = (
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [ast.unparse(target) for target in node.targets] == ["pytestmark"]
    )
    return ast.unparse(value).removeprefix("pytest.mark.")


def test_integration_tests_collect_and_are_deselected_by_default() -> None:
    assert {path.name for path in INTEGRATION.glob("test_*.py")} == set(MARKERS)
    assert {name: module_marker(name) for name in MARKERS} == MARKERS
    assert collect() == []
    selected = collect("-m", "graph or graph_write or cuda")
    assert {node.split("::")[0].rsplit("/", 1)[1] for node in selected} == set(MARKERS)
