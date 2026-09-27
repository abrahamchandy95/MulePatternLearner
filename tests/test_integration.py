"""The integration tests collect cleanly, and none of them runs by default.

The tests that write to the graph are skipped even when -m selects them, unless
--allow-graph-writes gives consent.
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

from mule_pattern_learner.data.contexts import ContextSource, check_coverage
from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

INTEGRATION = REPOSITORY_ROOT / "tests" / "integration"
# The integration test modules and the marker every test in them carries.
MARKERS = {
    "test_context_query.py": "graph",
    "test_cugraph_sampler.py": "cuda",
    "test_feature_parity.py": "graph",
    "test_label_reveal.py": "graph",
    "test_scope_isolation.py": "graph_write",
}
# A plugin that writes the skip reasons of every test a run would run, after the conftest
# hooks have marked them, to the file named in SKIP_REPORT. It never runs a test.
SKIP_REPORT = """
import json
import os


def pytest_collection_finish(session):
    reasons = {
        item.nodeid: [mark.kwargs.get("reason", "") for mark in item.iter_markers("skip")]
        for item in session.items
    }
    with open(os.environ["SKIP_REPORT"], "w") as report:
        json.dump(reasons, report)
"""


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


def graph_write_skips(folder: Path, *options: str) -> dict[str, list[str]]:
    """The skip reasons of each test that -m graph_write selects, collected without running."""
    (folder / "skip_report.py").write_text(SKIP_REPORT)
    report = folder / "skips.json"
    path = os.pathsep.join(filter(None, (str(folder), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + ["-p", "skip_report", "-m", "graph_write", *options, "tests"],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": path, "SKIP_REPORT": str(report)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(report.read_text())


def test_graph_write_tests_are_skipped_without_explicit_consent(tmp_path: Path) -> None:
    # Selecting graph_write is not consent: without the option every such test is skipped.
    refused = graph_write_skips(tmp_path)
    assert refused
    for node, reasons in refused.items():
        assert any("--allow-graph-writes" in reason for reason in reasons), node
    allowed = graph_write_skips(tmp_path, "--allow-graph-writes")
    assert allowed == {node: [] for node in refused}


def load(name: str) -> ModuleType:
    """An integration test module, imported without running its tests."""
    spec = importlib.util.spec_from_file_location(f"integration_{name}", INTEGRATION / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_isolation_source_requests_what_the_fixture_checks_and_the_model_reads() -> None:
    # Offline, so a wrong source plan shows before the test writes to a graph.
    isolation = load("test_scope_isolation.py")
    config = isolation.model_config()
    plan = isolation.source_plan(config.feature_plan())
    # The values the fixture asserts on: first-hop root features and message fields.
    checked = {
        "1h_out_count",
        "1h_out_amount",
        "1d_out_in_amount_ratio",
        "7d_out_in_amount_ratio",
        "pair_count_1h",
        "pair_count_1d",
        "pair_count_7d",
    }
    assert checked <= set(plan.node_names + plan.edge_names)
    flags = plan.query_flags(1)
    for group in ("rolling_windows", "amount_ratios", "pair_window_counts"):
        assert flags["include_" + group], group
    # The model it trains and the predictor that scores it read nothing the source skips.
    fetcher = TigerGraphContextFetcher(FakeTigerGraph())
    with ContextSource(fetcher, plan=plan, sampler=config.sampler, capacity=0) as source:
        check_coverage(source, config.feature_plan(), config.sampler)
