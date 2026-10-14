"""Fixtures several test modules share, and the options of the tests that use the graph.

Shared fakes and builders are in the package, in mule_pattern_learner.testing. Tests
marked graph, graph_write or cuda are deselected unless -m names them (pyproject.toml);
tests marked graph_write also need --allow-graph-writes, because they write to the
TigerGraph named in .env. --baseline-context-query names an earlier text of the context
query, which a graph test compares the current one with.
"""

from __future__ import annotations

import pytest

from mule_pattern_learner.contract.server import CONTEXT_QUERY_FILE
from mule_pattern_learner.paths import GSQL_DIR


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--allow-graph-writes",
        action="store_true",
        default=False,
        help="run the tests marked graph_write, which write to the TigerGraph named in .env",
    )
    parser.addoption(
        "--baseline-context-query",
        default=None,
        metavar="PATH",
        help="an earlier text of the context query; a graph test checks that the current "
        "text returns the same rows (tests/integration/test_context_query.py), on a graph "
        "loaded with the earlier schema, whose payments keep the channel it reads",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--allow-graph-writes"):
        return
    skip = pytest.mark.skip(reason="writes to the graph; pass --allow-graph-writes to run it")
    for item in items:
        if item.get_closest_marker("graph_write") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def text() -> str:
    """The training context query, gsql/queries/training_context.gsql."""
    return (GSQL_DIR / CONTEXT_QUERY_FILE).read_text()
