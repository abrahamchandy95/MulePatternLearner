"""Fixtures several test modules share, and the option that lets tests write to the graph.

Shared fakes and builders live in the package, in mule_pattern_learner.testing. Tests
marked graph, graph_write or cuda are deselected unless -m names them (pyproject.toml);
tests marked graph_write also need --allow-graph-writes, because they write to the
TigerGraph named in .env.
"""

from __future__ import annotations

import pytest

from mule_pattern_learner.tigergraph.render import render_context_query


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--allow-graph-writes",
        action="store_true",
        default=False,
        help="run the tests marked graph_write, which write to the TigerGraph named in .env",
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
    return render_context_query()
