"""The connection the integration tests read the graph through.

Every test here is marked graph, graph_write or cuda, so none runs by default. The
connection has the built-in run's retry budgets and reads .env when a test first
asks for it.
"""

from __future__ import annotations

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.pipeline.connect import connect
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor


@pytest.fixture(scope="session")
def graph() -> TigerGraphExecutor:
    return connect(DEFAULT_CONFIG.transport)
