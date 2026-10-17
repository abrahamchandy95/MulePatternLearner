"""Every query written so far compiles on the live MulePatternLearner workspace that the
repository's .env names (docs/design.md, "Query plumbing"). Marked graph, so it runs only with
`pytest -m graph`, and only after the early wipe has left the workspace with the schema of
gsql/schema/. It installs the queries through the installer, which reinstalls changed ones
only, and runs none: installing a query changes no data.
"""

from __future__ import annotations

import pytest

from mule_pattern_learner.tigergraph import installer
from mule_pattern_learner.tigergraph.connection import ConnectionSettings
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor
from mule_pattern_learner.tigergraph.queries import GRAPH_NAME, written_files

pytestmark = pytest.mark.graph


def test_every_written_query_compiles_on_the_live_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = written_files()
    if not written:
        pytest.skip("No query is written yet")
    monkeypatch.setattr(installer, "QUERY_FILES", written)
    executor = TigerGraphExecutor(settings=ConnectionSettings())
    assert executor.graph_name == GRAPH_NAME
    installer.install(executor)
    installer.verify(executor)
