"""The installed label reveal agrees with its Python mirror, in a dry run on the graph.

Read-only. The test fetches every input the reveal uses (mules, scope partitions, draw
keys, fraud-labelled Zelle inflows and mule-to-mule events), recomputes the whole plan
with reference.label_reveal.plan, runs the job with apply = FALSE, and compares
discovery channel, availability clock and revealed set mule by mule. The dry run passes
force = TRUE only to skip the job's already-revealed check, so it also works on a graph
whose labels were revealed; with apply = FALSE nothing is written.
"""

from __future__ import annotations

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.reference.label_reveal import INPUTS_QUERY, dry_run_differences, plan
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor, merged_rows
from mule_pattern_learner.tigergraph.reveal import REVEAL_QUERY, reveal_parameters

pytestmark = pytest.mark.graph


def test_a_dry_run_of_the_reveal_matches_its_python_mirror(graph: TigerGraphExecutor) -> None:
    config = DEFAULT_CONFIG
    params = {**reveal_parameters(config.scope, config.dataset.dates, apply=False), "force": True}
    assert params["apply"] is False
    inputs = graph.client.conn.runInterpretedQuery(INPUTS_QUERY, {"scope_id": params["scope_id"]})
    expected = plan(inputs, params)
    result = merged_rows(graph.run(REVEAL_QUERY, params, timeout_s=3600.0, attempts=1))
    assert result.get("status") == "dry_run", result
    assert dry_run_differences(expected, result) == []
