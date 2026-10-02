"""The cutoff query: the last event visible at each cutoff time."""

from __future__ import annotations

from mule_pattern_learner.contract.server import CUTOFF_QUERY
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader


def test_the_last_visible_sequence_of_each_cutoff_in_one_request() -> None:
    executor = FakeTigerGraph(last_visible=lambda index, ms: 10 * index + ms)
    assert TigerGraphCutoffReader(executor).last_visible_seqs([5, 7]) == {5: 5, 7: 17}
    assert executor.names() == [CUTOFF_QUERY]
