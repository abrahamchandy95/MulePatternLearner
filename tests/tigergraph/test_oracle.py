"""The oracle readers: the reveal's inputs, read through the executor without writing."""

from __future__ import annotations

import pytest

from mule_pattern_learner.testing.builders import reveal_inputs
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.oracle import TigerGraphRevealInputReader


def test_the_inputs_of_the_reveal_are_read_through_the_executor() -> None:
    graph = FakeTigerGraph(reveal=reveal_inputs())
    rows = TigerGraphRevealInputReader(graph).read("scope")
    assert rows == reveal_inputs() and graph.calls == [("reveal inputs", {"scope_id": "scope"})]
    # Nothing was written, and no installed query ran: the job itself never runs.
    assert graph.writes == [] and graph.names() == ["reveal inputs"]
    empty = FakeTigerGraph(reveal=[{"zelle_links": []}])
    with pytest.raises(ValueError, match="no mules result"):
        TigerGraphRevealInputReader(empty).read("scope")
