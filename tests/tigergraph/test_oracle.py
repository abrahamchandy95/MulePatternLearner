"""The oracle readers: the ground truth, paged, and the reveal's inputs, read without writing."""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.contract.server import TRUTH_QUERY
from mule_pattern_learner.testing.builders import reveal_inputs
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, page
from mule_pattern_learner.tigergraph.oracle import (
    TigerGraphRevealInputReader,
    TigerGraphTruthReader,
)


def test_the_inputs_of_the_reveal_are_read_through_the_executor() -> None:
    graph = FakeTigerGraph(reveal=reveal_inputs())
    rows = TigerGraphRevealInputReader(graph).read("scope")
    assert rows == reveal_inputs() and graph.calls == [("reveal inputs", {"scope_id": "scope"})]
    # Nothing was written, and no installed query ran: the job itself never runs.
    assert graph.writes == [] and graph.names() == ["reveal inputs"]
    empty = FakeTigerGraph(reveal=[{"zelle_links": []}])
    with pytest.raises(ValueError, match="no mules result"):
        TigerGraphRevealInputReader(empty).read("scope")


def test_the_graph_truth_pages_the_label_contract() -> None:
    rows = [
        {
            "account_id": f"A{i:05}",
            "is_mule": i % 2,
            "mule_label_known": i % 3 != 0,
            "mule_ring_id": i // 10 if i % 2 else -1,
            "mule_label_source": "phantomledger_role",
        }
        for i in range(10050)
    ]
    graph = FakeTigerGraph(truth=rows)
    truth = TigerGraphTruthReader(graph).read()
    assert graph.calls == [
        (TRUTH_QUERY, {"after_id": "", "batch_size": 10000}),
        (TRUTH_QUERY, {"after_id": "A09999", "batch_size": 10000}),
    ]
    assert truth.account_id.tolist() == [r["account_id"] for r in rows]
    # An account whose label is not known is -1, never a negative.
    assert truth.is_mule.tolist() == [r["is_mule"] if r["mule_label_known"] else -1 for r in rows]
    # Each mule's ring and every label's source come with it.
    assert truth.ring_id.tolist() == [r["mule_ring_id"] for r in rows]
    assert set(truth.label_source) == {"phantomledger_role"}

    # GSQL prints the rows of a vertex set under attributes, and they read the same.
    def wrapped(params: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"status": "ok"}, {"accounts": [{"attributes": r} for r in page(rows, params)]}]

    assert (
        TigerGraphTruthReader(FakeTigerGraph(answers={TRUTH_QUERY: wrapped})).read().equals(truth)
    )

    unordered = [
        {"account_id": a, "is_mule": 0, "mule_label_known": True, "mule_ring_id": -1} for a in "BA"
    ]
    graph = FakeTigerGraph(
        answers={TRUTH_QUERY: lambda p: [{"status": "ok", "accounts": unordered}]}
    )
    with pytest.raises(ValueError, match="not strictly increasing"):
        TigerGraphTruthReader(graph).read()
    silent = FakeTigerGraph(answers={TRUTH_QUERY: lambda p: [{"status": "ok"}]})
    with pytest.raises(ValueError, match="accounts missing from response"):
        TigerGraphTruthReader(silent).read()
