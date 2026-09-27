"""The summary model of the controls: root columns only, no fetches beyond the roots."""

from __future__ import annotations

import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.model.summary_mlp import SummaryMLP
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.testing.builders import context, message
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

TABULAR = DEFAULT_CONFIG.with_changes({"model": {"architecture": "summary"}})


def test_the_architecture_chooses_the_model() -> None:
    fanout = DEFAULT_CONFIG.sampler.fanouts[0]
    assert isinstance(
        build_model(DEFAULT_CONFIG.model, DEFAULT_CONFIG.feature_plan(), fanout), TGAT
    )
    summary = build_model(TABULAR.model, TABULAR.feature_plan(), fanout)
    assert isinstance(summary, SummaryMLP)
    # A projection of every root column, then the head: the parameters of the summary
    # model the code before the layered restructure built, in the same order.
    width = len(TABULAR.feature_plan().node_names)
    assert [(name, tuple(p.shape)) for name, p in summary.named_parameters()] == [
        ("node.0.weight", (64, width)),
        ("node.0.bias", (64,)),
        ("node.2.weight", (64,)),
        ("node.2.bias", (64,)),
        ("head.0.weight", (64, 64)),
        ("head.0.bias", (64,)),
        ("head.3.weight", (1, 64)),
        ("head.3.bias", (1,)),
    ]


def test_summary_models_fetch_only_the_roots_and_have_no_graph_parameters() -> None:
    root = ContextKey("Account", "root", 100, 1000)
    summary = FeaturePlan(("decayed_activity",), "summary")
    executor = FakeTigerGraph({root: context(root, [message(80, 800, root)])})
    source = ContextSource(TigerGraphContextFetcher(executor), plan=summary)
    batch = build_batch(source, [root], plan=summary)
    assert executor.requested == [root]
    assert set(batch) == {"x", "root_positions"}
    model = SummaryMLP(16, 0, plan=summary)
    assert not hasattr(model, "edge")
    assert torch.isfinite(model(batch)).all()
    source.close()


def test_a_summary_model_needs_a_summary_plan_and_valid_sizes() -> None:
    with pytest.raises(ValueError, match="SummaryMLP needs a summary feature plan, not 'tgat'"):
        SummaryMLP(16, 0, plan=FeaturePlan())
    summary = TABULAR.feature_plan()
    with pytest.raises(ValueError, match="Hidden size"):
        SummaryMLP(2, 0, plan=summary)
    with pytest.raises(ValueError, match="Dropout"):
        SummaryMLP(16, 1.0, plan=summary)
