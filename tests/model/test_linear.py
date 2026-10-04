"""The linear controls: one linear layer of the root's own inputs, alone or beside TGAT."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import FeaturePlan, extraction_plan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.inference.predictor import Predictor, score_batch
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.model.linear import LinearModel, WideAndDeep
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.testing.builders import context, message, saved_model
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

LINEAR = DEFAULT_CONFIG.with_changes({"model": {"architecture": "linear", "slot_sum": False}})
WIDE = DEFAULT_CONFIG.with_changes({"model": {"architecture": "wide_and_deep"}})
FANOUT = DEFAULT_CONFIG.sampler.fanouts[0]
ROOT = ContextKey("Account", "root", 1000, 100_000_000)
PAYMENTS = [
    message(seq, seq * 60_000, ROOT, relation="payment_in", rail="ach", node_id=f"P{seq % 3}")
    for seq in range(10, 16)
]


def batch_of(plan: FeaturePlan) -> tuple[dict[str, torch.Tensor], FakeTigerGraph]:
    """The built-in sampler's batch of ROOT under a plan, and the graph that served it."""
    executor = FakeTigerGraph({ROOT: context(ROOT, PAYMENTS)})
    sampler = DEFAULT_CONFIG.sampler
    with ContextSource(
        TigerGraphContextFetcher(executor), plan=extraction_plan(plan), sampler=sampler
    ) as source:
        return build_batch(source, [ROOT], plan=plan, sampler=sampler), executor


def test_the_linear_model_is_one_layer_of_the_roots_own_inputs() -> None:
    plan = LINEAR.feature_plan()
    assert plan.root_only and plan.node_names == FeaturePlan(architecture="summary").node_names
    model = build_model(LINEAR.model, plan, FANOUT)
    assert isinstance(model, LinearModel)
    width = len(plan.node_names)
    assert [(name, tuple(p.shape)) for name, p in model.named_parameters()] == [
        ("head.weight", (1, width)),
        ("head.bias", (1,)),
    ]
    # Like the summary model, it fetches the roots alone.
    batch, executor = batch_of(plan)
    assert set(batch) == {"x", "root_positions"} and executor.requested == [ROOT]
    # The logit is a weighted sum of the inputs, which are its embedding.
    logits = model(batch)
    expected = batch["x"] @ model.head.weight[0] + model.head.bias
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(model.logits(batch, model.encode(batch)), logits)


def test_wide_and_deep_adds_a_linear_layer_of_the_roots_inputs_to_the_graph_models_logit() -> None:
    plan = WIDE.feature_plan()
    assert not plan.root_only
    # The graph model's inputs, contexts and query flags at both hops.
    built_in = DEFAULT_CONFIG.feature_plan()
    for hop in (1, 2):
        assert plan.query_flags(hop) == built_in.query_flags(hop)
    torch.manual_seed(0)
    graph = build_model(DEFAULT_CONFIG.model, built_in, FANOUT, dropout=0.0)
    torch.manual_seed(0)
    model = build_model(WIDE.model, plan, FANOUT, dropout=0.0)
    assert isinstance(model, WideAndDeep) and isinstance(graph, TGAT)
    # TGAT's modules come first, with the built-in model's initial weights for a seed;
    # the wide path after them.
    state, before = model.state_dict(), graph.state_dict()
    assert list(state) == [*before, "wide.weight", "wide.bias"]
    assert all(torch.equal(state[name], before[name]) for name in before)
    assert state["wide.weight"].shape == (1, len(plan.node_names))
    model.eval()
    graph.eval()
    batch, _ = batch_of(plan)
    roots = batch["x"][batch["root_positions"]]
    with torch.no_grad():
        wide = roots @ model.wide.weight[0] + model.wide.bias
        torch.testing.assert_close(model(batch), graph(batch) + wide)
        # The embeddings a predictor keeps are the graph model's, and its logits the sum.
        hidden = model.encode(batch)
        torch.testing.assert_close(hidden, graph.encode(batch))
        torch.testing.assert_close(model.logits(batch, hidden), model(batch))


def test_a_saved_wide_and_deep_model_scores_with_its_wide_path(tmp_path: Path) -> None:
    path = saved_model(tmp_path / "model.pt", WIDE)
    executor = FakeTigerGraph({ROOT: context(ROOT, PAYMENTS)})
    plan = WIDE.feature_plan()
    with ContextSource(
        TigerGraphContextFetcher(executor), plan=extraction_plan(plan), sampler=WIDE.sampler
    ) as source:
        predictor = Predictor(SavedModel.load(path), source, "cpu")
        assert isinstance(predictor.model, WideAndDeep)
        prepared = predictor.prepare([ROOT])
        scored = score_batch(predictor.model, prepared, predictor.device, embeddings=True)
        assert prepared.batch is not None and scored.logits is not None
        with torch.no_grad():
            expected = predictor.model(prepared.batch)
        torch.testing.assert_close(scored.logits, expected)


def test_each_class_needs_its_own_architecture() -> None:
    with pytest.raises(ValueError, match="LinearModel needs a linear feature plan, not 'summary'"):
        LinearModel(plan=FeaturePlan(architecture="summary"))
    with pytest.raises(ValueError, match="WideAndDeep needs a wide_and_deep feature plan"):
        WideAndDeep(16, 4, 0, plan=FeaturePlan(), slot_sum=False, first_fanout=16)
    with pytest.raises(ValueError, match="TGAT needs a tgat feature plan, not 'wide_and_deep'"):
        TGAT(
            16,
            4,
            0,
            plan=FeaturePlan(architecture="wide_and_deep"),
            slot_sum=False,
            first_fanout=16,
        )
    # A linear model of the root's inputs needs some, and the wide path's graph model
    # needs message_core.
    with pytest.raises(ValueError, match="needs node or summary inputs"):
        FeaturePlan(("message_core",), "linear")
    with pytest.raises(ValueError, match="Graph models require message_core"):
        FeaturePlan(("entity_meta",), "wide_and_deep")
