"""The feature group registry, extraction plans and plan fingerprints."""

from __future__ import annotations

import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.contract.feature_groups import (
    ARCHITECTURES,
    BUILT_IN_GROUPS,
    CLIENT_GROUPS,
    CORE_GROUPS,
    FEATURE_GROUPS,
    POOL_GROUPS,
    FeaturePlan,
)
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.reference.batch_features import node_features
from mule_pattern_learner.testing.builders import context, message
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

# What TigerGraph computes of the training groups besides message_core.
SERVER_GROUPS = ("entity_meta", "time_encoding", "pair_history", "flow_timing")


def test_training_keeps_only_the_built_in_groups() -> None:
    # The owner's decision (docs/architecture.md): the registry is the built-in run's, and a
    # group outside it is unknown.
    assert set(FEATURE_GROUPS) == set(BUILT_IN_GROUPS)
    for group in ("rolling_windows", "event_channel", "sampler_meta"):
        with pytest.raises(ValueError, match="unknown"):
            FeaturePlan(("message_core", group))


def test_contract_constants_and_client_groups() -> None:
    assert CORE_GROUPS[:2] == ("entity_meta", "hub_indicator")
    assert CLIENT_GROUPS == {"hub_indicator", *POOL_GROUPS}
    assert FEATURE_GROUPS["hub_indicator"].identity == ("history_withheld",)
    plan = FeaturePlan(CORE_GROUPS, "tgat")
    assert plan.names("node")[-1] == "history_withheld"
    # TigerGraph computes every group but the client's own.
    server = tuple(g for g in FEATURE_GROUPS if g not in CLIENT_GROUPS and g != "message_core")
    assert server == SERVER_GROUPS
    # The default plan is the built-in run's.
    assert FeaturePlan() == FeaturePlan(BUILT_IN_GROUPS, "tgat")


def test_the_models_of_the_roots_own_inputs_fetch_no_children() -> None:
    assert {name: FeaturePlan(CORE_GROUPS, name).root_only for name in ARCHITECTURES} == {
        "tgat": False,
        "summary": True,
        "linear": True,
        "wide_and_deep": False,
    }


def test_extraction_plan_is_the_model_groups_without_the_client_groups() -> None:
    from mule_pattern_learner.contract.feature_groups import extraction_plan

    built_in = extraction_plan(FeaturePlan())
    assert set(built_in.groups) == set(CORE_GROUPS) - {"hub_indicator"}
    assert built_in.architecture == "tgat"
    summary = extraction_plan(FeaturePlan(CORE_GROUPS, "summary"))
    assert summary.architecture == "summary" and set(summary.groups) == set(built_in.groups)


@pytest.mark.parametrize("group", list(FEATURE_GROUPS))
def test_each_group_has_consistent_transport_batch_and_model_width(group: str) -> None:
    groups = set(CORE_GROUPS) | {group} | set(FEATURE_GROUPS[group].requires)
    plan = FeaturePlan(tuple(g for g in FEATURE_GROUPS if g in groups), "tgat")
    root = ContextKey("Account", "root", 100, 1000)
    msg = message(80, 800, root)
    # A real zero-gap predecessor remains distinguishable from no predecessor.
    msg.update(
        pair_prior_count=1,
        pair_first_age_seconds=0,
        pair_first_present=True,
        flow_delay_seconds=0,
        flow_present=False,
        flow_censored=False,
        flow_observation_seconds=0,
        flow_amount_ratio=0,
        flow_ratio_present=False,
        flow_same_rail=False,
    )
    executor = FakeTigerGraph({root: context(root, [msg])})
    source = ContextSource(TigerGraphContextFetcher(executor), plan=plan, sampler=SamplerPlan())
    try:
        batch = build_batch(source, [root], plan=plan, sampler=SamplerPlan(fanouts=(2, 2)))
        assert batch["x"].shape[1] == len(plan.node_names)
        assert batch["first_edge"].shape[-1] == len(plan.edge_names)
        model = TGAT(16, 4, 0, plan=plan, slot_sum=False, first_fanout=8)
        output = model(batch)
        output.sum().backward()
        assert torch.isfinite(output).all()
    finally:
        source.close()


def test_registry_dependencies_fingerprints_and_unknown_fields():
    with pytest.raises(ValueError, match="dependencies"):
        FeaturePlan(("message_core", "pool_internal_inflows"))
    with pytest.raises(ValueError, match="unknown"):
        FeaturePlan(("message_core", "made_up"))
    a = FeaturePlan(CORE_GROUPS, "tgat")
    assert a.fingerprint() == FeaturePlan(tuple(reversed(CORE_GROUPS)), "tgat").fingerprint()
    assert a.fingerprint() != FeaturePlan(CORE_GROUPS, "summary").fingerprint()
    with pytest.raises(ValueError, match="Architecture must be one of"):
        FeaturePlan(CORE_GROUPS, "single")
    row = context(ContextKey("Account", "root", 100, 1000))
    row["features"]["fraud_label"] = 1
    with pytest.raises(ValueError, match="Unrecognized"):
        node_features(row, a)
