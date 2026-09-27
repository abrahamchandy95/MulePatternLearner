"""The feature group registry, query flags, extraction plans and plan fingerprints."""

from __future__ import annotations

import pytest
import torch

from mule_pattern_learner.batching.assemble import make_live_batch
from mule_pattern_learner.contract.feature_groups import (
    BUILT_IN_GROUPS,
    CLIENT_GROUPS,
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    POOL_GROUPS,
    FeaturePlan,
)
from mule_pattern_learner.contract.graph_schema import CHANNELS, ContextKey
from mule_pattern_learner.contract.server import CONTRACT_VERSION
from mule_pattern_learner.data.contexts import StreamingContextSource
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.reference.batch_features import node_features
from mule_pattern_learner.testing.builders import context, message
from mule_pattern_learner.testing.fake_graph import FakeExecutor
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

WINDOW_GROUPS = (
    "entity_meta",
    "entity_age",
    "rolling_windows",
    "recency",
    "association_counts",
    "amount_ratios",
    "message_core",
    "time_encoding",
    "pair_window_counts",
)


def test_contract_constants_and_client_groups() -> None:
    assert CONTRACT_VERSION == "temporal_live_v5_candidate_pools"
    assert CHANNELS[:4] == ("unknown", "digital", "branch_or_atm", "bank")
    assert CHANNELS[-1] == "other" and len(set(CHANNELS)) == len(CHANNELS)
    assert "event_channel" not in DEFAULT_GROUPS and "event_channel" in FEATURE_GROUPS
    assert DEFAULT_GROUPS[:2] == ("entity_meta", "hub_indicator")
    assert CLIENT_GROUPS == {"hub_indicator", *POOL_GROUPS}
    assert FEATURE_GROUPS["hub_indicator"].identity == ("history_withheld",)
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    assert plan.names("node")[-1] == "history_withheld"
    for hop in (1, 2):
        assert not any("hub" in flag for flag in plan.query_flags(hop))
    # The default plan is the built-in run's.
    assert FeaturePlan() == FeaturePlan(BUILT_IN_GROUPS, "tgat")


def test_query_flags_skip_child_summaries_only_for_tgat_models() -> None:
    groups = ("entity_meta", "entity_age", "rolling_windows", "amount_ratios", "message_core")
    groups += ("time_encoding", "flow_timing", "decayed_activity", "hub_indicator")
    tgat, summary = FeaturePlan(groups, "tgat"), FeaturePlan(groups, "summary")
    assert set(tgat.query_flags(2)) == set(tgat.query_flags(1))
    assert tgat.query_flags(1)["include_rolling_windows"]
    for name, on in tgat.query_flags(2).items():
        spec = FEATURE_GROUPS[name.removeprefix("include_")]
        assert on == (spec.path != "summary" and name.removeprefix("include_") in groups)
    assert summary.query_flags(2) == summary.query_flags(1) == tgat.query_flags()
    with pytest.raises(ValueError, match="Hop"):
        tgat.query_flags(3)


def test_extraction_plan_is_the_model_groups_without_the_client_groups() -> None:
    from mule_pattern_learner.contract.feature_groups import extraction_plan

    tgat = extraction_plan(FeaturePlan((*DEFAULT_GROUPS, "rolling_windows")))
    assert set(tgat.groups) == set(DEFAULT_GROUPS) - {"hub_indicator"} | {"rolling_windows"}
    assert tgat.architecture == "tgat" and not tgat.query_flags(2)["include_rolling_windows"]
    summary = extraction_plan(FeaturePlan(WINDOW_GROUPS, "summary"))
    assert summary.architecture == "summary" and summary.query_flags(1)["include_rolling_windows"]
    built_in = extraction_plan(FeaturePlan())
    assert set(built_in.groups) == set(DEFAULT_GROUPS) - {"hub_indicator"}


@pytest.mark.parametrize("group", list(FEATURE_GROUPS))
def test_each_group_has_consistent_transport_batch_and_model_width(group: str) -> None:
    groups = set(DEFAULT_GROUPS) | {group} | set(FEATURE_GROUPS[group].requires)
    plan = FeaturePlan(tuple(g for g in FEATURE_GROUPS if g in groups), "tgat")
    root = ContextKey("Account", "root", 100, 1000)
    msg = message(80, 800, root)
    # A real zero-gap predecessor remains distinguishable from no predecessor.
    msg.update(
        device_age_seconds=0,
        device_present=False,
        ip_age_seconds=0,
        ip_present=False,
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
        channel="p2p",
        stratum="recent",
    )
    executor = FakeExecutor({root: context(root, [msg])})
    source = StreamingContextSource(TigerGraphContextFetcher(executor), plan=plan)
    try:
        batch = make_live_batch(source, [root], fanouts=(2, 2), plan=plan)
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
        FeaturePlan(("message_core", "amount_ratios"))
    with pytest.raises(ValueError, match="unknown"):
        FeaturePlan(("message_core", "made_up"))
    a = FeaturePlan(DEFAULT_GROUPS, "tgat")
    assert a.fingerprint() == FeaturePlan(tuple(reversed(DEFAULT_GROUPS)), "tgat").fingerprint()
    assert a.fingerprint() != FeaturePlan(DEFAULT_GROUPS, "summary").fingerprint()
    with pytest.raises(ValueError, match="Architecture must be one of"):
        FeaturePlan(DEFAULT_GROUPS, "single")
    assert not a.query_flags()["include_rolling_windows"]
    row = context(ContextKey("Account", "root", 100, 1000))
    row["features"]["fraud_label"] = 1
    with pytest.raises(ValueError, match="Unrecognized"):
        node_features(row, a)
