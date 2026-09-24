from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.temporal.live.contract import (
    ContextKey,
    FeaturePlan,
    SamplerPlan,
    FEATURE_GROUPS,
    DEFAULT_GROUPS,
    LEGACY_GROUPS,
)
from mule_pattern_learner.temporal.live.batching import (
    make_live_batch,
    node_features,
    select_messages,
)
from mule_pattern_learner.temporal.live.history_reference import payment_features, stratify
from mule_pattern_learner.temporal.live.evaluation import final_evaluation_sample, evaluate_weighted
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.source import StreamingContextSource, ContextStore
from mule_pattern_learner.temporal.live.queries import render_context_query
from temporal_fakes import FakeExecutor, context, message


def event(
    seq: int,
    ts: int,
    direction: str = "in",
    peer: str = "peer",
    amount: float = 100,
    rail: str = "zelle",
    currency: str = "USD",
) -> dict[str, Any]:
    return dict(
        event_id=f"e{seq}",
        event_seq=seq,
        event_ts_ms=ts,
        relation=f"zelle_{direction}",
        node_type="Account",
        node_id=peer,
        rail=rail,
        currency=currency,
        amount=amount,
        amount_present=True,
    )


def test_timing_uses_both_clocks_and_marks_censored_receipts():
    root = ContextKey("Account", "root", 10, 10000)
    history = [
        event(1, 1000),
        event(2, 1000, "out", amount=90),
        event(3, 2000),
        event(4, 3000, "out", amount=80, rail="cash"),
        event(5, 9000),
        event(6, 11000, "out"),
        event(10, 9500, "out"),
        event(8, 9200, "out", currency="EUR"),
    ]
    rows, _ = payment_features(history, root)
    first = rows["zelle_in:e1"]
    assert first["flow_present"] and first["flow_delay_seconds"] == 0
    assert first["flow_amount_ratio"] == 0.9
    assert rows["zelle_in:e3"]["flow_delay_seconds"] == 1
    assert not rows["zelle_in:e3"]["flow_same_rail"]
    assert rows["zelle_out:e4"]["flow_delay_seconds"] == 1
    last = rows["zelle_in:e5"]
    assert last["flow_censored"] and not last["flow_present"]
    assert last["flow_observation_seconds"] == 1
    assert rows == payment_features(history[:5], root)[0]
    assert rows["zelle_in:e3"]["pair_prior_count"] == 1
    assert rows["zelle_in:e5"]["pair_first_age_seconds"] == 8


def test_decay_is_smooth_and_missing_amount_is_not_invented():
    day = 86400000
    key = ContextKey("Account", "root", 10, 2 * day)
    e = event(1, day)
    e["amount"], e["amount_present"] = 0, False
    rows, sums = payment_features([e, event(2, day + 1000, "out")], key)
    assert sums["decay_1d_in_count"] == 0.5
    assert sums["decay_1d_in_amount"] == 0
    assert not rows["zelle_in:e1"]["flow_ratio_present"]
    later = ContextKey("Account", "root", 10, 2 * day + 1000)
    assert 0 < payment_features([e], later)[1]["decay_1d_in_count"] < 0.5


def test_rank_and_peer_strata_survive_a_recent_burst_without_duplicate_events():
    sampler = SamplerPlan("stratified", 4, 3, 2, 2, 2048)
    old = [event(i, i * 1000, peer=f"p{i}") for i in range(1, 61)]
    burst = [event(i, 61000 + i, peer="burst") for i in range(61, 101)]
    rows = stratify(old + burst, sampler)
    assert len({e["event_id"] for e in rows}) == len(rows) == 9
    assert {e["stratum"] for e in rows} == {"recent", "older", "distinct"}
    assert any(e["event_seq"] < 61 for e in rows)
    selected = select_messages({"messages": rows}, 8, sampler)
    assert any(e["stratum"] == "older" for e in selected)
    assert any(e["stratum"] == "distinct" for e in selected)
    assert stratify(list(reversed(old + burst)), sampler) == rows
    with pytest.raises(ValueError, match="capacity"):
        stratify(old + burst, SamplerPlan("stratified", 4, 3, 2, 2, 32))


@pytest.mark.parametrize("group", list(FEATURE_GROUPS))
def test_each_group_has_consistent_transport_batch_and_model_width(group: str) -> None:
    groups = set(DEFAULT_GROUPS) | {group} | set(FEATURE_GROUPS[group].requires)
    plan = FeaturePlan(tuple(g for g in FEATURE_GROUPS if g in groups), "split")
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
    source = StreamingContextSource(executor, plan=plan)
    try:
        batch = make_live_batch(source, [root], fanouts=(2, 2), plan=plan)
        assert batch["x"].shape[1] == len(plan.node_names)
        assert batch["first_edge"].shape[-1] == len(plan.edge_names)
        model = LiveTGAT(16, 4, 0, plan=plan)
        output = model(batch)
        output.sum().backward()
        assert torch.isfinite(output).all()
    finally:
        source.close()


def test_zero_node_features_and_summary_only_have_no_unused_projection_or_fetches():
    root = ContextKey("Account", "root", 100, 1000)
    zero = FeaturePlan(("message_core", "time_encoding"), "split")
    source = StreamingContextSource(FakeExecutor({}), plan=zero)
    batch = make_live_batch(source, [root], plan=zero)
    assert batch["x"].shape[-1] == batch["second_x"].shape[-1] == 0
    model = LiveTGAT(16, 4, 0, plan=zero)
    assert model.node is None and model.base is None
    assert torch.isfinite(model(batch)).all()
    source.close()
    summary = FeaturePlan(("decayed_activity",), "summary")
    executor = FakeExecutor({root: context(root, [message(80, 800, root)])})
    source = StreamingContextSource(executor, plan=summary)
    batch = make_live_batch(source, [root], plan=summary)
    assert executor.requested == [root]
    assert set(batch) == {"x", "root_positions"}
    assert not hasattr(LiveTGAT(16, 4, 0, plan=summary), "edge")
    assert torch.isfinite(LiveTGAT(16, 4, 0, plan=summary)(batch)).all()
    source.close()


def test_registry_dependencies_fingerprints_and_unknown_fields():
    with pytest.raises(ValueError, match="dependencies"):
        FeaturePlan(("message_core", "amount_ratios"))
    with pytest.raises(ValueError, match="unknown"):
        FeaturePlan(("message_core", "made_up"))
    a = FeaturePlan(DEFAULT_GROUPS, "split")
    assert a.fingerprint() == FeaturePlan(tuple(reversed(DEFAULT_GROUPS)), "split").fingerprint()
    assert a.fingerprint() != FeaturePlan(LEGACY_GROUPS).fingerprint()
    assert not a.query_flags()["include_rolling_windows"]
    row = context(ContextKey("Account", "root", 100, 1000))
    row["features"]["fraud_label"] = 1
    with pytest.raises(ValueError, match="Unrecognized"):
        node_features(row, a)


def test_extraction_cache_can_be_shared_across_model_seeds_but_not_sampling_or_scope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contexts.sqlite"
    base = {"dataset_id": "frozen", "scope_id": "one", "config_sha256": "seed1"}
    ContextStore(path, base).close()
    ContextStore(path, {**base, "config_sha256": "seed2"}).close()
    with pytest.raises(ValueError, match="provenance"):
        ContextStore(path, {**base, "scope_id": "two"})
    with pytest.raises(ValueError, match="provenance"):
        ContextStore(path, base, sampler=SamplerPlan("stratified", 4, 3, 2))


def test_final_evaluation_covers_rare_positives_and_recovers_population_prevalence():
    population = pd.DataFrame({"account_id": [str(i) for i in range(10000)], "split": "test"})
    truth = population[["account_id"]].assign(is_mule=np.r_[np.ones(40, int), np.zeros(9960, int)])
    sample = final_evaluation_sample(population, truth, negative_limit=100, seed=7)
    assert len(sample) == 140 and sample.is_mule.sum() == 40
    sample["score"] = sample.is_mule * 0.8 + 0.1
    metrics = evaluate_weighted(sample, 0.5)
    assert metrics["weighted_prevalence"] == pytest.approx(0.004)
    assert metrics["estimated_population"] == pytest.approx(10000)
    assert metrics["average_precision"] == pytest.approx(1)
    with pytest.raises(ValueError, match="binary"):
        final_evaluation_sample(population, truth.iloc[1:])
    with pytest.raises(ValueError, match="test-only"):
        final_evaluation_sample(population.assign(split="train"), truth)


def test_renderer_guards_currency_and_does_not_use_a_global_reference_date():
    query = render_context_query()
    assert '"excluded_non_usd_events"' in query
    assert '"history_capacity_exceeded"' in query
    assert '"nonmonotonic_pair_clock"' in query
    assert 't.currency != "USD"),\n' not in query
    assert "now()" not in query.lower()
    assert "IF include_pair_window_counts OR" in query
    for flag in FeaturePlan(DEFAULT_GROUPS, "split").query_flags():
        assert f"BOOL {flag}" in query
