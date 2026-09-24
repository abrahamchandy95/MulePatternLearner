"""Live temporal transport, batching and cache behaviour against fake TigerGraph endpoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.temporal.live.batching import child_key, make_live_batch, node_features
from mule_pattern_learner.temporal.live.contract import (
    DEFAULT_GROUPS,
    FEATURE_NAMES,
    ContextKey,
    RELATIONS,
    FeaturePlan,
    PoolPlan,
    SamplerPlan,
)
from mule_pattern_learner.temporal.live.dataset import assign_groups, validate_dates
from mule_pattern_learner.temporal.live.hubs import HUB_COLUMNS, HubRegistry
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live.predictor import build_root_batch
from mule_pattern_learner.temporal.live.queries import render_context_query
from mule_pattern_learner.temporal.live.source import (
    ContextStore,
    StreamingContextSource,
    validate_context,
)
from temporal_fakes import FakeExecutor, association, context, message

V5_PLAN = FeaturePlan(DEFAULT_GROUPS, "split")
ASSOCIATED = "Account_Owned_By_Party"
V5_SAMPLER = SamplerPlan(
    "resample",
    roots=PoolPlan(recent=4, older=1, distinct=1, associations=1),
    children=PoolPlan(recent=2, associations=0),
    relation_fanouts=(3, 2),
)


def test_recursive_context_keeps_same_neighbor_at_two_different_event_times(tmp_path: Path) -> None:
    root = ContextKey("Account", "root", 100, 1000)
    messages = [message(90, 900, root), message(80, 800, root)]
    source = FakeExecutor({root: context(root, messages)})
    store = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"}, source)
    batch = make_live_batch(store, [root], fanouts=(2, 2))
    assert child_key(messages[0]) in source.requested
    assert child_key(messages[1]) in source.requested
    assert len(set(batch["neighbor_positions"][0].tolist())) == 2
    model = LiveTGAT(hidden=16, heads=4, dropout=0)
    logits = model(batch)
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert model.edge.weight.grad is not None
    assert torch.count_nonzero(model.edge.weight.grad[:, 7:]) > 0
    # A real zero pair gap has cosine coordinates; it is not missing time.
    assert torch.all(batch["first_edge"][0, 0, 72::2] == 1)
    store.close()
    offline = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"})
    again = make_live_batch(offline, [root], fanouts=(2, 2))
    for key in batch:
        torch.testing.assert_close(batch[key], again[key])
    assert offline.query_calls == 0
    offline.close()


def test_future_and_same_event_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    for seq, ts in ((100, 1000), (101, 900), (90, 1001)):
        row = context(key, [message(seq, min(ts, 1000), key)])
        row["messages"][0]["event_ts_ms"] = ts
        with pytest.raises(ValueError):
            validate_context(key, row)


def test_basis_and_clock_corruption_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key, [message(90, 900, key)])
    bad = deepcopy(row)
    bad["age_encoding"]["zelle_out:E90"][0] += 0.1
    with pytest.raises(ValueError, match="encoding"):
        validate_context(key, bad)
    bad = deepcopy(row)
    bad["cutoff_seq"] = 101
    with pytest.raises(ValueError, match="differs"):
        validate_context(key, bad)
    bad = deepcopy(row)
    bad["messages"][0]["gap_present"] = False
    with pytest.raises(ValueError, match="Missing predecessor"):
        validate_context(key, bad)


def test_labels_cannot_enter_node_features() -> None:
    row = context(ContextKey("Account", "root", 100, 1000))
    assert node_features(row).shape == (len(FEATURE_NAMES),)
    row["features"]["is_mule"] = 1
    with pytest.raises(ValueError, match="Unrecognized"):
        node_features(row)


def test_amount_ratios_are_required_from_gsql_and_preserved_by_tensor_conversion() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key)
    row["features"].update({"1d_out_in_amount_ratio": 2.5, "7d_out_in_amount_ratio": 100.0})
    validate_context(key, row)
    features = node_features(row)
    assert features[FEATURE_NAMES.index("1d_out_in_amount_ratio")] == pytest.approx(np.log1p(2.5))
    assert features[FEATURE_NAMES.index("7d_out_in_amount_ratio")] == pytest.approx(np.log1p(100.0))
    del row["features"]["1d_out_in_amount_ratio"]
    with pytest.raises(ValueError, match="missing amount ratios"):
        validate_context(key, row)


def test_cache_provenance_offline_miss_and_batch_bound(tmp_path: Path) -> None:
    keys = [ContextKey("Account", str(i), 100, 1000) for i in range(35)]
    source = FakeExecutor({})
    store = ContextStore(
        tmp_path / "cache.sqlite", {"dataset": "one"}, source, request_batch_size=16
    )
    store.fetch(keys)
    assert store.query_calls == 3
    store.fetch(keys)
    assert store.query_calls == 3
    store.close()
    with pytest.raises(ValueError, match="provenance"):
        ContextStore(tmp_path / "cache.sqlite", {"dataset": "two"})
    offline = ContextStore(tmp_path / "cache.sqlite", {"dataset": "one"})
    with pytest.raises(ValueError, match="Offline"):
        offline.fetch([ContextKey("Account", "absent", 100, 1000)])
    offline.close()


def test_isolated_entities_and_model_ablations(tmp_path: Path) -> None:
    key = ContextKey("Token", "alone", 100, 1000)
    store = ContextStore(tmp_path / "cache.sqlite", {}, FakeExecutor({}))
    batch = make_live_batch(store, [key])
    assert not batch["first_mask"].any()
    for variant in ("temporal", "no_fourier", "tabular"):
        assert torch.isfinite(LiveTGAT(16, 4, 0, variant)(batch)).all()
    store.close()


def test_coowners_share_split_even_through_another_account() -> None:
    accounts = pd.DataFrame(
        {"account_id": ["a", "b", "c", "d"], "owner_ids": [["x"], ["x", "y"], ["y"], ["z"]]}
    )
    assigned = assign_groups(accounts, 42)
    assert assigned.iloc[:3]["group_id"].nunique() == 1
    assert assigned.iloc[:3]["split"].nunique() == 1


def test_query_renderer_matches_reviewed_source_and_uses_no_labels() -> None:
    root = Path(__file__).resolve().parents[2]
    text = render_context_query()
    assert re.sub(r"\s+", "", text) == re.sub(
        r"\s+", "", (root / "gsql/temporal/training_context.gsql").read_text()
    )
    for field in ("is_mule", "fraud_label", "pu_label", "ring_id", "pair_time_encoding"):
        assert field not in text
    assert "temporal_fourier64_values" in text
    assert "e.valid_from_seq <= state_seq" in text
    assert "state_seq < e.valid_to_seq" in text


def test_dates_must_have_forward_chronological_splits() -> None:
    config = {
        "dates": {"train": ["2024-07-01"], "validation": ["2024-06-01"], "test": ["2025-01-01"]}
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_dates(config)


def test_query_comparison_preserves_string_literal_case_and_spacing() -> None:
    from mule_pattern_learner.temporal.live.installation import normalized

    assert normalized('PRINT "USD";') != normalized('PRINT "usd";')
    assert normalized('PRINT "a b";') != normalized('PRINT "ab";')
    assert normalized('PRINT  "USD";') == normalized('print "USD";')


def hub_registry(
    *accounts: str, cutoff: int = 100, scope_id: str = "", phase: int = 3
) -> HubRegistry:
    """A registry whose accounts are hubs at one cutoff and phase (3 when unscoped)."""
    frame = pd.DataFrame(
        [[account, cutoff, phase, 5000, 5000, "visible_history"] for account in accounts],
        columns=list(HUB_COLUMNS),
    )
    return HubRegistry(frame, cutoff_seqs=[cutoff], threshold=2048, scope_id=scope_id)


def test_one_hub_or_rejected_child_no_longer_aborts_the_batch() -> None:
    root = ContextKey("Account", "root", 100, 1000, "strict", 1)
    messages = [
        message(90, 900, root, node_id="hub", relation="zelle_out"),
        message(80, 800, root, node_id="busy", relation="payment_out", rail="ach"),
        message(70, 700, root, node_id="fine", relation="zelle_in"),
        association(root),
    ]
    executor = FakeExecutor(
        {root: context(root, messages)}, statuses={"busy": "history_capacity_exceeded"}
    )
    stats: dict[str, Any] = {}
    with StreamingContextSource(executor, plan=V5_PLAN, sampler=V5_SAMPLER) as source:
        batch = make_live_batch(
            source,
            [root],
            fanouts=(8, 2),
            plan=V5_PLAN,
            sampler=V5_SAMPLER,
            hubs=hub_registry("hub", scope_id="strict", phase=1),
            stats=stats,
        )
        assert dict(source.rejections) == {"history_capacity_exceeded": 1}
    requested = {key.node_id for key in executor.requested}
    assert "hub" not in requested and "busy" in requested and "fine" in requested
    assert (stats["stub_children"], stats["rejected_children"]) == (1, 1)
    # Stub, association and healthy child stay; only the rejected child's slot is masked.
    assert batch["first_mask"][0].sum() == 3
    kept = batch["first_relation"][0][batch["first_mask"][0]].tolist()
    assert sorted(kept) == [RELATIONS.index(r) for r in ("zelle_out", "zelle_in", ASSOCIATED)]
    withheld = V5_PLAN.node_names.index("history_withheld")
    stub = batch["neighbor_positions"][0][batch["first_relation"][0] == 0][0]
    assert batch["x"][stub, withheld] == 1
    assert batch["x"][batch["root_positions"][0], withheld] == 0
    model = LiveTGAT(16, 4, 0, plan=V5_PLAN)
    assert torch.isfinite(model(batch)).all()


def test_rejected_roots_raise_in_batches_and_are_dropped_by_root_batches() -> None:
    roots = [ContextKey("Account", f"R{i:02}", 100, 1000) for i in range(64)]
    executor = FakeExecutor(statuses={"R07": "missing_entity"})
    with StreamingContextSource(
        executor, plan=V5_PLAN, sampler=V5_SAMPLER, request_batch_size=64
    ) as source:
        with pytest.raises(ValueError, match="rejected 1 of 64 root contexts"):
            make_live_batch(source, roots, plan=V5_PLAN, sampler=V5_SAMPLER)
        assert executor.names().count("temporal_training_context") == 1  # one 64-key request
        prepared = build_root_batch(
            source,
            roots,
            fanouts=(8, 4),
            device="cpu",
            plan=V5_PLAN,
            sampler=V5_SAMPLER,
            hubs=HubRegistry.empty(),
            mode="eval",
        )
    assert prepared.rejected == [roots[7]]
    assert prepared.batch is not None and len(prepared.batch["root_positions"]) == 63
    assert prepared.stats["rejected_roots"] == 1


def test_hops_use_their_own_pools_and_only_spot_checks_carry_encodings() -> None:
    root = ContextKey("Account", "root", 100, 1000)
    many = [message(99 - i, 990 - 10 * i, root, node_id=f"p{i}") for i in range(12)]
    executor = FakeExecutor({root: context(root, many)})
    with StreamingContextSource(
        executor, plan=V5_PLAN, sampler=V5_SAMPLER, encoding_check_every=1000
    ) as source:
        batch = make_live_batch(
            source, [root], fanouts=(8, 2), plan=V5_PLAN, sampler=V5_SAMPLER, mode="train"
        )
        assert source.diagnostics["encoding_checks"] == 1
    # The root pool returns 4 + 1 + 1 per payment relation; resampling keeps 3 of them.
    assert batch["first_mask"][0].sum() == 3
    roots_pool = (4, 1, 1, 1, 2048)
    children_pool = (2, 0, 0, 0, 2048)
    assert set(executor.pools) == {roots_pool, children_pool}
    assert executor.encoded_requests == 1
    first = next(params for name, params in executor.calls if name == "temporal_training_context")
    assert first["emit_encodings"] is True and "include_hub_indicator" not in first
    children = [p for n, p in executor.calls if n == "temporal_training_context"][1:]
    assert all(not p["emit_encodings"] and not p["include_pair_window_counts"] for p in children)


def test_same_context_in_two_scopes_or_hops_is_never_shared(tmp_path: Path) -> None:
    key = ContextKey("Account", "a", 100, 1000, "strict", 1)
    executor = FakeExecutor()
    store = ContextStore(tmp_path / "cache.sqlite", {}, executor, plan=V5_PLAN, sampler=V5_SAMPLER)
    store.fetch([key], hop=1)
    store.fetch([key], hop=2)
    store.fetch([replace(key, visibility_phase=2)], hop=1)
    store.fetch([key], hop=1)
    assert store.query_calls == 3
    store.close()
    bad = deepcopy(context(key))
    bad["features"]["history_withheld"] = 1
    with pytest.raises(ValueError, match="Unknown node feature"):
        validate_context(key, bad, V5_PLAN, V5_SAMPLER)
    assert np.isfinite(node_features(context(key), V5_PLAN)).all()


def test_extraction_plan_ignores_client_groups_and_keeps_the_model_architecture() -> None:
    from mule_pattern_learner.temporal.live.contract import LEGACY_GROUPS
    from mule_pattern_learner.temporal.live.source import extraction_plan

    superset = sorted({*LEGACY_GROUPS, *DEFAULT_GROUPS} - {"hub_indicator"})
    split = extraction_plan(
        {
            "feature_groups": list(DEFAULT_GROUPS),
            "architecture": "split",
            "extraction_groups": superset,
        }
    )
    assert "hub_indicator" not in split.groups and split.architecture == "split"
    assert not split.query_flags(2)["include_rolling_windows"]
    single = extraction_plan({"feature_groups": list(LEGACY_GROUPS), "extraction_groups": superset})
    # A single model reads child summaries, so its second hop keeps the summary flags.
    assert single.architecture == "single" and single.query_flags(2)["include_rolling_windows"]
    assert "hub_indicator" not in extraction_plan({"feature_groups": list(DEFAULT_GROUPS)}).groups
    with pytest.raises(ValueError, match="pair_history"):
        extraction_plan(
            {"feature_groups": list(DEFAULT_GROUPS), "extraction_groups": list(LEGACY_GROUPS)}
        )


def test_feature_arms_and_model_seeds_share_one_streamed_preparation() -> None:
    from mule_pattern_learner.temporal.live.contract import LEGACY_GROUPS
    from mule_pattern_learner.temporal.live.dataset import preparation_view
    from mule_pattern_learner.temporal.live.experiments import feature_experiments
    from temporal_fakes import live_config

    groups = sorted({*LEGACY_GROUPS, *DEFAULT_GROUPS, "event_channel", "decayed_activity"})
    groups += ["history_support", "identity_order", "device_ip_context"]
    base = live_config(context_storage="stream", extraction_groups=groups)
    views = {json_key(preparation_view(arm)) for arm in feature_experiments(base).values()}
    assert len(views) == 1  # single, split and summary arms all fit the same preparation
    # Model variants read fewer inputs but keep the configured extraction.
    variants = {
        json_key(preparation_view({**base, "variant": v})) for v in ("no_fourier", "tabular")
    }
    assert variants == views
    reseeded = {**base, "seed": 7, "cohort_seed": base["seed"]}
    assert preparation_view(reseeded) == preparation_view(base)
    assert preparation_view({**base, "seed": 7})["cohort_seed"] == 7
    # A SQLite cache holds one architecture's hop-2 features, so there it is recorded.
    sqlite = {**base, "context_storage": "sqlite"}
    single = {**sqlite, "feature_groups": list(LEGACY_GROUPS), "architecture": "single"}
    assert (
        preparation_view(single)["sqlite_selection"] != preparation_view(sqlite)["sqlite_selection"]
    )


def json_key(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True)
