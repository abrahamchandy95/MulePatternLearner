"""Two-hop batches: resampled slots, hub stubs, rejected children and roots, Fourier columns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch, build_root_batch, child_key
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS, FeaturePlan
from mule_pattern_learner.contract.graph_schema import HUB_COLUMNS, RELATIONS, ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.data.hub_registry import HubRegistry
from mule_pattern_learner.model.tgat import TGAT
from mule_pattern_learner.testing.builders import (
    DEFAULT_TGAT_PLAN,
    POOLED,
    RESAMPLE,
    SMALL_SAMPLER,
    association,
    context,
    message,
    roots,
    slots,
)
from mule_pattern_learner.testing.fake_graph import FakeStore, FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher

MPS = torch.backends.mps.is_available()


ASSOCIATED = "Account_Owned_By_Party"


class Hubs:
    def __init__(self, stubs: Iterable[str] = ()) -> None:
        self.stubs = set(stubs)
        self.calls: list[tuple[str, str, int, int]] = []

    def is_stub(self, node_type: str, node_id: str, root_cutoff_seq: int, phase: int = 3) -> bool:
        self.calls.append((node_type, node_id, root_cutoff_seq, phase))
        return node_type == "Account" and node_id in self.stubs


def test_fourier_columns_come_from_scalar_deltas_on_every_device() -> None:
    sampler = POOLED
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    store = FakeStore(sampler, encodings=True)
    cpu = build_batch(store, roots(4), fanouts=(8, 4), plan=plan, sampler=sampler)
    start = plan.edge_names.index("age_fourier_0")
    for prefix in ("first_", "second_"):
        edge, mask = cpu[prefix + "edge"], cpu[prefix + "mask"]
        assert torch.all(edge[~mask] == 0)
        assert torch.count_nonzero(edge[mask][:, start : start + 128]) > 0
    if MPS:
        mps = build_batch(store, roots(4), fanouts=(8, 4), plan=plan, sampler=sampler, device="mps")
        for name, value in cpu.items():
            atol = 1e-5 if name.endswith("edge") else 0
            torch.testing.assert_close(mps[name].cpu(), value, atol=atol, rtol=0)


def test_resampled_batches_respect_caps_hops_and_time_in_both_modes() -> None:
    store = FakeStore(RESAMPLE)
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    keys = roots(16)
    for mode in ("train", "eval"):
        stats: dict[str, Any] = {}
        batch = build_batch(
            store,
            keys,
            fanouts=(8, 4),
            plan=plan,
            sampler=RESAMPLE,
            mode=mode,
            step_seed=7,
            stats=stats,
        )
        assert stats["sampler_backend"] == "torch"
        assert set(stats) >= {"roots", "contexts", "stub_children", "rejected_children"}
        assert stats["first_edges"] == int(batch["first_mask"].sum())
        relation = batch["first_relation"][batch["first_mask"]]
        assert relation.numel() and (relation >= 4).any()
        second = batch["second_relation"][batch["second_mask"]]
        assert second.numel() and bool((second < 4).all())
        for prefix, quota in (("first_", 3), ("second_", 2)):
            codes = batch[prefix + "relation"].masked_fill(~batch[prefix + "mask"], -1)
            for r in range(4):
                assert int((codes == r).sum(1).max()) <= quota
        assert (batch["first_mask"].sum(1) <= 8).all()
        # Hop-2 fetches use the children pool and never re-request roots.
        assert [hop for hop, _ in store.calls[-2:]] == [1, 2]
        assert not set(store.calls[-1][1]) & set(keys)
        model = TGAT(16, 4, 0, plan=plan, slot_sum=False, first_fanout=8)
        assert torch.isfinite(model(batch)).all()


def _first_children(
    store: FakeStore, keys: list[ContextKey], sampler: SamplerPlan, fanout: int = 8
) -> set[ContextKey]:
    rows = [store.row(k) for k in keys]
    return {
        child_key(m, k)
        for k, msgs in zip(keys, slots(keys, rows, sampler, fanout), strict=True)
        for m in msgs
    }


def test_hub_children_become_local_stubs_and_mark_outer_peers() -> None:
    sampler = POOLED
    plan = FeaturePlan((*DEFAULT_GROUPS, "entity_age"), "tgat")
    store = FakeStore(sampler)
    keys = roots(6)
    children = _first_children(store, keys, sampler)
    hub_ids = {k.node_id for k in children if k.node_type == "Account"}
    hub = sorted(hub_ids)[0]
    hubs = Hubs({hub})
    stats: dict[str, Any] = {}
    batch = build_batch(
        store, keys, fanouts=(8, 4), plan=plan, sampler=sampler, hubs=hubs, stats=stats
    )
    fetched = {k for hop, ks in store.calls if hop == 2 for k in ks}
    stubbed = {k for k in children if k.node_type == "Account" and k.node_id == hub}
    assert stubbed and not stubbed & fetched and stats["stub_children"] == len(stubbed)
    assert {cutoff for _, _, cutoff, _ in hubs.calls} == {keys[0].cutoff_seq}
    # Phase-1 roots look hub status up in phase 1 (children and outer peers alike).
    assert {phase for *_, phase in hubs.calls} == {1}
    column = plan.node_names.index("history_withheld")
    withheld = batch["x"][:, column]
    assert int(withheld.sum()) == len(stubbed)
    # A stub carries peer metadata and no history, so it has no second-hop edges.
    stub_rows = torch.nonzero(withheld == 1).flatten()
    assert not batch["second_mask"][stub_rows].any()
    reached: dict[ContextKey, dict[str, Any]] = {}
    reaching = []
    first = slots(keys, [store.row(k) for k in keys], sampler, 8)
    for i, (root, msgs) in enumerate(zip(keys, first, strict=True)):
        for j, m in enumerate(msgs):
            reached.setdefault(child_key(m, root), m)
            reaching.append((i, j, child_key(m, root)))
    for i, j, key in reaching:
        if key not in stubbed:
            continue
        message = reached[key]  # the first message that reaches a stub supplies its metadata
        row = batch["x"][batch["neighbor_positions"][i, j]]
        expected = {
            "type_Account": 1.0,
            "is_external": float(message["peer_external"]),
            "is_deposit": float(message["peer_deposit"]),
            "age_days": float(np.log1p((key.cutoff_ms - message["peer_first_ms"]) / 86_400_000)),
            "history_withheld": 1.0,
        }
        for name, value in expected.items():
            assert float(row[plan.node_names.index(name)]) == pytest.approx(value)
        assert float(row.sum()) == pytest.approx(sum(expected.values()))
    # Outermost peers get the same flag in second_x.
    base = plan.names("node").index("history_withheld")
    flagged = batch["second_x"][..., base][batch["second_mask"]]
    outer = [*keys, *(c for c in children if c not in stubbed)]
    rows = [store.row(k, 2 if k not in keys else 1) for k in outer]
    peers = [m for msgs in slots(outer, rows, sampler, 4, hop=2) for m in msgs]
    assert int(flagged.sum()) == sum(m["node_id"] == hub for m in peers) > 0
    reference = build_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler)
    assert torch.equal(reference["first_mask"], batch["first_mask"])
    assert int(reference["x"][:, column].sum()) == 0


@pytest.mark.parametrize(
    ("scope", "phase", "expected"),
    [("s", 1, 1), ("s", 2, 2), ("s", 3, 3), ("", 3, 3), ("", 1, 3)],
    ids=["train", "validation", "test", "unscoped", "unscoped-phase-ignored"],
)
def test_hub_lookups_use_the_batch_visibility_phase(scope: str, phase: int, expected: int) -> None:
    sampler = POOLED
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    store = FakeStore(sampler)
    keys = roots(4, scope=scope, phase=phase)
    hubs = Hubs()
    build_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler, hubs=hubs)
    # Children (stub decision) and outer peers (history_withheld) are both looked up.
    assert len(hubs.calls) > len(_first_children(store, keys, sampler))
    assert {p for *_, p in hubs.calls} == {expected}


def test_rejected_children_are_masked_and_rejected_roots_raise() -> None:
    sampler = POOLED
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    keys = roots(6)
    clean = FakeStore(sampler)
    children = sorted(_first_children(clean, keys, sampler))
    bad = {k for k in children if k.node_type == "Account"}
    bad = set(sorted(bad)[:3])
    store = FakeStore(sampler, reject=bad)
    stats: dict[str, Any] = {}
    batch = build_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler, stats=stats)
    good = build_batch(clean, keys, fanouts=(8, 4), plan=plan, sampler=sampler)
    assert stats["rejected_children"] == len(bad)
    assert stats["contexts"] == good["x"].shape[0] - len(bad) == batch["x"].shape[0]
    first = slots(keys, [clean.row(k) for k in keys], sampler, 8)
    for i, (key, msgs) in enumerate(zip(keys, first)):
        for j, m in enumerate(msgs):
            assert bool(batch["first_mask"][i, j]) == (child_key(m, key) not in bad)
    dropped = good["first_mask"] & ~batch["first_mask"]
    assert int(dropped.sum()) >= len(bad)
    assert torch.all(batch["first_edge"][dropped] == 0)
    assert torch.all(batch["neighbor_positions"][dropped] == 0)
    kept = batch["first_mask"]
    assert torch.equal(batch["first_edge"][kept], good["first_edge"][kept])
    with pytest.raises(ValueError, match="rejected 1 of 6 root.*history_capacity_exceeded"):
        build_batch(FakeStore(sampler, reject={keys[2]}), keys, plan=plan, sampler=sampler)


def test_tigergraph_cannot_supply_client_features() -> None:
    sampler = POOLED
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    keys = roots(3)
    store = FakeStore(sampler)
    store.row(keys[1])["features"]["history_withheld"] = 1.0
    with pytest.raises(ValueError, match="client-only"):
        build_batch(store, keys, plan=plan, sampler=sampler)
    store = FakeStore(sampler)
    child = sorted(_first_children(store, keys, sampler, fanout=8))[0]
    store.row(child, 2)["features"]["history_withheld"] = 0.0
    with pytest.raises(ValueError, match="client-only"):
        build_batch(store, keys, plan=plan, sampler=sampler)


def test_summary_models_fetch_only_roots() -> None:
    plan = FeaturePlan(("decayed_activity",), "summary")
    store = FakeStore(RESAMPLE)
    stats: dict[str, Any] = {}
    batch = build_batch(store, roots(3), plan=plan, sampler=RESAMPLE, stats=stats)
    assert set(batch) == {"root_positions", "x"} and len(store.calls) == 1
    assert stats["contexts"] == 3


def test_recursive_context_keeps_same_neighbor_at_two_different_event_times() -> None:
    root = ContextKey("Account", "root", 100, 1000)
    messages = [message(90, 900, root), message(80, 800, root)]
    source = FakeTigerGraph({root: context(root, messages)})
    store = ContextSource(TigerGraphContextFetcher(source))
    batch = build_batch(store, [root], fanouts=(2, 2))
    assert child_key(messages[0]) in source.requested
    assert child_key(messages[1]) in source.requested
    assert len(set(batch["neighbor_positions"][0].tolist())) == 2
    model = TGAT(16, 4, 0, plan=FeaturePlan(), slot_sum=False, first_fanout=8)
    logits = model(batch)
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert model.edge.weight.grad is not None
    assert torch.count_nonzero(model.edge.weight.grad[:, 7:]) > 0
    # A real zero pair gap has cosine coordinates; it is not missing time.
    gap = FeaturePlan().edge_names.index("gap_fourier_0")
    assert torch.all(batch["first_edge"][0, 0, gap + 1 : gap + 64 : 2] == 1)
    store.close()


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
    executor = FakeTigerGraph(
        {root: context(root, messages)}, statuses={"busy": "history_capacity_exceeded"}
    )
    stats: dict[str, Any] = {}
    with ContextSource(
        TigerGraphContextFetcher(executor), plan=DEFAULT_TGAT_PLAN, sampler=SMALL_SAMPLER
    ) as source:
        batch = build_batch(
            source,
            [root],
            fanouts=(8, 2),
            plan=DEFAULT_TGAT_PLAN,
            sampler=SMALL_SAMPLER,
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
    withheld = DEFAULT_TGAT_PLAN.node_names.index("history_withheld")
    stub = batch["neighbor_positions"][0][batch["first_relation"][0] == 0][0]
    assert batch["x"][stub, withheld] == 1
    assert batch["x"][batch["root_positions"][0], withheld] == 0
    model = TGAT(16, 4, 0, plan=DEFAULT_TGAT_PLAN, slot_sum=False, first_fanout=8)
    assert torch.isfinite(model(batch)).all()


def test_rejected_roots_raise_in_batches_and_are_dropped_by_root_batches() -> None:
    roots = [ContextKey("Account", f"R{i:02}", 100, 1000) for i in range(64)]
    executor = FakeTigerGraph(statuses={"R07": "missing_entity"})
    with ContextSource(
        TigerGraphContextFetcher(executor),
        plan=DEFAULT_TGAT_PLAN,
        sampler=SMALL_SAMPLER,
        request_batch_size=64,
    ) as source:
        with pytest.raises(ValueError, match="rejected 1 of 64 root contexts"):
            build_batch(source, roots, plan=DEFAULT_TGAT_PLAN, sampler=SMALL_SAMPLER)
        assert executor.names().count(CONTEXT_QUERY) == 1  # one 64-key request
        prepared = build_root_batch(
            source,
            roots,
            fanouts=(8, 4),
            device="cpu",
            plan=DEFAULT_TGAT_PLAN,
            sampler=SMALL_SAMPLER,
            hubs=HubRegistry.empty(),
            mode="eval",
        )
    assert prepared.rejected == [roots[7]]
    assert prepared.batch is not None and len(prepared.batch["root_positions"]) == 63
    assert prepared.stats["rejected_roots"] == 1
