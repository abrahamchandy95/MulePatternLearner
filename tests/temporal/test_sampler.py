"""Contract, resampling, hub stubs and vectorized assembly of live temporal batches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import copy
from dataclasses import asdict, replace
import pickle
import random
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
import pytest
import torch

from mule_pattern_learner.temporal.encoding import fourier64, fourier64_torch
from mule_pattern_learner.temporal.live import batching
from mule_pattern_learner.temporal.live.batching import (
    base_features,
    child_key,
    edge_features,
    make_live_batch,
    node_features,
    select_messages,
)
from mule_pattern_learner.temporal.live.contract import (
    ASSOCIATIONS,
    CHANNELS,
    CLIENT_GROUPS,
    CONTRACT_VERSION,
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    RAILS,
    RELATIONS,
    SELECTION_KEYS_VERSION,
    STRATA,
    ContextKey,
    FeaturePlan,
    PoolPlan,
    SamplerPlan,
    fingerprint,
)
from mule_pattern_learner.temporal.live.memory import BatchCapacityError, BatchIndex, BatchLimits
from mule_pattern_learner.temporal.live.model import LiveTGAT
from mule_pattern_learner.temporal.live import sampler as sampling
from mule_pattern_learner.temporal.live.sampler import (
    CandidateTable,
    CuGraphProbe,
    CuGraphSampler,
    fanout_array,
    graph_arrays,
    probe_cugraph,
    resolve_backend,
    select_resampled,
    selection_keys,
    splitmix64,
)
from temporal_fakes import encode

MPS = torch.backends.mps.is_available()
MS_PER_SEQ = 3_600_000  # synthetic clocks: one event sequence number per hour
TARGETS = (
    ("Account", "Party"),
    ("Token", "Party"),
    ("Account", "Token"),
    ("Device", "Party"),
    ("Device", "Account"),
    ("IP", "Party"),
    ("Address", "Party"),
)
ASSOCIATION_TARGET = {
    rel: typ for pair, types in zip(ASSOCIATIONS, TARGETS) for rel, typ in zip(pair, types)
}
PAYMENTS = RELATIONS[:4]
V4_GROUPS = tuple(g for g in FEATURE_GROUPS if g not in ("sampler_meta",))
RESAMPLE = SamplerPlan(
    "resample",
    roots=PoolPlan(recent=4, older=3, distinct=2, associations=2),
    relation_fanouts=(3, 2),
)


# Synthetic rows ------------------------------------------------------------------


def _rng(key: ContextKey, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([sampling.context_hash(key) & 0xFFFFFFFF, salt])


def payment(
    key: ContextKey, relation: str, seq: int, peer: str, stratum: str, rng: np.random.Generator
) -> dict[str, Any]:
    ts = seq * MS_PER_SEQ
    gap_present = bool(rng.integers(0, 2))
    flow = bool(rng.integers(0, 2))
    return {
        "node_type": "Account",
        "node_id": peer,
        "relation": relation,
        "rail": "zelle" if relation.startswith("zelle") else str(rng.choice(RAILS[2:])),
        "event_id": f"E{relation[0]}{seq}",
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": float(rng.integers(0, 5000)) / 7,
        "amount_present": bool(rng.integers(0, 4)),
        "age_ms": key.cutoff_ms - ts,
        "gap_ms": int(rng.integers(0, ts)) if gap_present else 0,
        "gap_present": gap_present,
        "pair_count_1h": int(rng.integers(0, 3)),
        "pair_count_1d": int(rng.integers(0, 9)),
        "pair_count_7d": int(rng.integers(0, 40)),
        "peer_first_ms": int(rng.integers(1, ts + 1)),
        "peer_external": bool(rng.integers(0, 2)),
        "peer_deposit": bool(rng.integers(0, 2)),
        "channel": str(rng.choice(["unknown", "digital", "bank", "p2p", "branch_or_atm", "odd"])),
        "stratum": stratum,
        "pair_prior_count": int(rng.integers(0, 30)),
        "pair_first_age_seconds": float(rng.integers(0, 10**6)),
        "pair_first_present": bool(rng.integers(0, 2)),
        "flow_delay_seconds": float(rng.integers(0, 5000)) if flow else 0.0,
        "flow_present": flow,
        "flow_censored": (not flow) and bool(rng.integers(0, 2)),
        "flow_observation_seconds": float(rng.integers(0, 9000)),
        "flow_amount_ratio": float(rng.random() * 3) if flow else 0.0,
        "flow_ratio_present": flow,
        "flow_same_rail": flow and bool(rng.integers(0, 2)),
        "device_age_seconds": float(rng.integers(0, 10**5)),
        "device_present": bool(rng.integers(0, 2)),
        "ip_age_seconds": float(rng.integers(0, 10**5)),
        "ip_present": bool(rng.integers(0, 2)),
    }


def association(
    key: ContextKey, relation: str, peer: str, rng: np.random.Generator
) -> dict[str, Any]:
    typ = ASSOCIATION_TARGET[relation]
    zero = {name: 0 for name in payment(key, "zelle_out", 1, "x", "recent", rng) if name}
    return zero | {
        "node_type": typ,
        "node_id": peer,
        "relation": relation,
        "rail": "unknown",
        "event_id": "",
        "event_seq": key.cutoff_seq,
        "event_ts_ms": key.cutoff_ms,
        "amount": 0.0,
        "amount_present": False,
        "gap_present": False,
        "peer_first_ms": int(rng.integers(1, key.cutoff_ms + 1)),
        "peer_external": typ == "Account" and bool(rng.integers(0, 2)),
        "peer_deposit": typ == "Account" and bool(rng.integers(0, 2)),
        "channel": "unknown",
        "stratum": "association",
        "pair_first_present": False,
        "flow_present": False,
        "flow_censored": False,
        "flow_ratio_present": False,
        "flow_same_rail": False,
        "device_present": False,
        "ip_present": False,
    }


def synthetic_row(
    key: ContextKey, pool: PoolPlan, *, encodings: bool = True, full: bool = False
) -> dict[str, Any]:
    """Deterministic context row in the TigerGraph response format for any key.

    `full` fills every relation to the pool bound (the saturated profile).
    """
    rng = _rng(key)
    messages: list[dict[str, Any]] = []
    if key.node_type == "Account":
        for relation in PAYMENTS:
            count = int(rng.integers(0, pool.recent + pool.older + pool.distinct + 1))
            count = pool.recent + pool.older + pool.distinct if full else count
            count = min(count, key.cutoff_seq - 1)
            seqs = sorted(rng.choice(np.arange(1, key.cutoff_seq), count, replace=False))[::-1]
            strata = ["recent"] * pool.recent + ["older"] * pool.older + ["distinct"] * 16
            for seq, stratum in zip(seqs, strata):
                peer = f"a{int(rng.integers(0, 12))}"
                messages.append(payment(key, relation, int(seq), peer, stratum, rng))
    for relation in RELATIONS[4:]:
        if relation.split("_")[0] != key.node_type:
            continue
        for n in range(pool.associations if full else int(rng.integers(0, pool.associations + 1))):
            messages.append(association(key, relation, f"{relation[-5:]}{n}", rng))
    rng.shuffle(messages)
    features: dict[str, float] = {
        "is_external": float(rng.integers(0, 2)),
        "is_deposit": float(rng.integers(0, 2)),
        "age_days": float(rng.random() * 400),
    }
    for name in FEATURE_NAMES[9:30]:
        features[name] = float(rng.integers(0, 20))
    features.update({"1d_out_in_amount_ratio": 1.5, "7d_out_in_amount_ratio": 0.25})
    features.update({"visible_event_count": 7.0, "decay_1d_out_count": 0.5})
    row: dict[str, Any] = {
        "node_type": key.node_type,
        "node_id": key.node_id,
        "cutoff_seq": key.cutoff_seq,
        "cutoff_ms": key.cutoff_ms,
        "scope_id": key.scope_id,
        "visibility_phase": key.visibility_phase,
        "status": "ok",
        "features": features,
        "messages": messages,
        "age_encoding": {},
        "gap_encoding": {},
    }
    return encode(row) if encodings else row


class FakeStore:
    """`ContextSource` with `fetch(keys, *, hop)`; None marks a rejected context."""

    plan: FeaturePlan
    sampler: SamplerPlan
    query_calls: int
    rejections: Counter[str]

    def __init__(
        self,
        sampler: SamplerPlan,
        *,
        reject: Iterable[ContextKey] = (),
        encodings: bool = False,
        plan: FeaturePlan = FeaturePlan(),
    ) -> None:
        self.sampler, self.reject, self.encodings = sampler, set(reject), encodings
        self.plan = plan
        self.rows: dict[tuple[int, ContextKey], dict[str, Any]] = {}
        self.calls: list[tuple[int, list[ContextKey]]] = []
        self.rejections = Counter()
        self.query_calls = 0

    def row(self, key: ContextKey, hop: int = 1) -> dict[str, Any]:
        if (hop, key) not in self.rows:
            pool = self.sampler.pool(hop)
            self.rows[hop, key] = synthetic_row(key, pool, encodings=self.encodings)
        return self.rows[hop, key]

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        self.calls.append((hop, list(keys)))
        self.query_calls += 1
        out: list[dict[str, Any] | None] = []
        for key in keys:
            if key in self.reject:
                self.rejections["history_capacity_exceeded"] += 1
                out.append(None)
            else:
                out.append(self.row(key, hop))
        return out

    def close(self) -> None: ...


class Hubs:
    def __init__(self, stubs: Iterable[str] = ()) -> None:
        self.stubs = set(stubs)
        self.calls: list[tuple[str, str, int, int]] = []

    def is_stub(self, node_type: str, node_id: str, root_cutoff_seq: int, phase: int = 3) -> bool:
        self.calls.append((node_type, node_id, root_cutoff_seq, phase))
        return node_type == "Account" and node_id in self.stubs


def roots(n: int = 8, cutoff: int = 400, scope: str = "s", phase: int = 1) -> list[ContextKey]:
    return [
        ContextKey("Account", f"r{i}", cutoff, cutoff * MS_PER_SEQ, scope, phase) for i in range(n)
    ]


def reference_batch(
    store: FakeStore,
    keys: list[ContextKey],
    fanouts: tuple[int, int],
    plan: FeaturePlan,
    sampler: SamplerPlan,
) -> dict[str, np.ndarray]:
    """The previous per-message assembly loop, on the scalar feature functions."""
    root_rows = [store.row(k) for k in keys]
    first = [select_messages(row, fanouts[0], sampler) for row in root_rows]
    lookup = BatchIndex(
        keys + [child_key(m, k) for k, msgs in zip(keys, first) for m in msgs], capacity=4096
    )
    unique = lookup.keys
    root_set = set(keys)
    contexts = [store.row(k) if k in root_set else store.row(k, 2) for k in unique]
    second = [select_messages(row, fanouts[1], sampler, second_hop=True) for row in contexts]
    arrays = {
        "root_positions": np.asarray([lookup[k] for k in keys], dtype=np.int64),
        "x": np.stack([node_features(row, plan) for row in contexts]),
        "neighbor_positions": np.zeros((len(keys), fanouts[0]), dtype=np.int64),
        "second_x": np.zeros((len(unique), fanouts[1], len(plan.names("node"))), np.float32),
    }
    for prefix, rows, messages, fanout in (
        ("first_", root_rows, first, fanouts[0]),
        ("second_", contexts, second, fanouts[1]),
    ):
        arrays[prefix + "edge"] = np.zeros((len(rows), fanout, len(plan.edge_names)), np.float32)
        for name in ("relation", "rail", "channel", "stratum"):
            arrays[prefix + name] = np.zeros((len(rows), fanout), dtype=np.int64)
        arrays[prefix + "mask"] = np.zeros((len(rows), fanout), dtype=bool)
        for i, neighbors in enumerate(messages):
            for j, m in enumerate(neighbors):
                arrays[prefix + "edge"][i, j] = edge_features(m, plan)
                arrays[prefix + "relation"][i, j] = RELATIONS.index(m["relation"])
                arrays[prefix + "rail"][i, j] = RAILS.index(m["rail"])
                channel = m.get("channel", "unknown")
                arrays[prefix + "channel"][i, j] = CHANNELS.index(
                    channel if channel in CHANNELS else "other"
                )
                arrays[prefix + "stratum"][i, j] = STRATA.index(m["stratum"])
                arrays[prefix + "mask"][i, j] = True
                if prefix == "first_":
                    arrays["neighbor_positions"][i, j] = lookup[child_key(m, keys[i])]
                else:
                    arrays["second_x"][i, j] = base_features(m, plan)
    return arrays


# Contract ------------------------------------------------------------------------


def test_contract_constants_and_client_groups() -> None:
    assert CONTRACT_VERSION == "temporal_live_v5_candidate_pools"
    assert CHANNELS[:4] == ("unknown", "digital", "branch_or_atm", "bank")
    assert CHANNELS[-1] == "other" and len(set(CHANNELS)) == len(CHANNELS)
    assert "event_channel" not in DEFAULT_GROUPS and "event_channel" in FEATURE_GROUPS
    assert DEFAULT_GROUPS[:2] == ("entity_meta", "hub_indicator")
    assert CLIENT_GROUPS == {"hub_indicator"}
    assert FEATURE_GROUPS["hub_indicator"].identity == ("history_withheld",)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    assert plan.names("node")[-1] == "history_withheld"
    for hop in (1, 2):
        assert not any("hub" in flag for flag in plan.query_flags(hop))
    # Legacy plans keep their node layout.
    assert FeaturePlan().node_names == FEATURE_NAMES


def test_query_flags_skip_child_summaries_only_for_split_models() -> None:
    groups = ("entity_meta", "entity_age", "rolling_windows", "amount_ratios", "message_core")
    groups += ("time_encoding", "flow_timing", "decayed_activity", "hub_indicator")
    split, single = FeaturePlan(groups, "split"), FeaturePlan(groups, "single")
    assert set(split.query_flags(2)) == set(split.query_flags(1))
    assert split.query_flags(1)["include_rolling_windows"]
    for name, on in split.query_flags(2).items():
        spec = FEATURE_GROUPS[name.removeprefix("include_")]
        assert on == (spec.path != "summary" and name.removeprefix("include_") in groups)
    assert single.query_flags(2) == single.query_flags(1) == single.query_flags()
    with pytest.raises(ValueError, match="Hop"):
        split.query_flags(3)


def test_sampler_plan_legacy_forms_properties_and_bounds() -> None:
    legacy = SamplerPlan("stratified", 4, 3, 2, 2, 2048)
    assert legacy == SamplerPlan("stratified", roots=PoolPlan(4, 3, 2, 2, 2048))
    assert legacy.children == legacy.roots
    assert (legacy.recent, legacy.older, legacy.distinct) == (4, 3, 2)
    assert (legacy.associations, legacy.max_history) == (2, 2048)
    assert legacy.response_bound == 64 and legacy.response_bound(2) == 64
    assert legacy.response_bound + 0 == 64 and not legacy.response_bound < 64
    assert legacy.query_params() == {
        "per_relation": 4,
        "k_old": 3,
        "k_div": 2,
        "k_assoc": 2,
        "max_history": 2048,
    }
    assert SamplerPlan(recent=5).roots == PoolPlan(recent=5)
    assert RESAMPLE.children == replace(RESAMPLE.roots, associations=0)
    assert RESAMPLE.query_params(2)["k_assoc"] == 0 and RESAMPLE.response_bound(2) == 36
    assert RESAMPLE.pool(1) is RESAMPLE.roots and RESAMPLE.pool(2) is RESAMPLE.children
    with pytest.raises(ValueError, match="Recent sampler"):
        SamplerPlan("recent", roots=PoolPlan(older=1))
    with pytest.raises(ValueError, match="do not apply"):
        SamplerPlan("stratified", relation_fanouts=(2, 2))
    with pytest.raises(ValueError, match="history sampler"):
        SamplerPlan("random")
    with pytest.raises(ValueError, match="recent"):
        PoolPlan(recent=33)
    with pytest.raises(ValueError, match="backend"):
        SamplerPlan("resample", backend="gpu")
    assert PoolPlan(associations=0).response_bound == 8
    restored = pickle.loads(pickle.dumps((RESAMPLE, RESAMPLE.response_bound)))
    assert restored == (RESAMPLE, 64) and type(restored[1]) is int
    assert copy.deepcopy(RESAMPLE) == RESAMPLE and hash(copy.deepcopy(RESAMPLE)) == hash(RESAMPLE)


def test_sampler_from_config_children_unknown_keys_and_round_trip() -> None:
    config = {
        "per_relation": 3,
        "sampler": {
            "policy": "resample",
            "older": 2,
            "relation_fanouts": [5, 2],
            "evaluation_seed": 11,
            "children": {"recent": 2, "max_history": 1024},
        },
    }
    plan = SamplerPlan.from_config(config)
    assert plan.roots == PoolPlan(3, 2, 0, 2, 2048)
    assert plan.children == PoolPlan(2, 2, 0, 0, 1024)
    assert plan.relation_fanouts == (5, 2) and plan.evaluation_seed == 11
    assert SamplerPlan.from_config({"sampler": plan.to_config()}) == plan
    assert SamplerPlan.from_config({"per_relation": 4}) == SamplerPlan(recent=4)
    with pytest.raises(ValueError, match="fanuots"):
        SamplerPlan.from_config({"sampler": {"fanuots": 1}})
    with pytest.raises(ValueError, match="sampler.children.*nope"):
        SamplerPlan.from_config({"sampler": {"children": {"nope": 1}}})


def test_sampler_fingerprints_ignore_backend_and_unused_resample_fields() -> None:
    legacy = SamplerPlan("stratified", 4, 3, 2)
    assert legacy.fingerprint() == SamplerPlan("stratified", 4, 3, 2).fingerprint()
    assert legacy.fingerprint() != replace(legacy, association_slots=1).fingerprint()
    assert RESAMPLE.fingerprint() == replace(RESAMPLE, backend="torch").fingerprint()
    assert RESAMPLE.fingerprint() != replace(RESAMPLE, evaluation_seed=1).fingerprint()
    assert (
        RESAMPLE.pool_fingerprint() == replace(RESAMPLE, relation_fanouts=(1, 1)).pool_fingerprint()
    )
    assert legacy.pool_fingerprint() != RESAMPLE.pool_fingerprint()
    # The resample key scheme is versioned; the legacy policies do not use it.
    assert SELECTION_KEYS_VERSION == 2
    unversioned = RESAMPLE.to_config() | {"children": asdict(RESAMPLE.children)}
    unversioned.pop("backend")
    assert RESAMPLE.fingerprint() == fingerprint(unversioned | {"selection_keys": 2})
    assert RESAMPLE.fingerprint() != fingerprint(unversioned)
    legacy_value = legacy.to_config() | {"children": asdict(legacy.children)}
    assert legacy.fingerprint() == fingerprint(legacy_value)


# Legacy parity and assembly ------------------------------------------------------


@pytest.mark.parametrize("policy", ["recent", "stratified"])
@pytest.mark.parametrize(
    "plan",
    [FeaturePlan(), FeaturePlan(DEFAULT_GROUPS, "split"), FeaturePlan(V4_GROUPS, "single")],
    ids=["legacy", "v5-split", "all-single"],
)
def test_legacy_policies_match_the_previous_assembly_bit_for_bit(
    policy: str, plan: FeaturePlan
) -> None:
    sampler = (
        SamplerPlan("stratified", 4, 3, 2, 2) if policy == "stratified" else SamplerPlan(recent=3)
    )
    store = FakeStore(sampler)
    keys = roots(12) + roots(2)  # duplicate roots, as PU batches draw with replacement
    batch = make_live_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler)
    expected = reference_batch(store, keys, (8, 4), plan, sampler)
    assert set(batch) == set(expected)
    for name, value in expected.items():
        assert batch[name].dtype == torch.from_numpy(value).dtype, name
        assert torch.equal(batch[name], torch.from_numpy(value)), name
    assert batch["first_mask"].sum() > len(keys) and batch["second_mask"].any()


def test_fourier_columns_come_from_scalar_deltas_on_every_device() -> None:
    sampler = SamplerPlan("stratified", 4, 3, 2, 2)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    store = FakeStore(sampler, encodings=True)
    cpu = make_live_batch(store, roots(4), fanouts=(8, 4), plan=plan, sampler=sampler)
    start = plan.edge_names.index("age_fourier_0")
    for prefix in ("first_", "second_"):
        edge, mask = cpu[prefix + "edge"], cpu[prefix + "mask"]
        assert torch.all(edge[~mask] == 0)
        assert torch.count_nonzero(edge[mask][:, start : start + 128]) > 0
    if MPS:
        mps = make_live_batch(
            store, roots(4), fanouts=(8, 4), plan=plan, sampler=sampler, device="mps"
        )
        for name, value in cpu.items():
            atol = 1e-5 if name.endswith("edge") else 0
            torch.testing.assert_close(mps[name].cpu(), value, atol=atol, rtol=0)


def test_fourier_matches_gsql_basis_and_rejects_negative() -> None:
    actual = fourier64(np.array([0, 1000, 34_560_000_000], dtype=np.int64))
    np.testing.assert_array_equal(actual[0, ::2], 0)
    np.testing.assert_array_equal(actual[0, 1::2], 1)
    frequencies = 0.125 * 16 ** (np.arange(32) / 31)
    np.testing.assert_allclose(actual[2, ::2], np.sin(2 * np.pi * frequencies), atol=1e-6)
    np.testing.assert_allclose(actual[2, 1::2], np.cos(2 * np.pi * frequencies), atol=1e-6)
    with pytest.raises(ValueError):
        fourier64(np.array([-1], dtype=np.int64))


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if MPS else []))
def test_torch_fourier_matches_numpy(device: str) -> None:
    rng = np.random.default_rng(3)
    delta = np.concatenate(
        [[0, 1, 999, 86_400_000, 34_560_000_000, 2**40], rng.integers(0, 40_000_000_000, 5000)]
    )
    expected = fourier64(delta)
    actual = fourier64_torch(torch.from_numpy(delta).to(device)).cpu().numpy()
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    grid = fourier64_torch(torch.from_numpy(delta[:6].reshape(2, 3)).to(device))
    assert grid.shape == (2, 3, 64)
    with pytest.raises(ValueError, match="Future"):
        fourier64_torch(torch.tensor([-1], device=device))
    with pytest.raises(TypeError, match="integer"):
        fourier64_torch(torch.tensor([1.5], device=device))


def test_missing_required_message_fields_raise_instead_of_defaulting() -> None:
    sampler = SamplerPlan(recent=3)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    key = roots(1)[0]
    rng = _rng(key)
    for field in ("flow_present", "pair_prior_count", "age_ms", "amount"):
        store = FakeStore(sampler)
        message = payment(key, "zelle_out", 50, "peer", "recent", rng)
        store.row(key)["messages"] = [message]
        make_live_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
        del message[field]
        with pytest.raises(ValueError, match=field):
            make_live_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
    store = FakeStore(sampler)
    store.row(key)["messages"] = [
        payment(key, "zelle_out", 50, "peer", "recent", rng) | {"age_ms": -1}
    ]
    with pytest.raises(ValueError, match="Future"):
        make_live_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)
    # Fields of groups outside the plan stay optional.
    store = FakeStore(sampler)
    message = payment(key, "zelle_out", 50, "peer", "recent", rng)
    del message["device_present"]
    store.row(key)["messages"] = [message]
    make_live_batch(store, [key], fanouts=(4, 2), plan=plan, sampler=sampler)


# Resampling semantics ------------------------------------------------------------


def _table(
    counts: dict[str, int], cutoff: int = 1000, contexts: int = 1
) -> tuple[list[ContextKey], list[dict[str, Any]]]:
    keys = [ContextKey("Account", f"c{c}", cutoff, cutoff * MS_PER_SEQ) for c in range(contexts)]
    rows: list[dict[str, Any]] = []
    for key in keys:
        rng = _rng(key, 1)
        messages = []
        for relation, count in counts.items():
            for n in range(count):
                if relation in PAYMENTS:
                    m = payment(key, relation, cutoff - 1 - n, f"p{n}", "recent", rng)
                else:
                    m = association(key, relation, f"n{n}", rng)
                messages.append(m)
        rows.append({"messages": messages})
    return keys, rows


def test_resample_caps_reserve_and_backfill_follow_the_stratified_merge() -> None:
    sampler = replace(RESAMPLE, relation_fanouts=(3, 2), association_fanout=1, association_slots=2)
    counts = {"zelle_out": 9, "zelle_in": 5, "payment_out": 1, "Account_Owned_By_Party": 3}
    counts |= {"Account_Bound_From_Token": 2, "Account_Uses_Device": 1}
    keys, rows = _table(counts)
    for seed in range(30):
        table = CandidateTable.build(keys, rows)
        slots = select_resampled(
            table, hop=1, sampler=sampler, fanout=8, mode="train", step_seed=seed
        )
        chosen = [table.messages[j] for j in slots[0] if j >= 0]
        relations = Counter(m["relation"] for m in chosen)
        assert all(relations[r] <= 3 for r in PAYMENTS)
        assert all(relations[r] <= 1 for r in RELATIONS[4:])
        # 7 payments (3+3+1) fill K - reserve = 6 slots, then 2 reserved associations.
        assert [m["relation"] in PAYMENTS for m in chosen] == [True] * 6 + [False] * 2
        assert [m["relation"] for m in chosen[:3]] == ["zelle_out", "zelle_in", "payment_out"]
        assert [m["relation"] for m in chosen[3:6]] == ["zelle_out", "zelle_in", "zelle_out"]
        assert [m["relation"] for m in chosen[6:]] == [
            "Account_Owned_By_Party",
            "Account_Bound_From_Token",
        ]
    # Few payments: associations backfill the free slots after the reserve.
    keys, rows = _table({"zelle_out": 1} | {r: 2 for r in counts if r not in PAYMENTS})
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(table, hop=1, sampler=sampler, fanout=8, mode="eval")
    chosen = [table.messages[j]["relation"] for j in slots[0] if j >= 0]
    assert chosen == [
        "zelle_out",
        "Account_Owned_By_Party",
        "Account_Bound_From_Token",
        "Account_Uses_Device",
    ]
    # Hop 2 is payments-only with the second relation fan-out.
    keys, rows = _table(counts)
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(table, hop=2, sampler=sampler, fanout=4, mode="train", step_seed=3)
    chosen = [table.messages[j]["relation"] for j in slots[0] if j >= 0]
    assert chosen == ["zelle_out", "zelle_in", "payment_out", "zelle_out"]
    slots = select_resampled(table, hop=2, sampler=sampler, fanout=64, mode="eval")
    assert Counter(table.messages[j]["relation"] for j in slots[0] if j >= 0) == {
        "zelle_out": 2,
        "zelle_in": 2,
        "payment_out": 1,
    }


def test_reserve_uses_association_slots_and_quarter_fanout_limit() -> None:
    counts = {
        "zelle_out": 20,
        "zelle_in": 20,
        "Account_Owned_By_Party": 4,
        "Account_Uses_Device": 4,
    }
    keys, rows = _table(counts)
    table = CandidateTable.build(keys, rows)
    for slots_, fanout, expected in ((2, 16, 2), (0, 16, 0), (8, 16, 4), (8, 8, 2), (8, 4, 1)):
        sampler = replace(RESAMPLE, relation_fanouts=(16, 4), association_slots=slots_)
        sampler = replace(sampler, association_fanout=4)
        slots = select_resampled(table, hop=1, sampler=sampler, fanout=fanout, mode="eval")
        chosen = [table.messages[j] for j in slots[0] if j >= 0]
        assert len(chosen) == fanout
        assert sum(not m["event_id"] for m in chosen) == min(slots_, 8, fanout // 4)
        assert sum(not m["event_id"] for m in chosen) == expected


def test_candidates_must_be_strictly_before_the_context_cutoff() -> None:
    key = ContextKey("Account", "a", 100, 1000)
    rng = _rng(key)
    ok = [
        payment(key, "zelle_out", 99, "p", "recent", rng),
        association(key, "Account_Uses_Device", "d", rng),
    ]
    table = CandidateTable.build([key], [{"messages": ok}])
    assert table.time_key.tolist() == [198, 199] and table.seed_time.tolist() == [200]
    for seq in (100, 101):
        bad = payment(key, "zelle_in", seq, "p", "recent", rng)
        with pytest.raises(ValueError, match="strictly before"):
            CandidateTable.build([key], [{"messages": [*ok, bad]}])


def test_resampled_batches_respect_caps_hops_and_time_in_both_modes() -> None:
    store = FakeStore(RESAMPLE)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    keys = roots(16)
    for mode in ("train", "eval"):
        stats: dict[str, Any] = {}
        batch = make_live_batch(
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
        model = LiveTGAT(16, 4, 0, plan=plan)
        assert torch.isfinite(model(batch)).all()


def test_eval_is_deterministic_and_device_independent() -> None:
    store = FakeStore(RESAMPLE)
    keys = roots(8)
    table = CandidateTable.build(keys, [store.row(k) for k in keys])
    expected = select_resampled(table, hop=1, sampler=RESAMPLE, fanout=8, mode="eval")
    devices = ["cpu"] + (["mps"] if MPS else [])
    for device in devices:
        for step_seed in (0, 99):
            again = select_resampled(
                table,
                hop=1,
                sampler=RESAMPLE,
                fanout=8,
                mode="eval",
                step_seed=step_seed,
                device=device,
            )
            assert np.array_equal(again, expected)
    other = replace(RESAMPLE, evaluation_seed=5)
    assert not np.array_equal(
        select_resampled(table, hop=1, sampler=other, fanout=8, mode="eval"), expected
    )
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    one = make_live_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="eval", step_seed=1)
    two = make_live_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="eval", step_seed=2)
    for name in one:
        assert torch.equal(one[name], two[name])
    if MPS:
        three = make_live_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="eval", device="mps")
        for name in one:
            if name.endswith("edge"):
                torch.testing.assert_close(three[name].cpu(), one[name], atol=1e-5, rtol=0)
            else:
                assert torch.equal(three[name].cpu(), one[name]), name


V5_RESAMPLE = SamplerPlan(
    "resample",
    roots=PoolPlan(recent=8, older=4, distinct=4, associations=2),
    children=PoolPlan(recent=4, older=2, distinct=2, associations=0),
    relation_fanouts=(8, 4),
)


def _is_prefix(table: CandidateTable, hop_one: np.ndarray, hop_two: np.ndarray) -> bool:
    """Whether a context's hop-2 draw is the first payments of its hop-1 draw."""
    payments = [int(j) for j in hop_one if j >= 0 and table.relation[j] < 4]
    picked = [int(j) for j in hop_two if j >= 0]
    return picked == payments[: len(picked)]


def test_eval_keys_mix_the_hop_and_keep_hop_one_draws() -> None:
    keys = roots(32)
    table = CandidateTable.build(
        keys, [synthetic_row(k, V5_RESAMPLE.roots, full=True) for k in keys]
    )
    for seed in (0, 7):
        # Hop 1 keeps the SplitMix64(seed, context, item) keys of the first scheme.
        state = splitmix64(splitmix64(np.uint64(seed)) ^ table.context_hash[table.context])
        legacy = torch.from_numpy(
            (splitmix64(state ^ table.item_hash) >> np.uint64(1)).astype(np.int64)
        )
        options: dict[str, Any] = {"mode": "eval", "step_seed": 3, "evaluation_seed": seed}
        assert torch.equal(selection_keys(table, hop=1, **options), legacy)
        hop_two = selection_keys(table, hop=2, **options)
        assert bool((hop_two >= 0).all()) and not bool((hop_two == legacy).any())
    # A root is selected at both hops (its hop-2 row is its hop-1 row); its eval hop-2
    # draw must not simply repeat its first hop-1 picks, as it does not in training.
    first = select_resampled(table, hop=1, sampler=V5_RESAMPLE, fanout=16, mode="eval")
    second = select_resampled(table, hop=2, sampler=V5_RESAMPLE, fanout=4, mode="eval")
    prefixes = sum(_is_prefix(table, a, b) for a, b in zip(first, second, strict=True))
    assert prefixes <= len(keys) // 4, prefixes


def test_eval_batches_draw_root_hops_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore(V5_RESAMPLE)
    keys = roots(16)
    for key in keys:
        store.rows[1, key] = synthetic_row(key, V5_RESAMPLE.roots, encodings=False, full=True)
    draws: dict[int, list[list[dict[str, Any]]]] = {}
    select = batching._select  # pyright: ignore[reportPrivateUsage]

    def recording(
        keys_: list[ContextKey], rows: list[dict[str, Any]], **kw: Any
    ) -> list[list[dict[str, Any]]]:
        chosen = select(keys_, rows, **kw)
        draws[kw["hop"]] = chosen
        return chosen

    monkeypatch.setattr(batching, "_select", recording)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    for mode in ("eval", "train"):
        make_live_batch(
            store, keys, fanouts=(16, 4), plan=plan, sampler=V5_RESAMPLE, mode=mode, step_seed=5
        )
        prefixes = 0
        for root in range(len(keys)):  # roots come first in the hop-2 context order
            payments = [m["event_id"] for m in draws[1][root] if m["event_id"]]
            picked = [m["event_id"] for m in draws[2][root]]
            prefixes += picked == payments[: len(picked)]
        assert prefixes <= len(keys) // 4, (mode, prefixes)


def _ids(table: CandidateTable, slots: np.ndarray) -> list[list[str]]:
    return [
        [table.messages[j]["event_id"] + table.messages[j]["node_id"] for j in row if j >= 0]
        for row in slots
    ]


def test_train_mode_varies_with_step_seed_and_ignores_wire_order() -> None:
    store = FakeStore(RESAMPLE)
    keys = roots(8)
    rows = [store.row(k) for k in keys]
    table = CandidateTable.build(keys, rows)
    draws = {
        seed: select_resampled(
            table, hop=1, sampler=RESAMPLE, fanout=8, mode="train", step_seed=seed
        )
        for seed in range(6)
    }
    assert len({d.tobytes() for d in draws.values()}) == 6
    shuffled = []
    for row in rows:
        messages = list(row["messages"])
        random.Random(4).shuffle(messages)
        shuffled.append({**row, "messages": messages})
    reordered = CandidateTable.build(keys, shuffled)
    again = select_resampled(
        reordered, hop=1, sampler=RESAMPLE, fanout=8, mode="train", step_seed=3
    )
    assert _ids(reordered, again) == _ids(table, draws[3])
    if MPS:
        mps = select_resampled(
            table, hop=1, sampler=RESAMPLE, fanout=8, mode="train", step_seed=3, device="mps"
        )
        assert np.array_equal(mps, draws[3])
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    a = make_live_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="train", step_seed=10)
    b = make_live_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="train", step_seed=10)
    c = make_live_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="train", step_seed=11)
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert not all(a[n].shape == c[n].shape and torch.equal(a[n], c[n]) for n in a)


def test_subset_is_uniform_without_replacement() -> None:
    n, quota, trials = 12, 3, 3000
    keys, rows = _table({"zelle_out": n})
    table = CandidateTable.build(keys, rows)
    sampler = replace(RESAMPLE, relation_fanouts=(quota, 2))
    counts = np.zeros(n)
    first = np.zeros(n)
    for seed in range(trials):
        slots = select_resampled(
            table, hop=1, sampler=sampler, fanout=8, mode="train", step_seed=seed
        )
        chosen = slots[0][slots[0] >= 0]
        assert len(chosen) == quota == len(set(chosen.tolist()))
        counts[chosen] += 1
        first[chosen[0]] += 1
    # Pearson chi-square with n-1 = 11 degrees of freedom; 31.26 is the 0.999 quantile.
    for observed, total in ((counts, trials * quota), (first, trials)):
        expected = total / n
        assert ((observed - expected) ** 2 / expected).sum() < 31.26
    # Evaluation hashes are uniform across contexts too.
    keys, rows = _table({"zelle_out": n}, contexts=trials)
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(table, hop=1, sampler=sampler, fanout=8, mode="eval")
    rank = np.asarray([[int(table.messages[j]["event_id"][2:]) for j in r[:quota]] for r in slots])
    observed = np.bincount((1000 - 1) - rank.ravel(), minlength=n)
    expected = trials * quota / n
    assert ((observed - expected) ** 2 / expected).sum() < 31.26


# Hubs and rejected children ------------------------------------------------------


def _first_children(
    store: FakeStore, keys: list[ContextKey], sampler: SamplerPlan, fanout: int = 8
) -> set[ContextKey]:
    return {child_key(m, k) for k in keys for m in select_messages(store.row(k), fanout, sampler)}


def test_hub_children_become_local_stubs_and_mark_outer_peers() -> None:
    sampler = SamplerPlan("stratified", 4, 3, 2, 2)
    plan = FeaturePlan((*DEFAULT_GROUPS, "entity_age"), "split")
    store = FakeStore(sampler)
    keys = roots(6)
    children = _first_children(store, keys, sampler)
    hub_ids = {k.node_id for k in children if k.node_type == "Account"}
    hub = sorted(hub_ids)[0]
    hubs = Hubs({hub})
    stats: dict[str, Any] = {}
    batch = make_live_batch(
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
    slots = []
    for i, root in enumerate(keys):
        for j, m in enumerate(select_messages(store.row(root), 8, sampler)):
            reached.setdefault(child_key(m, root), m)
            slots.append((i, j, child_key(m, root)))
    for i, j, key in slots:
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
    peers = [
        m
        for k in [*keys, *(c for c in children if c not in stubbed)]
        for m in select_messages(
            store.row(k, 2 if k not in keys else 1), 4, sampler, second_hop=True
        )
    ]
    assert int(flagged.sum()) == sum(m["node_id"] == hub for m in peers) > 0
    reference = make_live_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler)
    assert torch.equal(reference["first_mask"], batch["first_mask"])
    assert int(reference["x"][:, column].sum()) == 0


@pytest.mark.parametrize(
    ("scope", "phase", "expected"),
    [("s", 1, 1), ("s", 2, 2), ("s", 3, 3), ("", 3, 3), ("", 1, 3)],
    ids=["train", "validation", "test", "unscoped", "unscoped-phase-ignored"],
)
def test_hub_lookups_use_the_batch_visibility_phase(scope: str, phase: int, expected: int) -> None:
    sampler = SamplerPlan("stratified", 4, 3, 2, 2)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    store = FakeStore(sampler)
    keys = roots(4, scope=scope, phase=phase)
    hubs = Hubs()
    make_live_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler, hubs=hubs)
    # Children (stub decision) and outer peers (history_withheld) are both looked up.
    assert len(hubs.calls) > len(_first_children(store, keys, sampler))
    assert {p for *_, p in hubs.calls} == {expected}


def test_rejected_children_are_masked_and_rejected_roots_raise() -> None:
    sampler = SamplerPlan("stratified", 4, 3, 2, 2)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    keys = roots(6)
    clean = FakeStore(sampler)
    children = sorted(_first_children(clean, keys, sampler))
    bad = {k for k in children if k.node_type == "Account"}
    bad = set(sorted(bad)[:3])
    store = FakeStore(sampler, reject=bad)
    stats: dict[str, Any] = {}
    batch = make_live_batch(store, keys, fanouts=(8, 4), plan=plan, sampler=sampler, stats=stats)
    good = make_live_batch(clean, keys, fanouts=(8, 4), plan=plan, sampler=sampler)
    assert stats["rejected_children"] == len(bad)
    assert stats["contexts"] == good["x"].shape[0] - len(bad) == batch["x"].shape[0]
    first = [select_messages(clean.row(k), 8, sampler) for k in keys]
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
        make_live_batch(FakeStore(sampler, reject={keys[2]}), keys, plan=plan, sampler=sampler)


def test_tigergraph_cannot_supply_client_features() -> None:
    sampler = SamplerPlan("stratified", 4, 3, 2, 2)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    keys = roots(3)
    store = FakeStore(sampler)
    store.row(keys[1])["features"]["history_withheld"] = 1.0
    with pytest.raises(ValueError, match="client-only"):
        make_live_batch(store, keys, plan=plan, sampler=sampler)
    store = FakeStore(sampler)
    child = sorted(_first_children(store, keys, sampler, fanout=8))[0]
    store.row(child, 2)["features"]["history_withheld"] = 0.0
    with pytest.raises(ValueError, match="client-only"):
        make_live_batch(store, keys, plan=plan, sampler=sampler)


def test_batch_limits_bound_candidate_pools_before_fetching() -> None:
    wide = SamplerPlan("resample", roots=PoolPlan(32, 16, 16, 8), relation_fanouts=(8, 4))
    store = FakeStore(wide)
    with pytest.raises(BatchCapacityError, match="Candidate pools"):
        make_live_batch(store, roots(120), fanouts=(16, 4), sampler=wide)
    assert not store.calls
    BatchLimits().validate(120, (16, 4), FeaturePlan(), SamplerPlan())
    BatchLimits().validate(120, (16, 4), FeaturePlan(), replace(wide, children=PoolPlan()))


def test_summary_models_fetch_only_roots() -> None:
    plan = FeaturePlan(("decayed_activity",), "summary")
    store = FakeStore(RESAMPLE)
    stats: dict[str, Any] = {}
    batch = make_live_batch(store, roots(3), plan=plan, sampler=RESAMPLE, stats=stats)
    assert set(batch) == {"root_positions", "x"} and len(store.calls) == 1
    assert stats["contexts"] == 3


# cuGraph with a mocked pylibcugraph ----------------------------------------------


class MockPLC:
    """Emulates the pylibcugraph calls CuGraphSampler makes, with their dtype rules.

    Batch ids are numbered as pylibcugraph 26.08 numbers them: by rank among the
    labels that got at least one edge. `seed_labels` returns the label itself.

    Faults: `leak` returns future edges, `drop` applies an off-by-one hop-0 time
    filter (drops same-cutoff associations), `jitter` ignores random_state and
    `broken` fails in SGGraph like a GPU without kernels for its architecture.
    """

    def __init__(
        self,
        version: str = "26.08.00",
        unified: bool = False,
        leak: bool = False,
        *,
        drop: bool = False,
        jitter: bool = False,
        broken: bool = False,
        seed_labels: bool = False,
    ) -> None:
        self.__version__, self.leak, self.drop = version, leak, drop
        self.jitter, self.broken, self.seed_labels = jitter, broken, seed_labels
        self.calls: list[dict[str, Any]] = []
        if unified:
            self.neighbor_sample = self._unified
        else:
            self.heterogeneous_uniform_temporal_neighbor_sample = self._legacy

    @staticmethod
    def ResourceHandle(handle: Any = None) -> SimpleNamespace:
        return SimpleNamespace(kind="handle")

    @staticmethod
    def GraphProperties(is_symmetric: bool = False, is_multigraph: bool = False) -> SimpleNamespace:
        return SimpleNamespace(symmetric=is_symmetric, multigraph=is_multigraph)

    def SGGraph(
        self, handle: Any, properties: Any, src: Any, dst: Any, **kw: Any
    ) -> SimpleNamespace:
        if self.broken:
            raise RuntimeError("CUDA error: no kernel image is available for execution")
        src, dst = np.asarray(src), np.asarray(dst)
        assert src.dtype == dst.dtype == kw["vertices_array"].dtype == kw["edge_id_array"].dtype
        assert kw["edge_type_array"].dtype == np.int32
        assert kw["edge_start_time_array"].dtype == np.int64
        assert np.array_equal(kw["vertices_array"], np.arange(len(kw["vertices_array"])))
        assert kw["renumber"] is False and kw["weight_array"] is None
        return SimpleNamespace(src=src, dst=dst, **kw)

    def _sample(
        self,
        graph: Any,
        seeds: Any,
        times: Any,
        labels: Any,
        fan: Any,
        *,
        num_edge_types: int,
        random_state: int,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        self.calls.append(dict(kw, num_edge_types=num_edge_types, random_state=random_state))
        assert np.asarray(seeds).dtype == graph.src.dtype
        assert np.asarray(times).dtype == np.int64 and np.asarray(labels).dtype == np.int64
        assert isinstance(fan, np.ndarray) and fan.dtype == np.int32
        assert len(fan) % num_edge_types == 0 and num_edge_types > 1
        assert kw["temporal_sampling_comparison"] == "strictly_decreasing"
        assert kw["with_replacement"] is False and kw["compression"] == "COO"
        assert labels[-1] == len(seeds)
        rng = np.random.default_rng(random_state + len(self.calls) * self.jitter)
        names = ("majors", "minors", "edge_id", "edge_type", "edge_start_time", "batch_id")
        out: dict[str, list[int]] = {k: [] for k in names}
        etype, etime = graph.edge_type_array, graph.edge_start_time_array
        for label, seed in enumerate(seeds):
            for t in range(num_edge_types):
                edges = np.nonzero((graph.src == seed) & (etype == t))[0]
                if not self.leak:
                    edges = edges[etime[edges] < times[label] - int(self.drop)]
                take = rng.permutation(edges)[: fan[t]] if fan[t] >= 0 else edges
                for e in take:
                    out["majors"].append(seed)
                    out["minors"].append(graph.dst[e])
                    out["edge_id"].append(graph.edge_id_array[e])
                    out["edge_type"].append(t)
                    out["edge_start_time"].append(etime[e] + (2 if self.leak else 0))
                    out["batch_id"].append(label)
        if not self.seed_labels:
            out["batch_id"] = np.unique(out["batch_id"], return_inverse=True)[1].tolist()
        dtypes = {"edge_type": np.int32, "edge_start_time": np.int64, "batch_id": np.int32}
        return {k: np.asarray(v, dtype=dtypes.get(k, graph.src.dtype)) for k, v in out.items()}

    def _legacy(
        self,
        handle: Any,
        graph: Any,
        prop: Any,
        seeds: Any,
        times: Any,
        labels: Any,
        vtypes: Any,
        fan: Any,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        assert prop is None and vtypes is None and kw.pop("disjoint_sampling") is False
        return self._sample(graph, seeds, times, labels, fan, **kw)

    def _unified(
        self,
        handle: Any,
        graph: Any,
        seeds: Any,
        fan: Any,
        *,
        starting_vertex_end_times: Any,
        starting_vertex_label_offsets: Any,
        disjoint_sampling: bool,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        assert disjoint_sampling is True
        return self._sample(
            graph, seeds, starting_vertex_end_times, starting_vertex_label_offsets, fan, **kw
        )


def host(values: np.ndarray, device: Any = None) -> np.ndarray:
    return np.ascontiguousarray(values)


def test_graph_arrays_and_fanout_layout() -> None:
    keys, rows = _table({"zelle_out": 3, "payment_in": 2, "Account_Uses_Device": 2}, contexts=3)
    table = CandidateTable.build(keys, rows)
    arrays = graph_arrays(table)
    assert arrays.src.dtype == arrays.dst.dtype == arrays.vertices.dtype == np.int32
    assert arrays.edge_type.dtype == np.int32 and arrays.edge_time.dtype == np.int64
    assert arrays.seed_time.dtype == arrays.label_offsets.dtype == np.int64
    assert arrays.dst.tolist() == list(range(3, 3 + len(table)))
    assert arrays.label_offsets.tolist() == [0, 1, 2, 3]
    assert np.all(arrays.edge_time < arrays.seed_time[arrays.src])
    quotas = [[1, 2, 3], [4, 5, 6]]
    layout = fanout_array(quotas)
    assert layout.dtype == np.int32 and layout.tolist() == [1, 2, 3, 4, 5, 6]
    assert layout[1 * 3 + 2] == quotas[1][2]  # hop * num_edge_types + edge_type
    with pytest.raises(ValueError):
        fanout_array([[1, 2], [3]])


@pytest.mark.parametrize("unified", [False, True], ids=["26.08", "26.10"])
def test_cugraph_sampler_maps_results_to_slots_with_the_torch_merge(unified: bool) -> None:
    plc = MockPLC("26.10.00" if unified else "26.08.00", unified=unified)
    engine = CuGraphSampler(plc, to_device=host)
    counts = {"zelle_out": 7, "zelle_in": 4, "payment_out": 2, "Account_Owned_By_Party": 3}
    keys, rows = _table(counts | {"Account_Uses_Device": 2}, contexts=5)
    table = CandidateTable.build(keys, rows)
    sampler = replace(RESAMPLE, relation_fanouts=(3, 2), backend="cugraph")
    for seed in range(5):
        slots = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=8,
            mode="train",
            step_seed=seed,
            backend="cugraph",
            device="cpu",
            cugraph=engine,
        )
        for c, row in enumerate(slots):
            chosen = [table.messages[j] for j in row if j >= 0]
            assert all(table.context[j] == c for j in row if j >= 0)
            relations = Counter(m["relation"] for m in chosen)
            assert all(relations[r] <= 3 for r in PAYMENTS)
            assert all(relations[r] <= 1 for r in RELATIONS[4:])
            assert [m["relation"] in PAYMENTS for m in chosen] == [True] * 6 + [False] * 2
    assert [call["random_state"] for call in plc.calls] == [0, 1, 2, 3, 4]
    assert all(call["num_edge_types"] == len(RELATIONS) for call in plc.calls)
    hop2 = select_resampled(
        table,
        hop=2,
        sampler=sampler,
        fanout=4,
        mode="train",
        step_seed=1,
        backend="cugraph",
        device="cpu",
        cugraph=engine,
    )
    assert all(table.relation[j] < 4 for j in hop2.ravel() if j >= 0)
    # Evaluation never calls cuGraph.
    before = len(plc.calls)
    table_eval = CandidateTable.build(keys, rows)
    select_resampled(
        table_eval, hop=1, sampler=sampler, fanout=8, mode="eval", backend="cugraph", cugraph=engine
    )
    assert len(plc.calls) == before


def test_cugraph_sampler_rejects_leaks_bad_versions_and_quota_overruns() -> None:
    keys, rows = _table({"zelle_out": 4, "Account_Uses_Device": 1})
    table = CandidateTable.build(keys, rows)
    quotas = sampling.relation_quotas(RESAMPLE, 1)
    leaky = CuGraphSampler(MockPLC(leak=True), to_device=host)
    with pytest.raises(RuntimeError, match="temporal leakage"):
        leaky.subset(table, quotas, random_state=1, device="cpu")
    greedy = CuGraphSampler(MockPLC(), to_device=host)
    with pytest.raises(RuntimeError, match="fan-out"):
        greedy.subset(table, np.full_like(quotas, -1), random_state=1, device="cpu")
    mask = greedy.subset(table, quotas, random_state=1, device="cpu")
    assert mask.sum() == 3 + 1
    assert (
        not CuGraphSampler(MockPLC(), to_device=host)
        .subset(
            CandidateTable.build(keys, [{"messages": []}]), quotas, random_state=1, device="cpu"
        )
        .any()
    )
    fake = SimpleNamespace(__version__="25.12.00")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(__import__("sys").modules, "pylibcugraph", fake)
        with pytest.raises(RuntimeError, match="predates"):
            sampling.import_pylibcugraph()


def test_cugraph_sampler_rejects_under_sampling() -> None:
    # An off-by-one hop-0 time filter drops the same-cutoff association silently.
    keys, rows = _table({"zelle_out": 2, "Account_Uses_Device": 1})
    table = CandidateTable.build(keys, rows)
    quotas = sampling.relation_quotas(RESAMPLE, 1)
    dropping = CuGraphSampler(MockPLC(drop=True), to_device=host)
    with pytest.raises(RuntimeError, match="under-sampled"):
        dropping.subset(table, quotas, random_state=1, device="cpu")
    torch_keep = sampling.TorchGroupedSampler().subset(
        table, quotas, selection_keys(table, mode="train", step_seed=1, evaluation_seed=0, hop=1)
    )
    assert bool(torch_keep.all())
    # Rows at or after the cutoff (the verify script's boundary table) are not expected.
    cutoff = 100
    key = ContextKey("Account", "boundary", cutoff, cutoff * 1000)
    boundary = CandidateTable(
        keys=(key,),
        context=np.zeros(4, dtype=np.int64),
        relation=np.asarray([0, 4, 0, 0], dtype=np.int64),
        time_key=np.asarray([198, 199, 200, 201], dtype=np.int64),
        seed_time=np.asarray([200], dtype=np.int64),
        messages=tuple({"relation": RELATIONS[r]} for r in (0, 4, 0, 0)),
    )
    engine = CuGraphSampler(MockPLC(), to_device=host)
    mask = engine.subset(boundary, np.full(len(RELATIONS), 8), random_state=3, device="cpu")
    assert mask.tolist() == [True, True, False, False]
    assert sampling.expected_counts(boundary, np.full(len(RELATIONS), 8))[0, :5].tolist() == [
        1,
        0,
        0,
        0,
        1,
    ]


@pytest.mark.parametrize("unified", [False, True], ids=["26.08", "26.10"])
def test_cugraph_probe_passes_on_a_correct_sampler(unified: bool) -> None:
    plc = MockPLC("26.10.00" if unified else "26.08.00", unified=unified)
    assert probe_cugraph("cpu", CuGraphSampler(plc, to_device=host)) is None
    # Two draws per hop, one random_state per hop, the association fan-out 0 at hop 2.
    assert len(plc.calls) == 4
    states = [call["random_state"] for call in plc.calls]
    assert states[0] == states[1] != states[2] == states[3]


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ({"broken": True}, "no kernel image"),
        ({"drop": True}, "under-sampled"),
        ({"jitter": True}, "different subsets"),
        ({"leak": True}, "temporal leakage"),
    ],
)
def test_cugraph_probe_reports_runtime_failures(fault: dict[str, Any], reason: str) -> None:
    engine = CuGraphSampler(MockPLC(**fault), to_device=host)
    failure = probe_cugraph("cpu", engine)
    assert failure is not None and reason in failure


def _fake_probes(monkeypatch: pytest.MonkeyPatch, probe: CuGraphProbe) -> list[int]:
    """Replace the device probe; returns the device indices it was run for."""
    calls: list[int] = []

    def fake(index: int) -> CuGraphProbe:
        calls.append(index)
        return probe

    monkeypatch.setattr(sampling, "_PROBES", {})
    monkeypatch.setattr(sampling, "_probe_device", fake)
    return calls


def test_auto_backend_falls_back_to_torch_when_the_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = CuGraphProbe(False, True, "RuntimeError: CUDA error: no kernel image")
    calls = _fake_probes(monkeypatch, broken)
    with pytest.warns(RuntimeWarning, match="no kernel image.*torch sampler"):
        assert resolve_backend(RESAMPLE, "cuda:1") == "torch"
    with pytest.raises(RuntimeError, match="cannot run on cuda:1: RuntimeError: CUDA error"):
        resolve_backend(replace(RESAMPLE, backend="cugraph"), "cuda:1")
    assert calls == [1]  # probed once per process and device, then cached
    # cuGraph not installed at all: torch without a warning.
    _fake_probes(monkeypatch, CuGraphProbe(False, False, "ModuleNotFoundError: cupy"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_backend(RESAMPLE, "cuda:0") == "torch"
    calls = _fake_probes(monkeypatch, CuGraphProbe(True, True, "ok"))
    assert resolve_backend(RESAMPLE, "cuda:0") == "cugraph"
    assert resolve_backend(replace(RESAMPLE, backend="cugraph"), "cuda:0") == "cugraph"
    assert resolve_backend(replace(RESAMPLE, backend="torch"), "cuda:0") == "torch"
    assert calls == [0]


def test_real_probe_without_cupy_reports_the_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "cupy", None)
    monkeypatch.setattr(sampling, "_PROBES", {})
    probe = sampling.cugraph_usable(0)
    assert not probe.usable and not probe.installed and "cupy" in probe.reason
    assert sampling.cugraph_usable(0) is probe


def test_backend_resolution_without_cuda() -> None:
    assert resolve_backend(RESAMPLE, "cpu") == "torch"
    assert resolve_backend(replace(RESAMPLE, backend="torch"), "cuda") == "torch"
    assert resolve_backend(SamplerPlan("stratified", 4, 3, 2), "cuda") == "deterministic"
    with pytest.raises(RuntimeError, match="cugraph needs a CUDA device"):
        resolve_backend(replace(RESAMPLE, backend="cugraph"), "cpu")
    stats: dict[str, Any] = {}
    make_live_batch(
        FakeStore(RESAMPLE),
        roots(2),
        plan=FeaturePlan(DEFAULT_GROUPS, "split"),
        sampler=SamplerPlan("stratified", 4, 3, 2),
        stats=stats,
    )
    assert stats["sampler_backend"] == "deterministic"


def test_make_live_batch_uses_the_run_backend_without_resolving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any) -> str:
        raise AssertionError("make_live_batch resolved the backend again")

    monkeypatch.setattr(batching, "resolve_backend", unexpected)
    plan = FeaturePlan(DEFAULT_GROUPS, "split")
    store = FakeStore(RESAMPLE)
    keys = roots(4)
    options: dict[str, Any] = {"plan": plan, "sampler": RESAMPLE, "step_seed": 3}
    for mode, given, used in (
        ("train", "torch", "torch"),
        ("eval", "torch", "torch"),
        ("eval", "cugraph", "torch"),  # evaluation is hash-keyed on the torch path
    ):
        stats: dict[str, Any] = {}
        make_live_batch(store, keys, mode=mode, sampler_backend=given, stats=stats, **options)
        assert stats["sampler_backend"] == used
    with pytest.raises(ValueError, match="cugraph needs a CUDA batch device"):
        make_live_batch(store, keys, mode="train", sampler_backend="cugraph", **options)
    with pytest.raises(ValueError, match="does not fit"):
        make_live_batch(store, keys, mode="train", sampler_backend="deterministic", **options)
    pinned: dict[str, Any] = options | {"sampler": replace(RESAMPLE, backend="torch")}
    with pytest.raises(ValueError, match="does not fit"):
        make_live_batch(store, keys, mode="eval", sampler_backend="cugraph", **pinned)
    stratified = SamplerPlan("stratified", 4, 3, 2)
    stats = {}
    make_live_batch(
        FakeStore(stratified),
        keys,
        plan=plan,
        sampler=stratified,
        sampler_backend="deterministic",
        stats=stats,
    )
    assert stats["sampler_backend"] == "deterministic"
    with pytest.raises(ValueError, match="does not fit"):
        make_live_batch(FakeStore(stratified), keys, sampler=stratified, sampler_backend="torch")
    # The resolved backend gives the same batch as resolving per call.
    monkeypatch.undo()
    a = make_live_batch(store, keys, mode="train", **options)
    b = make_live_batch(store, keys, mode="train", sampler_backend="torch", **options)
    assert all(torch.equal(a[n], b[n]) for n in a)


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        sampling.import_pylibcugraph()
        import cupy  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gpu_ready(), reason="needs CUDA, cupy and pylibcugraph>=26.4")
def test_real_cugraph_matches_torch_caps_on_gpu() -> None:
    keys, rows = _table({"zelle_out": 9, "zelle_in": 3, "Account_Owned_By_Party": 3}, contexts=64)
    table = CandidateTable.build(keys, rows)
    sampler = replace(RESAMPLE, backend="cugraph")
    for seed in (1, 2):
        a = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=8,
            mode="train",
            step_seed=seed,
            backend="cugraph",
            device="cuda",
        )
        b = select_resampled(
            table,
            hop=1,
            sampler=sampler,
            fanout=8,
            mode="train",
            step_seed=seed,
            backend="cugraph",
            device="cuda",
        )
        assert np.array_equal(a, b)
        torch_slots = select_resampled(
            table, hop=1, sampler=sampler, fanout=8, mode="train", step_seed=seed
        )
        # Both backends fill 3 + 3 payment slots and 1 association (the fan-out of 1).
        assert np.array_equal((a >= 0).sum(1), (torch_slots >= 0).sum(1))
        assert ((a >= 0).sum(1) == 7).all()


@pytest.mark.skipif(not _gpu_ready(), reason="needs CUDA, cupy and pylibcugraph>=26.4")
def test_real_cugraph_handles_seeds_without_edges_on_gpu() -> None:
    # The shape of a hop-2 table: empty and association-only contexts between others.
    keys, rows = _table({"zelle_out": 5, "payment_in": 1, "Account_Uses_Device": 2}, contexts=40)
    for c in range(0, 40, 3):
        rows[c] = {"messages": []}
    for c in range(1, 40, 7):
        rows[c] = {
            "messages": [m for m in rows[c]["messages"] if m["relation"] == "Account_Uses_Device"]
        }
    table = CandidateTable.build(keys, rows)
    engine = CuGraphSampler()
    for hop in (1, 2):
        quotas = sampling.relation_quotas(RESAMPLE, hop)
        keep = engine.subset(table, quotas, random_state=hop, device="cuda")
        expected = sampling.expected_counts(table, quotas)
        assert np.array_equal(sampling.group_counts(table, keep), expected)


def test_gpu_self_test_script_runs_its_synthetic_checks_on_the_mock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/temporal/verify_cugraph_sampler.py"
    spec = importlib.util.spec_from_file_location("verify_cugraph_sampler", path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    engine = CuGraphSampler(MockPLC(), to_device=host)
    sampler = replace(RESAMPLE, relation_fanouts=(8, 4))
    table = script.synthetic_table(24, np.random.default_rng(0))
    script.probe(engine, "cpu")
    script.subset_checks(engine, table, sampler, "cpu")
    script.temporal_boundary(engine, "cpu")
    script.merged_slots(engine, sampler, "cpu")
    script.uniformity(engine, 150, "cpu")
    assert not script.FAILURES, capsys.readouterr().out
    leaky = CuGraphSampler(MockPLC(leak=True), to_device=host)
    script.temporal_boundary(leaky, "cpu")
    assert script.FAILURES
    script.FAILURES.clear()
    script.probe(CuGraphSampler(MockPLC(drop=True), to_device=host), "cpu")
    assert len(script.FAILURES) == 1 and "under-sampled" in script.FAILURES[0]


def test_sampled_rows_rejects_inconsistent_results() -> None:
    keys, rows = _table({"zelle_out": 3})
    arrays = graph_arrays(CandidateTable.build(keys, rows))
    good = {
        "edge_id": np.asarray([0, 2], dtype=np.int32),
        "majors": np.asarray([0, 0], dtype=np.int32),
        "minors": np.asarray([1, 3], dtype=np.int32),
        "edge_start_time": arrays.edge_time[[0, 2]],
        "batch_id": np.asarray([0, 0], dtype=np.int32),
    }
    assert sampling.sampled_rows(good, arrays, torch.device("cpu")).tolist() == [0, 2]
    for name, value in (
        ("minors", np.asarray([1, 2], dtype=np.int32)),
        ("edge_id", np.asarray([0, 9], dtype=np.int32)),
        ("batch_id", np.asarray([0, 1], dtype=np.int32)),
        ("edge_id", np.asarray([2, 2], dtype=np.int32)),
    ):
        broken = dict(good, **{name: value})
        if name == "edge_id" and value.tolist() == [2, 2]:
            broken["minors"] = np.asarray([3, 3], dtype=np.int32)
            broken["edge_start_time"] = arrays.edge_time[[2, 2]]
        with pytest.raises(RuntimeError):
            sampling.sampled_rows(broken, arrays, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="lacks edge_start_time"):
        sampling.sampled_rows(dict(good, edge_start_time=None), arrays, torch.device("cpu"))


def test_sampled_rows_accepts_batch_ids_ranked_over_seeds_with_edges() -> None:
    # Seed 1 has no candidates, so pylibcugraph 26.08 numbers seed 2's batch 1, not 2.
    keys, rows = _table({"zelle_out": 2}, contexts=3)
    rows[1] = {"messages": []}
    arrays = graph_arrays(CandidateTable.build(keys, rows))
    result = {
        "edge_id": np.asarray([0, 1, 2], dtype=np.int32),
        "majors": np.asarray([0, 0, 2], dtype=np.int32),
        "minors": np.asarray([3, 4, 5], dtype=np.int32),
        "edge_start_time": arrays.edge_time[:3],
    }
    cpu = torch.device("cpu")
    for batch in ([0, 0, 1], [0, 0, 2]):
        ranked = dict(result, batch_id=np.asarray(batch, dtype=np.int32))
        assert sampling.sampled_rows(ranked, arrays, cpu).tolist() == [0, 1, 2]
    for batch in ([0, 1, 1], [1, 1, 2], [0, 0, 0]):
        with pytest.raises(RuntimeError, match=r"\(batch_id\)"):
            sampling.sampled_rows(
                dict(result, batch_id=np.asarray(batch, dtype=np.int32)), arrays, cpu
            )


@pytest.mark.parametrize("unified", [False, True], ids=["26.08", "26.10"])
@pytest.mark.parametrize("seed_labels", [False, True], ids=["ranked", "seed"])
def test_cugraph_subset_handles_seeds_without_edges(unified: bool, seed_labels: bool) -> None:
    # Hop 2 of a batch holds stubbed hubs and association-only contexts between others.
    keys, rows = _table({"zelle_out": 5, "payment_in": 1, "Account_Uses_Device": 2}, contexts=5)
    rows[0] = rows[3] = {"messages": []}
    rows[1] = {
        "messages": [m for m in rows[1]["messages"] if m["relation"] == "Account_Uses_Device"]
    }
    table = CandidateTable.build(keys, rows)
    plc = MockPLC("26.10.00" if unified else "26.08.00", unified=unified, seed_labels=seed_labels)
    engine = CuGraphSampler(plc, to_device=host)
    for hop in (1, 2):
        quotas = sampling.relation_quotas(RESAMPLE, hop)
        keep = engine.subset(table, quotas, random_state=5, device="cpu")
        expected = sampling.expected_counts(table, quotas)
        assert np.array_equal(sampling.group_counts(table, keep), expected)
