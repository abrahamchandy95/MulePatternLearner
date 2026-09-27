"""Builders of test data: configurations, accounts, labels, contexts and messages.

Nothing here opens a connection. The builders come from several test modules and
still differ in detail (several build messages); they are shared, not yet merged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, cast
import zlib

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.artifacts import file_digest
from mule_pattern_learner.batching import assemble
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.contract.feature_groups import (
    CORE_GROUPS,
    FEATURE_GROUPS,
    FeaturePlan,
    contract_fingerprint,
)
from mule_pattern_learner.contract.fingerprints import hash64, stable_score
from mule_pattern_learner.contract.graph_schema import (
    ASSOCIATION_TARGETS,
    ASSOCIATIONS,
    HUB_COLUMNS,
    RAILS,
    RELATIONS,
    SPLIT_PHASE,
    ContextKey,
)
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_CONTRACT
from mule_pattern_learner.contract.time_basis import BASIS_ID, fourier64
from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.data.hub_registry import HubRegistry
from mule_pattern_learner.data.manifest import dataset_settings
from mule_pattern_learner.data.observed_labels import align_observed_labels, validate_label_table
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.paths import DatasetPaths
from mule_pattern_learner.sampling import candidates
from mule_pattern_learner.tigergraph.context_query import query_context_rows
from mule_pattern_learner.tigergraph.executor import QueryExecutor
from mule_pattern_learner.training import trainer

# The source ids the test datasets are prepared from.
UNIT_SOURCE = "unit_fixture"
SNAPSHOT_SOURCE = "unit_snapshot"
RUNTIME_SOURCE = "unit_runtime"


def example_config(**sections: Any) -> RunConfig:
    """A small, explicit unit run of the built-in settings in an example scope.

    Each keyword names a section and a table of the fields it changes, as in
    RunConfig.with_changes. Tests hand prepare() their labels (FrameObservedLabels) and
    the source id UNIT_SOURCE.
    """
    small = {
        "scope": {"id": "example_strict_scope", "create": False},
        "model": {"hidden": 16, "dropout": 0.0},
        "training": {"epochs": 2, "batch_size": 16, "steps_per_epoch": 2},
        "runtime": {"device": "cpu", "threads": 1},
    }
    return DEFAULT_CONFIG.with_changes(small).with_changes(sections)


def fixture_accounts(count: int = 1000, date: str = "2024-07-01") -> pd.DataFrame:
    """Accounts with one owner each, all visible at date (see assigned_accounts)."""
    return pd.DataFrame(
        {
            "account_id": [f"A{i:04}" for i in range(count)],
            "first_seen_ts_ms": timestamp(date) - 1000,
            "owner_ids": [[f"P{i:04}"] for i in range(count)],
            "observed_positive": False,
            "known_from_ms": 0,
        }
    )


def scope_population(count: int = 200) -> list[dict[str, Any]]:
    """Scope members as the scope population query prints them with include_observed.

    Partitions repeat 1, 1, 1, 2, 3 (train, validation, test). Every seventh account is
    a revealed mule, discovered on 2024-03-01, so every split has revealed mules. Every
    account was opened on 2024-01-01, before each split's cutoff.
    """
    rows = []
    for i in range(count):
        positive = i % 7 == 0
        rows.append(
            {
                "account_id": f"S{i:04}",
                "first_seen_seq": 5,
                "first_seen_ts_ms": timestamp("2024-01-01"),
                "partition": (1, 1, 1, 2, 3)[i % 5],
                "group_id": f"G{i:04}",
                "observed_positive": positive,
                "known_from_ms": timestamp("2024-03-01") if positive else 0,
            }
        )
    return rows


def assigned_accounts() -> pd.DataFrame:
    """fixture_accounts with an ownership group and a split drawn from the group's hash.

    Each account has its own owner, so it is its own group; about 70% of the groups are
    train, 15% validation and 15% test.
    """
    accounts = fixture_accounts()
    accounts["group_id"] = "Account:" + accounts.account_id
    scores = accounts.group_id.map(lambda group: stable_score(str(group), 42, "split"))
    accounts["split"] = np.where(
        scores < 0.7, "train", np.where(scores < 0.85, "validation", "test")
    )
    return accounts


def scoped_accounts() -> list[dict[str, Any]]:
    """assigned_accounts as scope population rows, the split as the partition."""
    accounts = assigned_accounts()
    rows = accounts.drop(columns=["owner_ids", "split"])
    records = rows.assign(partition=accounts.split.map(SPLIT_PHASE)).to_dict("records")
    return cast(list[dict[str, Any]], records)


@dataclass
class FrameObservedLabels:
    """Observed labels from a table: a label source for prepare() other than the graph.

    Population queries run without include_observed for it: its labels are not the
    graph's.
    """

    labels: pd.DataFrame
    from_graph = False

    def positive_ids(self) -> set[str]:
        validate_label_table(self.labels)
        return set(self.labels.loc[self.labels.known_positive.astype(bool), "account_id"])

    def read(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return align_observed_labels(metadata, self.labels)


def supplied_labels(per_split: int = 20) -> pd.DataFrame:
    """Observed positives: the first `per_split` accounts of every split."""
    chosen = assigned_accounts().groupby("split").head(per_split)
    return pd.DataFrame(
        {
            "account_id": chosen.account_id,
            "known_positive": True,
            "known_from_ms": timestamp("2024-01-01"),
        }
    )


def message(seq: int, ts: int, parent: ContextKey, **changes: Any) -> dict[str, Any]:
    """One visible payment message carrying every v5 message field."""
    value: dict[str, Any] = {
        "node_type": "Account",
        "node_id": "neighbor",
        "relation": "zelle_out",
        "rail": "zelle",
        "channel": "digital",
        "stratum": "recent",
        "event_id": "E" + str(seq),
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": 20,
        "amount_present": True,
        "age_ms": parent.cutoff_ms - ts,
        "gap_ms": 0,
        "gap_present": True,
        "pair_count_1h": 1,
        "pair_count_1d": 1,
        "pair_count_7d": 1,
        "pair_prior_count": 1,
        "pair_first_age_seconds": 3600.0,
        "pair_first_present": True,
        "flow_delay_seconds": 0.0,
        "flow_present": False,
        "flow_censored": True,
        "flow_observation_seconds": 60.0,
        "flow_amount_ratio": 0.0,
        "flow_ratio_present": False,
        "flow_same_rail": False,
        "device_age_seconds": 0.0,
        "device_present": False,
        "ip_age_seconds": 0.0,
        "ip_present": False,
        "peer_first_ms": 1,
        "peer_external": False,
        "peer_deposit": True,
    }
    value.update(changes)
    return value


def association(
    parent: ContextKey,
    *,
    relation: str = "Account_Owned_By_Party",
    node_type: str = "Party",
    node_id: str = "owner",
) -> dict[str, Any]:
    """A valid-time association, emitted at the context cutoff."""
    zeros = {k: 0 for k, v in message(1, 1, parent).items() if isinstance(v, int | float)}
    flags = dict.fromkeys(
        (
            "amount_present",
            "gap_present",
            "pair_first_present",
            "flow_present",
            "flow_censored",
            "flow_ratio_present",
            "flow_same_rail",
            "device_present",
            "ip_present",
            "peer_external",
            "peer_deposit",
        ),
        False,
    )
    return {
        **zeros,
        **flags,
        "node_type": node_type,
        "node_id": node_id,
        "relation": relation,
        "rail": "unknown",
        "channel": "unknown",
        "stratum": "association",
        "event_id": "",
        "event_seq": parent.cutoff_seq,
        "event_ts_ms": parent.cutoff_ms,
        "peer_first_ms": 1,
    }


def encode(row: dict[str, Any]) -> dict[str, Any]:
    """Fill age_encoding/gap_encoding for every event message, in place."""
    row["age_encoding"], row["gap_encoding"] = {}, {}
    for item in row["messages"]:
        if item["event_id"]:
            name = item["relation"] + ":" + item["event_id"]
            row["age_encoding"][name] = fourier64(np.array([item["age_ms"]]))[0].tolist()
            if item["gap_present"]:
                row["gap_encoding"][name] = fourier64(np.array([item["gap_ms"]]))[0].tolist()
    return row


def context(
    key: ContextKey, messages: list[dict[str, Any]] | None = None, *, encodings: bool = True
) -> dict[str, Any]:
    """An ok context row; `encodings` also prints the Fourier vectors (a spot check)."""
    row: dict[str, Any] = {
        **asdict(key),
        "status": "ok",
        "contract_version": CONTEXT_CONTRACT,
        "basis_id": BASIS_ID,
        "features": {
            "is_deposit": 1,
            "age_days": 1,
            "1d_out_count": 2,
            "1d_out_in_amount_ratio": 0,
            "7d_out_in_amount_ratio": 0,
        },
        "messages": list(messages or []),
        "age_encoding": {},
        "gap_encoding": {},
    }
    return encode(row) if encodings else row


def neighbourhood(key: ContextKey) -> dict[str, Any]:
    """Deterministic v5 history: payments on all four relations and an owner association."""
    if key.node_type != "Account":
        return context(key, encodings=False)
    h = zlib.crc32(f"{key.node_id}:{key.cutoff_seq}".encode())
    messages = []
    for j in range(8):
        seq, ts = key.cutoff_seq - 1 - 3 * j, key.cutoff_ms - (j + 1) * 3_600_000
        if seq <= 0 or ts <= 1:
            break
        relation = RELATIONS[(h + j) % 4]
        messages.append(
            message(
                seq,
                ts,
                key,
                node_id=f"N{(h + j) % 7}",
                relation=relation,
                rail="zelle" if relation.startswith("zelle") else "ach",
                event_id=f"E{key.node_id}.{seq}",
                amount=float(10 + (h + j) % 90),
                gap_ms=60_000 * j,
                gap_present=j > 0,
                pair_prior_count=j,
                pair_first_age_seconds=float(3600 * j),
                pair_first_present=j > 0,
            )
        )
    messages.append(association(key, node_id=f"Q{h % 3}"))
    return context(key, messages, encodings=False)


def reveal_inputs() -> list[dict[str, Any]]:
    """What the reveal's INPUTS_QUERY prints for five mules and one Zelle link.

    A (train) and D (test) received five fraud-labelled inflows about eight years
    before their split's cutoff; B (train) exchanged money with A before A could be
    discovered; C (validation) has no evidence; E has no split.
    """
    day = 86_400_000
    cutoffs = {1: "2024-07-01", 2: "2024-10-01", 3: "2025-01-01"}

    def mule(name: str, part: int, key: int, inflows: int) -> dict[str, Any]:
        start = timestamp(cutoffs.get(part, "2024-07-01")) - 3000 * day
        return {
            "v_id": name,
            "attributes": {
                "M.id": name,
                "M.first_seen_ts_ms": start,
                "M.@part": part,
                "M.@key": key,
                "M.@inflows": [f"{key * 10 + i}:{start + 100 * day + i}" for i in range(inflows)],
            },
        }

    mules = [mule("A", 1, 101, 5), mule("B", 1, 102, 0), mule("C", 2, 103, 0)]
    mules += [mule("D", 3, 104, 5), mule("E", 0, 105, 5)]
    link = {
        "LZ.event_seq": 2001,
        "LZ.event_ts_ms": timestamp(cutoffs[1]) - 2950 * day,
        "LZ.@ends": ["B", "A"],
    }
    return [{"M": mules}, {"zelle_links": [{"attributes": link}]}, {"payment_links": []}]


def query_context_batch(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
) -> list[dict[str, Any] | None]:
    """Validated contexts in key order; None where TigerGraph rejected one request."""
    return [
        row if row.get("status") == "ok" else None
        for row in query_context_rows(
            executor, batch, plan=plan, sampler=sampler, hop=hop, emit_encodings=emit_encodings
        )
    ]


# Contexts and messages of the transport tests.
PLAN = FeaturePlan(("entity_meta", "message_core", "time_encoding"), "tgat")
SAMPLER = SamplerPlan(
    roots=PoolPlan(recent=4, older=2, distinct=1, associations=2, max_history=2048),
    children=PoolPlan(recent=2, associations=0, max_history=1024),
)


def event(
    seq: int, parent: ContextKey, *, relation: str = "payment_out", gap: int = 5
) -> dict[str, Any]:
    ts = parent.cutoff_ms - 10 * (parent.cutoff_seq - seq)
    return {
        "node_type": "Account",
        "node_id": f"peer{seq}",
        "relation": relation,
        "rail": "card",
        "channel": "digital",
        "stratum": "recent",
        "event_id": f"E{seq}",
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": 12.5,
        "amount_present": True,
        "age_ms": parent.cutoff_ms - ts,
        "gap_ms": gap,
        "gap_present": gap > 0,
        "peer_first_ms": 1,
        "peer_external": False,
        "peer_deposit": True,
    }


def context_row(
    key: ContextKey, messages: list[dict[str, Any]], *, encodings: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **asdict(key),
        "status": "ok",
        "contract_version": CONTEXT_CONTRACT,
        "basis_id": BASIS_ID,
        "features": {
            "type_Account": 1,
            "is_deposit": 1,
            "1d_out_in_amount_ratio": 0,
            "7d_out_in_amount_ratio": 0,
        },
        "messages": deepcopy(messages),
        "age_encoding": {},
        "gap_encoding": {},
    }
    return encode(row) if encodings else row


def root(i: int, **changes: Any) -> ContextKey:
    return replace(ContextKey("Account", f"A{i:04}", 1000, 100_000, "scope", 1), **changes)


def hub_row(account: str, cutoff: int, phase: int = 3, **changes: Any) -> dict[str, Any]:
    return {
        "account_id": account,
        "cutoff_seq": cutoff,
        "visibility_phase": phase,
        "max_visible": 5000,
        "max_degree": 9000,
        "reason": "visible_history",
        **changes,
    }


def hub_rows(cutoffs: list[int], scope_id: str = "") -> list[dict[str, Any]]:
    if scope_id:
        hubs = [hub_row("H1", cutoffs[0], 2), hub_row("H1", cutoffs[0], 3)]
        hubs.append(hub_row("H2", cutoffs[1], 1))
    else:
        hubs = [hub_row("H1", cutoffs[0]), hub_row("H2", cutoffs[1])]
    return [
        {
            "status": "ok",
            "cutoff_seqs": cutoffs,
            "threshold": 1024,
            "scope_id": scope_id,
            "candidates": 2,
            "hubs": hubs,
        }
    ]


def unit_config(**sections: Any) -> RunConfig:
    """The built-in run in a unit scope; its datasets come from SNAPSHOT_SOURCE."""
    return DEFAULT_CONFIG.with_changes({"scope": {"id": "unit_scope"}}).with_changes(sections)


# The plan and pools of the batch and context source tests.
CORE_PLAN = FeaturePlan(CORE_GROUPS, "tgat")
SMALL_SAMPLER = SamplerPlan(
    roots=PoolPlan(recent=4, older=1, distinct=1, associations=1),
    children=PoolPlan(recent=2, associations=0),
    relation_fanouts=(3, 2),
)


# Synthetic candidate pools of the sampler and batch tests.
MS_PER_SEQ = 3_600_000  # synthetic clocks: one event sequence number per hour
ASSOCIATION_TARGET = {
    rel: typ
    for pair, types in zip(ASSOCIATIONS, ASSOCIATION_TARGETS)
    for rel, typ in zip(pair, types)
}
PAYMENTS = RELATIONS[:4]
RESAMPLE = SamplerPlan(
    roots=PoolPlan(recent=4, older=3, distinct=2, associations=2),
    relation_fanouts=(3, 2),
)
# The roots pool of the built-in run's shape, drawn into the default relation fan-outs.
POOLED = SamplerPlan(roots=PoolPlan(recent=4, older=3, distinct=2, associations=2))


def context_rng(key: ContextKey, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([candidates.context_hash(key) & 0xFFFFFFFF, salt])


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


def synthetic_association(
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
    rng = context_rng(key)
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
            messages.append(synthetic_association(key, relation, f"{relation[-5:]}{n}", rng))
    rng.shuffle(messages)
    features: dict[str, float] = {
        "is_external": float(rng.integers(0, 2)),
        "is_deposit": float(rng.integers(0, 2)),
        "age_days": float(rng.random() * 400),
    }
    for name in FEATURE_GROUPS["rolling_windows"].names[:21]:
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


def roots(n: int = 8, cutoff: int = 400, scope: str = "s", phase: int = 1) -> list[ContextKey]:
    return [
        ContextKey("Account", f"r{i}", cutoff, cutoff * MS_PER_SEQ, scope, phase) for i in range(n)
    ]


def slots(
    keys: list[ContextKey],
    rows: list[dict[str, Any]],
    sampler: SamplerPlan,
    fanout: int,
    hop: int = 1,
    *,
    mode: str = "eval",
    step_seed: int = 0,
) -> list[list[dict[str, Any]]]:
    """The messages resampling puts in each context's slots, as batches draw them."""
    return assemble._select(  # pyright: ignore[reportPrivateUsage]
        keys,
        rows,
        hop=hop,
        fanout=fanout,
        sampler=sampler,
        mode=mode,
        step_seed=step_seed,
        backend="torch",
        device=torch.device("cpu"),
    )


def candidate_table(
    counts: dict[str, int], cutoff: int = 1000, contexts: int = 1
) -> tuple[list[ContextKey], list[dict[str, Any]]]:
    keys = [ContextKey("Account", f"c{c}", cutoff, cutoff * MS_PER_SEQ) for c in range(contexts)]
    rows: list[dict[str, Any]] = []
    for key in keys:
        rng = context_rng(key, 1)
        messages = []
        for relation, count in counts.items():
            for n in range(count):
                if relation in PAYMENTS:
                    m = payment(key, relation, cutoff - 1 - n, f"p{n}", "recent", rng)
                else:
                    m = synthetic_association(key, relation, f"n{n}", rng)
                messages.append(m)
        rows.append({"messages": messages})
    return keys, rows


# A prepared dataset, a context source and a saved model for the training tests.
DATES = {"train": ["2024-07-01"], "validation": ["2024-10-01"], "test": ["2025-01-01"]}
CUTOFFS = {"2024-07-01": 10_000, "2024-10-01": 20_000, "2025-01-01": 30_000}
HUB = "P3"


def _message(parent: ContextKey, j: int) -> dict[str, Any]:
    h = hash64(parent.node_type, parent.node_id, parent.cutoff_seq, j)
    relation = RELATIONS[h % 4]
    seq = parent.cutoff_seq - 1 - 5 * j - h % 5
    ts = parent.cutoff_ms - (j + 1) * 3_600_000 - h % 997
    gap = j % 3 != 0
    return {
        "node_type": "Account",
        "node_id": f"P{h % 11}",
        "relation": relation,
        "rail": "zelle" if relation.startswith("zelle") else "ach",
        "channel": "digital",
        "stratum": ("recent", "older", "distinct")[j % 3],
        "event_id": f"E{parent.node_id}.{seq}",
        "event_seq": seq,
        "event_ts_ms": ts,
        "amount": float(h % 500),
        "amount_present": True,
        "age_ms": parent.cutoff_ms - ts,
        "gap_ms": (h % 50) * 60_000 if gap else 0,
        "gap_present": gap,
        "pair_count_1h": 0,
        "pair_count_1d": 1,
        "pair_count_7d": 2,
        "pair_prior_count": h % 4,
        "pair_first_age_seconds": float(h % 10_000),
        "pair_first_present": True,
        "flow_delay_seconds": 0.0,
        "flow_present": False,
        "flow_censored": True,
        "flow_observation_seconds": 600.0,
        "flow_amount_ratio": 0.0,
        "flow_ratio_present": False,
        "flow_same_rail": False,
        "device_age_seconds": 0.0,
        "device_present": False,
        "ip_age_seconds": 0.0,
        "ip_present": False,
        "peer_first_ms": 1_000,
        "peer_external": h % 5 == 0,
        "peer_deposit": h % 5 != 0,
    }


def _association(parent: ContextKey, j: int) -> dict[str, Any]:
    template = _message(parent, 0)
    zero = {k: 0 for k, v in template.items() if isinstance(v, (int, float))}
    return {
        **zero,
        "node_type": "Party",
        "node_id": f"Q{hash64(parent.node_id, j) % 5}",
        "relation": "Account_Owned_By_Party",
        "rail": "unknown",
        "channel": "unknown",
        "stratum": "association",
        "event_id": "",
        "event_seq": parent.cutoff_seq,
        "event_ts_ms": parent.cutoff_ms,
        "amount_present": False,
        "gap_present": False,
        "pair_first_present": False,
        "flow_present": False,
        "flow_censored": False,
        "flow_ratio_present": False,
        "flow_same_rail": False,
        "device_present": False,
        "ip_present": False,
        "peer_external": False,
        "peer_deposit": False,
        "peer_first_ms": 1_000,
    }


def fake_context(key: ContextKey) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if key.node_type == "Account":
        count = 3 + hash64(key.node_id, key.cutoff_seq) % 6
        messages = [_message(key, j) for j in range(count) if key.cutoff_seq - 5 * j > 10]
        messages += [_association(key, j) for j in range(hash64(key.node_id) % 2)]
    return {
        **asdict(key),
        "status": "ok",
        "features": {"is_external": 0.0, "is_deposit": 1.0},
        "messages": messages,
    }


# A small model without the slot sum, validated on its raw weights, over small pools.
RUNTIME_CHANGES: dict[str, Any] = {
    "scope": {"id": "unit_scope"},
    "dataset": {"dates": deepcopy(DATES), "seed": 7, "split_seed": 7},
    "sampler": {
        "fanouts": [4, 2],
        "roots": {"recent": 3, "older": 1, "distinct": 1, "associations": 1, "max_history": 2048},
        "children": {
            "recent": 3,
            "older": 1,
            "distinct": 1,
            "associations": 0,
            "max_history": 2048,
        },
        "relation_fanouts": [2, 2],
    },
    "features": list(CORE_GROUPS),
    "model": {"architecture": "tgat", "hidden": 16, "heads": 2, "dropout": 0.2, "slot_sum": False},
    "loss": {"class_prior": 0.05, "positive_weight": 0.5},
    "training": {
        "seed": 7,
        "epochs": 2,
        "steps_per_epoch": 3,
        "batch_size": 8,
        "patience": 5,
        "learning_rate": 0.01,
        "weight_average_decay": 0.0,
        "proxy_unlabeled_limit": 12,
    },
    "runtime": {"device": "cpu", "threads": 1, "log_every_steps": 2, "prefetch_batches": 2},
}


def base_config(**sections: Any) -> RunConfig:
    """The runtime tests' run; its datasets come from RUNTIME_SOURCE.

    Each keyword names a section and a table of the fields it changes, as in
    RunConfig.with_changes.
    """
    return DEFAULT_CONFIG.with_changes(RUNTIME_CHANGES).with_changes(sections)


def accounts_frame() -> pd.DataFrame:
    rows = []
    for i in range(72):
        split = ("train", "validation", "test")[i % 3]
        rows.append(
            {
                "account_id": f"A{i:03}",
                "first_seen_seq": 5,
                "first_seen_ts_ms": timestamp("2024-01-01"),
                "group_id": f"G{i:03}",
                "observed_positive": False,
                "known_from_ms": 0,
                "split": split,
                # Label-selected pool rows sit outside the marginal reservoir.
                "in_marginal": i % 11 != 0,
            }
        )
    return pd.DataFrame(rows)


def labels_frame(accounts: pd.DataFrame) -> pd.DataFrame:
    chosen = accounts[accounts.index % 5 == 0]
    return pd.DataFrame(
        {
            "account_id": chosen.account_id,
            "known_positive": True,
            "known_from_ms": timestamp("2024-03-01"),
        }
    )


def hub_registry() -> HubRegistry:
    """The scoped registry: one hub at the training cutoff in phase 1 (training batches)."""
    row = {
        "account_id": HUB,
        "cutoff_seq": CUTOFFS["2024-07-01"],
        "visibility_phase": 1,
        "max_visible": 5000,
        "max_degree": 5000,
        "reason": "visible_history",
    }
    frame = pd.DataFrame([[row[name] for name in HUB_COLUMNS]], columns=list(HUB_COLUMNS))
    return HubRegistry(frame, cutoff_seqs=CUTOFFS.values(), threshold=2048, scope_id="unit_scope")


def prepared_dataset(
    path: Path,
    config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    source_id: str = RUNTIME_SOURCE,
) -> tuple[DatasetPaths, dict[str, Any], pd.DataFrame]:
    """A prepared dataset in directory path; load_prepared is replaced by its in-memory copy.

    Its manifest records config's dataset settings for the source source_id.
    """
    dataset = DatasetPaths(path)
    path.mkdir(parents=True, exist_ok=True)
    accounts = accounts_frame()
    align_observed_labels(accounts, labels_frame(accounts)).to_parquet(
        dataset.observed_labels, index=False
    )
    manifest = {
        "status": "ready",
        "cutoff_seqs": dict(CUTOFFS),
        "account_selection": "bounded_internal_deposit_seeds",
        "observed_labels_sha256": file_digest(dataset.observed_labels),
        "source": {
            "source_id": source_id,
            "settings": dataset_settings(source_id, config),
            "scope_id": config.scope.id,
        },
    }
    dataset.manifest.write_text(json.dumps(manifest))

    def load(loaded: DatasetPaths) -> tuple[dict[str, Any], pd.DataFrame]:
        assert loaded == dataset
        return deepcopy(manifest), accounts.copy()

    for module in (trainer, data_manifest):
        monkeypatch.setattr(module, "load_prepared", load)
    return dataset, manifest, accounts


def saved_model(
    path: Path,
    config: RunConfig,
    dataset: DatasetPaths | None = None,
    *,
    logit_shift: float = 0.0,
) -> Path:
    plan = config.feature_plan()
    torch.manual_seed(0)
    model = build_model(config.model, plan, config.sampler.fanouts[0], dropout=0.0)
    with torch.no_grad():
        bias = model.head[-1].bias
        assert isinstance(bias, torch.Tensor)
        bias += logit_shift
    payload = {
        "format": SavedModel.FORMAT,
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "contract": contract_fingerprint(),
        "basis_id": BASIS_ID,
        "threshold": 0.5,
        "input_fingerprint": plan.fingerprint(),
        "selected_on": "validation_observed_label_proxy_ap",
    }
    if dataset is not None:
        # A dataset's directory is named by its dataset id.
        payload["dataset_manifest_sha256"] = file_digest(dataset.manifest)
        payload["dataset_id"] = dataset.root.name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
