"""Builders of test data: configurations, accounts, labels, contexts and messages.

Nothing here opens a connection. There is one configuration builder (unit_config), whose
datasets are prepared from one source id (UNIT_SOURCE), one payment message (message),
one association (association) and one context row (context); the other builders are
those with particular values (a deterministic neighbourhood, random synthetic pools).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
import zlib

import numpy as np
import pandas as pd
import torch

from mule_pattern_learner.artifacts import (
    AUDIT_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    append_history,
    file_digest,
    write_audit_scores,
    write_diagnostic_table,
    write_epochs,
    write_feature_table,
    write_json,
    write_predictions,
    write_run_config,
)
from mule_pattern_learner.batching import assemble
from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.contract.feature_groups import (
    CORE_GROUPS,
    FeaturePlan,
    contract_fingerprint,
)
from mule_pattern_learner.contract.fingerprints import hash64, stable_score
from mule_pattern_learner.contract.graph_schema import (
    ASSOCIATION_RELATIONS,
    ASSOCIATION_TARGETS,
    ASSOCIATIONS,
    HUB_COLUMNS,
    PAYMENT_RELATIONS,
    RAILS,
    SPLIT_PHASE,
    ContextKey,
)
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.contract.server import ANALYTICS_CONTRACT, CONTEXT_CONTRACT
from mule_pattern_learner.contract.time_basis import BASIS_ID, fourier64
from mule_pattern_learner.data import manifest as data_manifest
from mule_pattern_learner.data.hub_registry import HubRegistry
from mule_pattern_learner.data.manifest import dataset_settings
from mule_pattern_learner.data.observed_labels import align_observed_labels
from mule_pattern_learner.diagnostics.baselines import baselines
from mule_pattern_learner.diagnostics.drift import drift
from mule_pattern_learner.diagnostics.learning_curve import learning_curve
from mule_pattern_learner.diagnostics.nnpu_simulation import Problem, nnpu_simulation
from mule_pattern_learner.diagnostics.proxy_validity import validity_table
from mule_pattern_learner.diagnostics.reveal_spread import reveal_spread
from mule_pattern_learner.diagnostics.subgroups import subgroups
from mule_pattern_learner.diagnostics.univariate import univariate
from mule_pattern_learner.evaluation import audit as evaluation_audit
from mule_pattern_learner.experiments.tables import COMPLETE, SuiteRun
from mule_pattern_learner.experiments.variants import Variant
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.metrics import (
    bootstrap_intervals,
    proxy_metrics,
    ranking_metrics,
    select_threshold,
)
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.paths import DatasetPaths, DiagnosticsPaths, RunPaths, SuitePaths
from mule_pattern_learner.sampling import candidates
from mule_pattern_learner.tigergraph.context_query import query_context_rows
from mule_pattern_learner.tigergraph.executor import QueryExecutor
from mule_pattern_learner.tigergraph.reveal import reveal_parameters
from mule_pattern_learner.training import trainer

if TYPE_CHECKING:
    import pytest

# The source id the test datasets are prepared from.
UNIT_SOURCE = "unit_fixture"
# The built-in run, small, in a unit scope, on the CPU.
UNIT_CHANGES: dict[str, Any] = {
    "scope": {"id": "unit_scope"},
    "model": {"hidden": 16, "dropout": 0.0},
    "training": {"epochs": 2, "batch_size": 16, "steps_per_epoch": 2},
    "runtime": {"device": "cpu", "threads": 1},
}


def unit_config(*changes: Mapping[str, Any], **sections: Any) -> RunConfig:
    """The tests' run: the built-in settings with UNIT_CHANGES, then each table of changes.

    A table maps sections to the fields it changes, as in RunConfig.with_changes; each
    keyword names a section and its table (applied last). RUNTIME_CHANGES is the runtime
    tests' table. Tests prepare the run's datasets from UNIT_SOURCE.
    """
    config = DEFAULT_CONFIG.with_changes(UNIT_CHANGES)
    for table in (*changes, sections):
        config = config.with_changes(dict(table))
    return config


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


def ground_truth_rows(population: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The ground-truth query's rows of scope population rows: every label known.

    In account order, the account at place i is a mule when the graph revealed it
    (observed_positive) or when i leaves 3 divided by 7, a hidden mule. The mules at
    places 14r to 14r + 13 form ring r, except the mules of every third ring (r divisible
    by 3), which have no ring (-1). The label source is what the reveal records.
    """
    rows = []
    ordered = sorted(population, key=lambda row: str(row["account_id"]))
    for i, row in enumerate(ordered):
        revealed = bool(row.get("observed_positive"))
        mule = revealed or i % 7 == 3
        ring = i // 14 if mule and (i // 14) % 3 else -1
        source = f"phantomledger_role;unit;salt=42;{'revealed:digital' if revealed else 'hidden'}"
        rows.append(
            {
                "account_id": str(row["account_id"]),
                "is_mule": int(mule),
                "mule_label_known": True,
                "is_mule_masked": not revealed,
                "pu_label": int(revealed),
                "mule_ring_id": ring,
                "mule_label_source": source if mule else "phantomledger_role",
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
    """assigned_accounts as scope population rows, the split as the partition.

    The accounts supplied_labels lists are revealed positives, with its discovery times,
    as the scope population query reports them with include_observed.
    """
    accounts = assigned_accounts()
    known = supplied_labels().set_index("account_id").known_from_ms
    revealed = accounts.account_id.map(known)
    rows = accounts.drop(columns=["owner_ids", "split"]).assign(
        observed_positive=revealed.notna(), known_from_ms=revealed.fillna(0).astype("int64")
    )
    records = rows.assign(partition=accounts.split.map(SPLIT_PHASE)).to_dict("records")
    return cast(list[dict[str, Any]], records)


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
    """One visible payment message carrying every field the context query prints."""
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
    **changes: Any,
) -> dict[str, Any]:
    """A valid-time association, emitted at the context cutoff; `changes` replace fields."""
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
        **changes,
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


# The node features of a context row unless a builder names others.
CONTEXT_FEATURES = {"is_deposit": 1}


def context(
    key: ContextKey,
    messages: list[dict[str, Any]] | None = None,
    *,
    encodings: bool = True,
    features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """An ok context row; `encodings` also prints the Fourier vectors (a spot check)."""
    row: dict[str, Any] = {
        **asdict(key),
        "status": "ok",
        "contract_version": CONTEXT_CONTRACT,
        "basis_id": BASIS_ID,
        "features": dict(CONTEXT_FEATURES if features is None else features),
        "messages": list(messages or []),
        "age_encoding": {},
        "gap_encoding": {},
    }
    return encode(row) if encodings else row


def neighbourhood(key: ContextKey) -> dict[str, Any]:
    """Deterministic history: payments on all four relations and an owner association."""
    if key.node_type != "Account":
        return context(key, encodings=False)
    h = zlib.crc32(f"{key.node_id}:{key.cutoff_seq}".encode())
    messages = []
    for j in range(8):
        seq, ts = key.cutoff_seq - 1 - 3 * j, key.cutoff_ms - (j + 1) * 3_600_000
        if seq <= 0 or ts <= 1:
            break
        relation = PAYMENT_RELATIONS[(h + j) % len(PAYMENT_RELATIONS)]
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


def event(
    seq: int, parent: ContextKey, *, relation: str = "payment_out", gap: int = 5
) -> dict[str, Any]:
    """A card payment of 12.5 at sequence ``seq``, 10 ms before the cutoff per sequence."""
    ts = parent.cutoff_ms - 10 * (parent.cutoff_seq - seq)
    changes = {"node_id": f"peer{seq}", "relation": relation, "rail": "card", "amount": 12.5}
    return message(seq, ts, parent, event_id=f"E{seq}", gap_ms=gap, gap_present=gap > 0, **changes)


def payments_context(key: ContextKey) -> dict[str, Any]:
    """An ok context of two payments before the cutoff, the second without a gap."""
    messages = [event(key.cutoff_seq - 1, key), event(key.cutoff_seq - 3, key, gap=0)]
    return context(key, messages, encodings=False)


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
RESAMPLE = SamplerPlan(
    roots=PoolPlan(recent=4, older=3, distinct=2, associations=2),
    relation_fanouts=(3, 2),
)
# The roots pool of the built-in run's shape, drawn into the default relation fan-outs.
POOLED = SamplerPlan(roots=PoolPlan(recent=4, older=3, distinct=2, associations=2))
# Pools with every kind of candidate at both hops, and relation fan-outs below them.
POOLED_RESAMPLE = SamplerPlan(
    roots=PoolPlan(recent=8, older=4, distinct=4, associations=2),
    children=PoolPlan(recent=4, older=2, distinct=2, associations=0),
    relation_fanouts=(8, 4),
)


def context_rng(key: ContextKey, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([candidates.context_hash(key) & 0xFFFFFFFF, salt])


def payment(
    key: ContextKey, relation: str, seq: int, peer: str, stratum: str, rng: np.random.Generator
) -> dict[str, Any]:
    ts = seq * MS_PER_SEQ
    gap_present = bool(rng.integers(0, 2))
    flow = bool(rng.integers(0, 2))
    # The fields are drawn in this order.
    drawn = {
        "node_id": peer,
        "relation": relation,
        "rail": "zelle" if relation.startswith("zelle") else str(rng.choice(RAILS[2:])),
        "event_id": f"E{relation[0]}{seq}",
        "amount": float(rng.integers(0, 5000)) / 7,
        "amount_present": bool(rng.integers(0, 4)),
        "gap_ms": int(rng.integers(0, ts)) if gap_present else 0,
        "gap_present": gap_present,
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
    }
    return message(seq, ts, key, **drawn)


def synthetic_association(
    key: ContextKey, relation: str, peer: str, rng: np.random.Generator
) -> dict[str, Any]:
    typ = ASSOCIATION_TARGET[relation]
    return association(
        key,
        relation=relation,
        node_type=typ,
        node_id=peer,
        amount=0.0,
        peer_first_ms=int(rng.integers(1, key.cutoff_ms + 1)),
        peer_external=typ == "Account" and bool(rng.integers(0, 2)),
        peer_deposit=typ == "Account" and bool(rng.integers(0, 2)),
    )


def synthetic_row(
    key: ContextKey, pool: PoolPlan, *, encodings: bool = True, full: bool = False
) -> dict[str, Any]:
    """Deterministic context row in the TigerGraph response format for any key.

    `full` fills every relation to the pool bound (the saturated profile).
    """
    rng = context_rng(key)
    messages: list[dict[str, Any]] = []
    if key.node_type == "Account":
        for relation in PAYMENT_RELATIONS:
            count = int(rng.integers(0, pool.recent + pool.older + pool.distinct + 1))
            count = pool.recent + pool.older + pool.distinct if full else count
            count = min(count, key.cutoff_seq - 1)
            seqs = sorted(rng.choice(np.arange(1, key.cutoff_seq), count, replace=False))[::-1]
            strata = ["recent"] * pool.recent + ["older"] * pool.older + ["distinct"] * 16
            for seq, stratum in zip(seqs, strata):
                peer = f"a{int(rng.integers(0, 12))}"
                messages.append(payment(key, relation, int(seq), peer, stratum, rng))
    for relation in ASSOCIATION_RELATIONS:
        if relation.split("_")[0] != key.node_type:
            continue
        for n in range(pool.associations if full else int(rng.integers(0, pool.associations + 1))):
            messages.append(synthetic_association(key, relation, f"{relation[-5:]}{n}", rng))
    rng.shuffle(messages)
    features = {
        "is_external": float(rng.integers(0, 2)),
        "is_deposit": float(rng.integers(0, 2)),
    }
    return context(key, messages, encodings=encodings, features=features)


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
                if relation in PAYMENT_RELATIONS:
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
    relation = PAYMENT_RELATIONS[h % len(PAYMENT_RELATIONS)]
    seq = parent.cutoff_seq - 1 - 5 * j - h % 5
    ts = parent.cutoff_ms - (j + 1) * 3_600_000 - h % 997
    gap = j % 3 != 0
    return message(
        seq,
        ts,
        parent,
        node_id=f"P{h % 11}",
        relation=relation,
        rail="zelle" if relation.startswith("zelle") else "ach",
        stratum=("recent", "older", "distinct")[j % 3],
        event_id=f"E{parent.node_id}.{seq}",
        amount=float(h % 500),
        gap_ms=(h % 50) * 60_000 if gap else 0,
        gap_present=gap,
        pair_prior_count=h % 4,
        pair_first_age_seconds=float(h % 10_000),
        flow_observation_seconds=600.0,
        peer_first_ms=1_000,
        peer_external=h % 5 == 0,
        peer_deposit=h % 5 != 0,
    )


def fake_context(key: ContextKey) -> dict[str, Any]:
    """The context FakeSource serves: a few hashed payments and at most one owner."""
    messages: list[dict[str, Any]] = []
    if key.node_type == "Account":
        count = 3 + hash64(key.node_id, key.cutoff_seq) % 6
        messages = [_message(key, j) for j in range(count) if key.cutoff_seq - 5 * j > 10]
        messages += [
            association(key, node_id=f"Q{hash64(key.node_id, j) % 5}", peer_first_ms=1_000)
            for j in range(hash64(key.node_id) % 2)
        ]
    features = {"is_external": 0.0, "is_deposit": 1.0}
    return context(key, messages, encodings=False, features=features)


# The runtime tests' changes to unit_config: a small model without the slot sum, validated
# on its raw weights, over small pools, with seeds and dates of its own.
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
    source_id: str = UNIT_SOURCE,
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
            "source_counts": {"Account": len(accounts)},
        },
    }
    dataset.manifest.write_text(json.dumps(manifest))

    def load(loaded: DatasetPaths) -> tuple[dict[str, Any], pd.DataFrame]:
        assert loaded == dataset
        return deepcopy(manifest), accounts.copy()

    for module in (trainer, data_manifest, evaluation_audit):
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


# The files of a complete, audited run for the reporting tests, shaped like the reference
# runs (docs/research/reference-run.md): each audited split weighs its 2,000 sampled
# non-mules and every mule to about 47,000 accounts, the test split's 47,749 with 40
# mules, and the scores sit near 0 and 1.
REPORTED_SPLITS = {
    # split: (cutoff, population accounts, mules, revealed mules, observed positives)
    "validation": ("2024-10-01", 47_120, 38, 11, 11),
    "test": ("2025-01-01", 47_749, 40, 11, 12),
}
REPORTED_EPOCHS = 11
REPORTED_SELECTED = 5
REPORTED_STEPS = 100
REPORTED_LOG_EVERY = 10


# The share of the mules a run of the reporting tests scores near 1.
FOUND = 0.62


def near_extremes(
    rng: np.random.Generator, positive: np.ndarray, found: float = FOUND
) -> np.ndarray:
    """Scores near 0 and 1: most positives score near 1, and a few other accounts do too.

    ``found`` is the share of positives that do; the run is better the higher it is.
    """
    high = np.where(positive, rng.random(len(positive)) < found, rng.random(len(positive)) < 0.009)
    logit = np.where(
        high,
        np.where(
            positive, rng.normal(12.0, 3.0, len(positive)), rng.normal(9.0, 3.0, len(positive))
        ),
        np.where(
            positive, rng.normal(-4.0, 3.5, len(positive)), rng.normal(-8.5, 2.5, len(positive))
        ),
    )
    return 1 / (1 + np.exp(-logit))


def audit_frame(rng: np.random.Generator, split: str, found: float = FOUND) -> pd.DataFrame:
    """A split's scored audit sample (artifacts.AUDIT_COLUMNS): every mule, 2,000 others.

    The accounts, their truth, rings and revealed flags are the split's, the same in every
    run, as the audit samples of a dataset are; the scores are the run's (near_extremes
    with ``found``), and every run ranks the louder mules higher.
    """
    _, accounts, mules, revealed, _ = REPORTED_SPLITS[split]
    negatives = 2000
    sample = np.random.default_rng(zlib.crc32(split.encode()))
    is_mule = np.r_[np.ones(mules, np.int64), np.zeros(negatives, np.int64)]
    loudness = sample.normal(0.0, 1.0, mules)
    score = near_extremes(rng, is_mule == 1, found)
    # Every run ranks the louder mules higher, give or take noise of its own.
    rank = np.argsort(np.argsort(-(loudness + rng.normal(0.0, 0.7, mules))))
    score[:mules] = np.sort(score[:mules])[::-1][rank]
    # The revealed mules are among the louder ones.
    flags = np.zeros(len(is_mule), bool)
    flags[np.argsort(-loudness)[:revealed]] = True
    # About two thirds of the mules are in rings of two to four.
    rings = np.full(len(is_mule), -1, np.int64)
    ringed = int(mules * 2 / 3)
    rings[:ringed] = np.repeat(np.arange(ringed), sample.integers(2, 5, ringed))[:ringed]
    return pd.DataFrame(
        {
            "account_id": [f"{split[0].upper()}{i:06d}" for i in range(len(is_mule))],
            "is_mule": is_mule,
            "inclusion_probability": np.where(is_mule == 1, 1.0, negatives / (accounts - mules)),
            "score": score,
            "revealed": flags,
            "ring_id": rings,
            "label_source": "phantomledger_role",
        }
    )


def predictions_frame(rng: np.random.Generator, split: str, found: float = FOUND) -> pd.DataFrame:
    """A split's proxy predictions: its observed positives and 2,000 unlabelled accounts."""
    date, accounts, mules, _, positives = REPORTED_SPLITS[split]
    unlabelled = 2000
    observed = np.r_[np.ones(positives, np.int64), np.zeros(unlabelled, np.int64)]
    # The unlabelled sample holds the split's hidden mules at their population rate.
    hidden = rng.random(len(observed)) < (mules - positives) / accounts
    score = near_extremes(rng, (observed == 1) | hidden, found)
    ids = [f"{split[0].upper()}P{i:06d}" for i in range(len(observed))]
    return pd.DataFrame(
        {
            "account_id": ids,
            "group_id": [f"G{i:06d}" for i in range(len(observed))],
            "date": date,
            "observed_label": observed,
            "score": score,
        }
    )


def history_rows(rng: np.random.Generator) -> list[dict[str, Any]]:
    """history.csv of the run: 10 log intervals of 10 steps in each of its 11 epochs.

    The dataset's disk cache already held about a third of the contexts memory did not.
    """
    rows: list[dict[str, Any]] = []
    totals = Counter[str]()
    for epoch in range(1, REPORTED_EPOCHS + 1):
        for step in range(REPORTED_LOG_EVERY, REPORTED_STEPS + 1, REPORTED_LOG_EVERY):
            position = epoch - 1 + step / REPORTED_STEPS
            loss = 0.22 + 0.55 * np.exp(-position / 1.6) + rng.normal(0, 0.025)
            corrected = int(rng.binomial(REPORTED_LOG_EVERY, min(0.02 + 0.03 * position, 0.4)))
            requested = int(rng.normal(1088, 40)) * REPORTED_LOG_EVERY
            hits = int(requested * min(0.12 + 0.03 * position, 0.45))
            disk = int((requested - hits) * rng.uniform(0.3, 0.4))
            totals.update(
                requested=requested,
                memory_hits=hits,
                disk_hits=disk,
                database_calls=(requested - hits - disk) // 8,
                stub_children=int(rng.poisson(6)),
            )
            seen = 900_000 * (1 - np.exp(-totals["requested"] / 900_000))
            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "date": "2024-07-01",
                    "loss": float(loss),
                    "objective": float(loss - 0.03 * corrected * rng.random()),
                    "corrected_steps": corrected,
                    "steps": REPORTED_STEPS,
                    "seconds_per_step": float(3.0 + rng.normal(0, 0.2) - 0.1 * (step == 10)),
                    "batch_wait_seconds": float(max(0.35 + rng.normal(0, 0.12), 0.0)),
                    "database_calls": totals["database_calls"],
                    "contexts_requested": totals["requested"],
                    "contexts_distinct": int(seen),
                    "memory_hits": totals["memory_hits"],
                    "disk_hits": totals["disk_hits"],
                    "rejected_roots": 0,
                    "stub_children": totals["stub_children"],
                }
            )
    return rows


def write_run_files(
    run: RunPaths,
    *,
    seed: int = 0,
    audited: bool = True,
    config: RunConfig = DEFAULT_CONFIG,
    found: float = FOUND,
) -> RunPaths:
    """The files of a complete run that reporting reads, audited unless ``audited`` is False.

    config.json (of ``config``), history.csv, epochs.csv, metrics.json and the proxy
    predictions, and the audit report and scored sample of validation and test, drawn
    with ``seed``; ``found`` is the share of mules the model scores near 1. The audits of
    every run score the same accounts. There is no model.pt: reporting never loads one.
    The audits' intervals come from 100 bootstrap replicates rather than the audit's
    1,000, to keep the tests fast.
    """
    rng = np.random.default_rng(seed)
    run.root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_commit": "0" * 40,
        "git_dirty": False,
        "device": "cuda",
        "sampler_backend": "cugraph",
        "dataset_id": "d" * 64,
        "started": "2026-09-28T09:00:00+00:00",
    }
    write_run_config(run.config, config, provenance)
    history = history_rows(rng)
    for row in history:
        append_history(run.history, row)
    predictions = {split: predictions_frame(rng, split, found) for split in REPORTED_SPLITS}
    validation = predictions["validation"]
    threshold = select_threshold(validation.observed_label.to_numpy(), validation.score.to_numpy())
    proxy = {
        split: proxy_metrics(frame.observed_label.to_numpy(), frame.score.to_numpy(), threshold)
        for split, frame in predictions.items()
    }
    for split, frame in predictions.items():
        write_predictions(run.predictions(split), frame)
    selected_ap = float(proxy["validation"]["average_precision"])
    epochs = []
    for epoch in range(1, REPORTED_EPOCHS + 1):
        ap = selected_ap if epoch == REPORTED_SELECTED else selected_ap * rng.uniform(0.3, 0.95)
        epochs.append(
            {
                "epoch": epoch,
                "loss": 0.22 + 0.55 * np.exp(-(epoch - 0.5) / 1.6),
                "steps": REPORTED_STEPS,
                "validation_ap": ap if epoch > 1 else selected_ap * 0.2,
                "validation_roc_auc": min(0.8 + 0.03 * epoch, 0.96) - rng.uniform(0, 0.02),
                "weights": "averaged",
                "selected": epoch == REPORTED_SELECTED,
                "stopped": epoch == REPORTED_EPOCHS,
            }
        )
    write_epochs(run.epochs, epochs)
    write_json(
        run.metrics,
        {
            "status": "complete",
            "dataset_id": provenance["dataset_id"],
            "seed": 42,
            "known_mules": {"train": 20, "validation": 11, "test": 12},
            "device": "cuda",
            "loss": "nnPU",
            "class_prior": 0.001,
            "positive_weight": 0.999,
            "objective": "imbalanced_nnPU",
            "parameter_count": 101_121,
            "best_epoch": REPORTED_SELECTED,
            "observed_label_proxy": proxy,
            "validation_proxy": proxy["validation"],
            "database_calls_during_training": history[-1]["database_calls"],
            "elapsed_seconds": 3.4 * 3600,
            "contexts": {
                "requested": history[-1]["contexts_requested"],
                "distinct": history[-1]["contexts_distinct"],
                "memory_hits": history[-1]["memory_hits"],
                "disk_hits": history[-1]["disk_hits"],
                "disk_hit_rate": history[-1]["disk_hits"]
                / (history[-1]["contexts_requested"] - history[-1]["memory_hits"]),
            },
            "rejections": {"history_capacity_exceeded": 412, "hub_stub": 38},
            "sampler_backend": "cugraph",
            "sampler_totals": {
                "rejected_roots": 0,
                "roots": 70_400,
                "contexts": history[-1]["contexts_requested"],
                "stub_children": history[-1]["stub_children"],
                "rejected_children": 412,
                "first_edges": 1_126_400,
                "second_edges": 4_505_600,
            },
            "rejected_roots": {
                split: {"requested": requested, "rejected": 0, "positive": 0, "unlabeled": 0}
                for split, requested in (("train", 70_400), ("validation", 2_011), ("test", 2_012))
            },
            "max_rejected_root_fraction": 0.0,
        },
    )
    if not audited:
        return run
    run.audit_report("test").parent.mkdir(parents=True, exist_ok=True)
    for split, (date, accounts, _, _, _) in REPORTED_SPLITS.items():
        frame = audit_frame(rng, split, found)
        write_audit_scores(run.audit_scores(split), frame)
        y, weight = frame.is_mule.to_numpy(), 1 / frame.inclusion_probability.to_numpy()
        intervals = bootstrap_intervals(
            y, frame.score.to_numpy(), weight, frame.ring_id.to_numpy(), replicates=100
        )
        mules = frame[frame.is_mule == 1]
        report = {
            "split": split,
            "purpose": evaluation_audit.AUDIT_SPLITS[split],
            "date": date,
            "population_accounts": accounts,
            "metrics": evaluation_audit.audit_metrics(frame, threshold),
            "intervals": intervals,
            "constants": evaluation_audit.audit_constants(42),
            "revealed_positives": int(mules.revealed.sum()),
            "hidden_positives": int((~mules.revealed).sum()),
            "rejected_accounts": 0,
        }
        write_json(run.audit_report(split), report)
    return run


def write_suite_runs(
    suite: SuitePaths,
    variants: Sequence[Variant],
    seeds: Sequence[int],
    found: Mapping[str, float] | None = None,
) -> list[SuiteRun]:
    """A suite's complete, audited runs (write_run_files), each variant's seeds in turn.

    ``found`` gives a variant's share of mules scored near 1 (default FOUND), so the
    variants rank differently; every audit scores the same accounts, so they pair.
    """
    runs = []
    for index, variant in enumerate(variants):
        for seed in seeds:
            paths = write_run_files(
                suite.run(variant.name, seed),
                seed=1000 * index + seed,
                config=variant.config(DEFAULT_CONFIG, seed),
                found=(found or {}).get(variant.name, FOUND),
            )
            runs.append(SuiteRun(variant, seed, paths, COMPLETE))
    return runs


# A synthetic diagnostic feature table for the analyses' tests: per split, its mules and
# sampled non-mules (each non-mule standing for FEATURE_WEIGHT accounts), and the months
# of history its cutoff has seen.
FEATURE_SAMPLE = {"train": (40, 400, 6), "validation": (12, 300, 9), "test": (15, 300, 12)}
FEATURE_WEIGHT = 25.0


def feature_frame(seed: int = 0) -> pd.DataFrame:
    """A diagnostic feature table (artifacts.FEATURE_TABLE_COLUMNS, then features).

    Mules have more distinct payers, first-time internal inflows and 30-day inflows than
    other accounts; the visible event count and the pair history grow with the months a
    split's cutoff has seen (drift); the account's age is one value per split; the entity
    flags are the same for every account; the pair window counts are noise. A third of
    each split's mules, the loudest, are revealed, most mules are in rings of three, and
    the third train non-mule is rejected, without features.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for split, (mules, negatives, months) in FEATURE_SAMPLE.items():
        mule = np.r_[np.ones(mules), np.zeros(negatives)].astype(np.int64)
        loud = rng.normal(0.0, 1.0, len(mule)) + mule
        count = len(mule)
        features = {
            "model__type_Account": np.ones(count),
            "model__is_deposit": np.ones(count),
            "model__history_withheld": np.zeros(count),
            "model__pool_in_unique": np.log1p(rng.poisson(3 + 2.5 * mule + 0.5 * loud.clip(0))),
            "model__pool_first_in_internal": np.log1p(rng.poisson(0.3 + 1.5 * mule)),
            "messages__mean_amount": rng.normal(4.0 + 0.3 * mule, 1.0),
            "messages__max_pair_prior_count": np.log1p(rng.poisson(0.4 * months, count)),
            "messages__stratum_distinct": rng.poisson(4 + 2 * mule).astype(np.float64),
            "account__visible_event_count": rng.poisson(25 * months + 40 * mule).astype(float),
            "account__30d_in_count": rng.poisson(5 + 3 * mule + loud.clip(0)).astype(float),
            "account__age_days": np.full(count, 30.5 * months),
            "message_context__mean_pair_count_7d": rng.gamma(2.0, 1.0, count),
        }
        revealed = np.zeros(count, bool)
        revealed[np.argsort(-np.where(mule == 1, loud, -np.inf))[: mules // 3]] = True
        rings = np.where((mule == 1) & (np.arange(count) < mules - 2), np.arange(count) // 3, -1)
        frame = pd.DataFrame(
            {
                "account_id": [f"{split[0].upper()}{i:05d}" for i in range(count)],
                "split": split,
                "date": DATES[split][0],
                "is_mule": mule,
                "revealed": revealed,
                "ring_id": rings.astype(np.int64),
                "label_source": np.where(
                    mule == 1, "phantomledger_role;unit", "phantomledger_role"
                ),
                "inclusion_probability": np.where(mule == 1, 1.0, 1 / FEATURE_WEIGHT),
                "weight": np.where(mule == 1, 1.0, FEATURE_WEIGHT),
                "rejected": False,
                "context_contract": CONTEXT_CONTRACT,
                "analytics_contract": ANALYTICS_CONTRACT,
                **features,
            }
        )
        if split == "train":
            frame.loc[mules + 2, "rejected"] = True
            frame.loc[mules + 2, list(features)] = np.nan
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    assert tuple(table.columns[: len(FEATURE_TABLE_COLUMNS)]) == FEATURE_TABLE_COLUMNS
    return table


# The mules of a synthetic reveal per split partition (train, validation, test), as many
# as the reference dataset has.
REVEAL_MULES = {1: 160, 2: 33, 3: 40}


def reveal_population(seed: int = 0) -> list[dict[str, Any]]:
    """What the reveal's inputs query prints for REVEAL_MULES mules, drawn with seed.

    Each mule is first seen at the start of 2024 and, four times in five, receives up to
    eleven fraud-labelled inflows an hour apart in a burst at a random day of the year,
    so whether a bank would have discovered it by its split's cutoff varies by salt.
    """
    rng = np.random.default_rng(seed)
    day, start = 86_400_000, timestamp("2024-01-01")
    mules = []
    for part, count in REVEAL_MULES.items():
        for _ in range(count):
            key = 1000 + len(mules)
            inflows = int(rng.integers(1, 12)) if rng.random() < 0.8 else 0
            burst = start + int(rng.integers(0, 365)) * day
            attributes = {
                "M.id": f"M{key}",
                "M.first_seen_ts_ms": start,
                "M.@part": part,
                "M.@key": key,
                "M.@inflows": [f"{key * 100 + i}:{burst + i * 3_600_000}" for i in range(inflows)],
            }
            mules.append({"v_id": f"M{key}", "attributes": attributes})
    return [{"M": mules}, {"zelle_links": []}, {"payment_links": []}]


def proxy_predictions(
    rng: np.random.Generator, split: str, found: float = FOUND
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A split's proxy predictions and the truth of their accounts.

    The predictions are predictions_frame's; a few of the unlabelled accounts are hidden
    mules, which the run scores lower than the revealed ones, as a proxy trained on the
    revealed mules does.
    """
    frame = predictions_frame(rng, split, found)
    observed = frame.observed_label.to_numpy() == 1
    hidden = ~observed & (rng.random(len(frame)) < 0.004)
    frame.loc[hidden, "score"] = near_extremes(rng, np.ones(int(hidden.sum()), bool), found / 3)
    truth = pd.DataFrame(
        {
            "account_id": frame.account_id,
            "is_mule": (observed | hidden).astype(np.int64),
            "ring_id": -1,
            "label_source": "phantomledger_role",
        }
    )
    return frame, truth


def study_audits(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """A synthetic run's scored audit samples on a feature table's held-out accounts.

    A study compares its baselines with the run on one population, so the run scores the
    feature table's accepted validation and test accounts (near_extremes with FOUND), its
    revealed mules above its hidden ones; the rest of each sample is the table's.
    """
    samples = {}
    for split in REPORTED_SPLITS:
        rows = frame[(frame.split == split) & ~frame.rejected.astype(bool)]
        mules = rows.is_mule.to_numpy() == 1
        score = near_extremes(rng, mules, FOUND)
        ranked = np.argsort(~rows.revealed.to_numpy()[mules], kind="stable")
        loudest = np.empty(int(mules.sum()))
        loudest[ranked] = np.sort(score[mules])[::-1]
        score[mules] = loudest
        sample = rows.assign(score=score)[list(AUDIT_COLUMNS)]
        samples[split] = sample.reset_index(drop=True)
    return samples


def diagnostic_tables(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Every analysis' table of a synthetic diagnostic study (artifacts.DIAGNOSTIC_TABLES).

    The feature table's analyses run on feature_frame, with the audit reports of a
    synthetic run on its accounts beside them (study_audits, with bootstrap intervals);
    the subgroups on that run's audit samples; the proxy validity on proxy_predictions;
    the reveal spread over 50 salts of reveal_population; the nnPU simulation on a small,
    short problem. The baselines' intervals take 40 replicates and the curve two draws,
    and the tables of a seed are computed once per process (each call gets copies), to
    keep the tests fast.
    """
    return {name: table.copy() for name, table in _diagnostic_tables(seed).items()}


@functools.cache
def _diagnostic_tables(seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frame = feature_frame(seed)
    samples = study_audits(frame, rng)
    audits = {}
    for split, sample in samples.items():
        y, score = sample.is_mule.to_numpy(), sample.score.to_numpy()
        weight = 1 / sample.inclusion_probability.to_numpy()
        audits[split] = {
            "metrics": ranking_metrics(y, score, weight),
            "intervals": bootstrap_intervals(
                y, score, weight, sample.ring_id.to_numpy(), replicates=40
            ),
        }
    predicted = {split: proxy_predictions(rng, split) for split in REPORTED_SPLITS}
    truth = pd.concat([found for _, found in predicted.values()], ignore_index=True)
    params = reveal_parameters(DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates, apply=False)
    small = Problem(marginal=2_000, test_positives=30, test_negatives=30_000, steps=20, epochs=3)
    return {
        "univariate": univariate(frame),
        "drift": drift(frame),
        "baselines": baselines(frame, run="baseline/seed-42", audits=audits, replicates=40),
        "learning_curve": learning_curve(frame, repeats=2, audits=audits),
        "subgroups": subgroups(samples),
        "proxy_validity": validity_table(
            {split: rows for split, (rows, _) in predicted.items()}, truth, 0.5
        ),
        "reveal_spread": reveal_spread(reveal_population(seed), params, salts=range(50)),
        "nnpu_simulation": nnpu_simulation(seeds=(1, 2), problem=small),
    }


def write_study_files(study: DiagnosticsPaths, seed: int = 0) -> DiagnosticsPaths:
    """The files of a synthetic diagnostic study that reporting reads.

    features.parquet (feature_frame), every analysis' table (diagnostic_tables) and a
    study.json that records each analysis as written, the built-in run as compared and
    the built-in reveal's salt and budget.
    """
    study.root.mkdir(parents=True, exist_ok=True)
    write_feature_table(study.features, feature_frame(seed))
    outcomes = {"features": {"status": "written", "rows": len(feature_frame(seed))}}
    for name, table in diagnostic_tables(seed).items():
        write_diagnostic_table(study.table(name), name, table)
        outcomes[name.replace("_", "-")] = {"status": "written", "rows": len(table)}
    scope = DEFAULT_CONFIG.scope
    record = {
        "dataset_id": study.root.name,
        "run": "baseline/seed-42",
        "run_compared": True,
        "reveal": {"salt": scope.reveal_salt, "budget": scope.reveal_per_split},
        "analyses": outcomes,
    }
    write_json(study.study, record)
    return study
