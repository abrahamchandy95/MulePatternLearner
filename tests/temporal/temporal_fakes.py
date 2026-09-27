"""Fake TigerGraph training endpoints and context fixtures for the v5 contract.

Nothing here opens a connection. `FakeExecutor` answers the read-only training
queries the way the repository GSQL does: context requests carry 1..64 keys and
exactly the query's parameters, messages are cut to the requested hop pool,
Fourier vectors are printed only when `emit_encodings` is set, every request index
gets exactly one row, and per-request failures are status rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
from typing import Any, cast
import zlib

import numpy as np
import pandas as pd

from mule_pattern_learner.temporal.common import stable_score, timestamp
from mule_pattern_learner.temporal.encoding import BASIS_ID, fourier64
from mule_pattern_learner.temporal.live.config_schema import run_config
from mule_pattern_learner.temporal.live.contract import (
    CONTRACT_VERSION,
    RELATIONS,
    SPLIT_PHASE,
    ContextKey,
)
from mule_pattern_learner.temporal.live.installation import definitions, parameter_names
from mule_pattern_learner.temporal.live.supervision import (
    align_observed_labels,
    validate_label_table,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CONTEXT_QUERY = "temporal_training_context"
HUB_QUERY = "temporal_hub_registry"
PAYMENT_RELATIONS = frozenset(RELATIONS[:4])
MAX_REQUEST_KEYS = 64


def signature(path: str, name: str) -> frozenset[str]:
    """Parameter names of one repository query."""
    return frozenset(parameter_names(definitions((REPOSITORY / path).read_text())[name]))


CONTEXT_PARAMETERS = signature("gsql/temporal/training_context.gsql", CONTEXT_QUERY)
HUB_PARAMETERS = signature("gsql/temporal/hub_registry.gsql", HUB_QUERY)
SCOPE_POLICY_QUERY = "temporal_scope_policy"
SCOPE_POLICY_PARAMETERS = signature("gsql/temporal/training_scope.gsql", SCOPE_POLICY_QUERY)
SCOPE_POPULATION_QUERY = "temporal_scope_population"
SCOPE_POPULATION_PARAMETERS = signature("gsql/temporal/training_scope.gsql", SCOPE_POPULATION_QUERY)


def live_config(**changes: Any) -> dict[str, Any]:
    """A small, explicit unit run of the built-in settings in an example scope.

    Tests hand prepare() their labels (FrameObservedLabels).
    """
    value = {**run_config(), "scope_id": "example_strict_scope", "create_scope": False}
    value.update(
        dataset_id="unit_fixture",
        epochs=2,
        hidden=16,
        dropout=0,
        batch_size=16,
        steps_per_epoch=2,
        device="cpu",
        threads=1,
    )
    value.update(changes)
    return value


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
    """Scope members as temporal_scope_population prints them with include_observed.

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
    scores = accounts.group_id.map(lambda group: stable_score(group, 42, "split"))
    accounts["split"] = np.where(
        scores < 0.7, "train", np.where(scores < 0.85, "validation", "test")
    )
    return accounts


def scoped_accounts() -> list[dict[str, Any]]:
    """assigned_accounts as temporal_scope_population rows, the split as the partition."""
    accounts = assigned_accounts()
    rows = accounts.drop(columns=["owner_ids", "split"])
    records = rows.assign(partition=accounts.split.map(SPLIT_PHASE)).to_dict("records")
    return cast(list[dict[str, Any]], records)


@dataclass
class FrameObservedLabels:
    """Observed labels from a table: a label source for prepare() other than the graph.

    Population queries run without include_observed for it, as for any source that is
    not GraphObservedLabels.
    """

    labels: pd.DataFrame

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
        "contract_version": CONTRACT_VERSION,
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


def pooled(messages: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Cut messages to the requested pool, per relation, most recent first."""
    payments = params["per_relation"] + params["k_old"] + params["k_div"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(messages, key=lambda m: (-int(m["event_seq"]), m["event_id"])):
        groups.setdefault(item["relation"], []).append(item)
    return [
        item
        for relation, items in groups.items()
        for item in items[: payments if relation in PAYMENT_RELATIONS else params["k_assoc"]]
    ]


def request_keys(params: dict[str, Any]) -> list[ContextKey]:
    return [
        ContextKey(*args, params["scope_id"], params["visibility_phase"])
        for args in zip(
            params["node_types"],
            params["node_ids"],
            params["cutoff_seqs"],
            params["cutoff_times"],
            strict=True,
        )
    ]


class FakeExecutor:
    """In-memory QueryExecutor for the read-only training queries.

    `rows` maps keys to fixed contexts; other keys come from `factory` (default: an
    ok context without messages). `statuses` maps a ContextKey or a node ID to a
    per-request status such as history_capacity_exceeded. `hubs` lists
    (account_id, cutoff_seq) pairs that temporal_hub_registry reports: once at
    phase 3 for an unscoped call, once per phase 1, 2 and 3 for a scoped one.
    `last_visible(index, cutoff_ms)` answers temporal_training_cutoffs.
    `scope_policy` names the scope_unowned rule temporal_scope_policy reports
    for every scope (default "linked", the configuration default). `population`
    holds the rows temporal_scope_population pages through (see scope_population);
    without include_observed their labels are withheld. Subclasses add the other
    population queries a test needs.
    """

    def __init__(
        self,
        rows: dict[ContextKey, dict[str, Any]] | None = None,
        *,
        factory: Callable[[ContextKey], dict[str, Any]] | None = None,
        statuses: dict[ContextKey | str, str] | None = None,
        hubs: Iterable[tuple[str, int]] = (),
        last_visible: Callable[[int, int], int] = lambda index, ms: 100 + index,
        scope_policy: str = "linked",
        population: Iterable[dict[str, Any]] = (),
    ) -> None:
        self.rows = rows or {}
        self.scope_policy = scope_policy
        self.population = sorted(population, key=lambda row: str(row["account_id"]))
        self.factory = factory or context
        self.statuses = statuses or {}
        self.hubs = list(hubs)
        self.last_visible = last_visible
        self.requested: list[ContextKey] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.pools: Counter[tuple[int, ...]] = Counter()
        self.encoded_requests = 0
        self.lock = threading.Lock()

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        with self.lock:
            self.calls.append((name, deepcopy(params)))
        if name == CONTEXT_QUERY:
            return self.context_rows(params)
        if name == HUB_QUERY:
            return self.hub_rows(params)
        if name == SCOPE_POLICY_QUERY:
            return self.scope_policy_rows(params)
        if name == SCOPE_POPULATION_QUERY:
            return self.population_rows(params)
        if name == "temporal_training_cutoffs":
            return [
                {
                    "status": "ok",
                    "last_visible_seqs": {
                        str(ms): self.last_visible(i, ms)
                        for i, ms in enumerate(params["cutoff_times"])
                    },
                }
            ]
        raise AssertionError(f"Unexpected query {name}")

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def status(self, key: ContextKey) -> str | None:
        return self.statuses.get(key) or self.statuses.get(key.node_id)

    def context_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert set(params) == CONTEXT_PARAMETERS, set(params) ^ CONTEXT_PARAMETERS
        keys = request_keys(params)
        assert 1 <= len(keys) <= MAX_REQUEST_KEYS
        pool = tuple(
            int(params[k]) for k in ("per_relation", "k_old", "k_div", "k_assoc", "max_history")
        )
        emit = bool(params["emit_encodings"] and params["include_time_encoding"])
        with self.lock:
            self.requested.extend(keys)
            self.pools[pool] += 1
            self.encoded_requests += int(params["emit_encodings"])
        rows = []
        for index, key in enumerate(keys):
            status = self.status(key)
            if status is not None:
                rows.append({"status": status, "request_index": index})
                continue
            row = deepcopy(self.rows[key]) if key in self.rows else self.factory(key)
            row["messages"] = pooled(row["messages"], params)
            if emit:
                encode(row)
            else:
                row["age_encoding"], row["gap_encoding"] = {}, {}
            rows.append({**row, "request_index": index})
        return rows

    def hub_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert set(params) == HUB_PARAMETERS, set(params) ^ HUB_PARAMETERS
        cutoffs = sorted(set(params["cutoff_seqs"]))
        threshold = int(params["threshold"])
        scope_id = str(params["scope_id"])
        phases = (1, 2, 3) if scope_id else (3,)
        hubs = [
            {
                "account_id": account,
                "cutoff_seq": cutoff,
                "visibility_phase": phase,
                "max_visible": threshold + 1,
                "max_degree": threshold + 1,
                "reason": "visible_history",
            }
            for account, cutoff in self.hubs
            if cutoff in cutoffs
            for phase in phases
        ]
        return [
            {
                "status": "ok",
                "cutoff_seqs": cutoffs,
                "threshold": threshold,
                "scope_id": scope_id,
                "candidates": len({account for account, _ in self.hubs}),
                "hubs": hubs,
            }
        ]

    def scope_policy_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Membership classes a scope created with `scope_policy` would report."""
        assert set(params) == SCOPE_POLICY_PARAMETERS, set(params) ^ SCOPE_POLICY_PARAMETERS
        return [{"status": "ok", "scope_id": params["scope_id"], **scope_counts(self.scope_policy)}]

    def population_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One page of `population` after after_id; labels only with include_observed."""
        assert set(params) == SCOPE_POPULATION_PARAMETERS, set(params) ^ SCOPE_POPULATION_PARAMETERS
        labels = bool(params["include_observed"])
        withheld = {} if labels else {"observed_positive": False, "known_from_ms": 0}
        page = [
            {**row, **withheld}
            for row in self.population
            if str(row["account_id"]) > params["after_id"]
        ]
        return [{"status": "ok"}, {"accounts": page[: params["batch_size"]]}]


def scope_counts(policy: str) -> dict[str, int]:
    """temporal_scope_policy counts of a small scope created with one scope_unowned rule."""
    counts = {
        "members": 12,
        "unowned_accounts": 4,
        "shared_internal": 0,
        "shared_external": 0,
        "independent_internal": 2,
        "independent_external": 2,
        "linked_internal": 0,
        "linked_external": 0,
        "shared_ledger": 0,
        "ledger_accounts": 0,
    }
    if policy in ("shared", "linked"):
        counts.update(shared_external=2, independent_external=0)
    if policy == "linked":
        counts.update(linked_internal=1, independent_internal=1)
    return counts


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
