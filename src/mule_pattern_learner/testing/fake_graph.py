"""In-memory stand-ins for TigerGraph and for context sources.

`FakeExecutor` answers the read-only training queries the way the repository GSQL
does: context requests carry 1..64 keys and exactly the query's parameters,
messages are cut to the requested hop pool, Fourier vectors are printed only when
`emit_encodings` is set, every request index gets exactly one row, and per-request
failures are status rows. The smaller fakes answer one query each, as the tests
that use them need.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from copy import deepcopy
import threading
import time
from types import SimpleNamespace
from typing import Any

import pandas as pd
from pyTigerGraph.common.exception import TigerGraphException

from mule_pattern_learner.config import RunConfig
from mule_pattern_learner.contract.bounds import REQUEST_KEYS
from mule_pattern_learner.contract.feature_groups import FeaturePlan, extraction_plan
from mule_pattern_learner.contract.graph_schema import RELATIONS, ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY_FILE
from mule_pattern_learner.data.contexts import ContextCounts
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.builders import (
    HUB,
    context,
    context_row,
    encode,
    event,
    fake_context,
    synthetic_row,
)
from mule_pattern_learner.tigergraph.context_query import CONTEXT_QUERY
from mule_pattern_learner.tigergraph.cutoffs import CUTOFF_QUERY
from mule_pattern_learner.tigergraph.gsql_text import definitions, parameter_names
from mule_pattern_learner.tigergraph.hubs import HUB_QUERY
from mule_pattern_learner.tigergraph.scope import POPULATION_QUERY, SCOPE_POLICY_QUERY

PAYMENT_RELATIONS = frozenset(RELATIONS[:4])


def signature(path: str, name: str) -> frozenset[str]:
    """Parameter names of one repository query; path is relative to GSQL_DIR."""
    return frozenset(parameter_names(definitions((GSQL_DIR / path).read_text())[name]))


CONTEXT_PARAMETERS = signature(CONTEXT_QUERY_FILE, CONTEXT_QUERY)
HUB_PARAMETERS = signature("queries/hub_accounts.gsql", HUB_QUERY)
SCOPE_POLICY_PARAMETERS = signature("queries/training_scope.gsql", SCOPE_POLICY_QUERY)
POPULATION_PARAMETERS = signature("queries/training_scope.gsql", POPULATION_QUERY)


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
    `scope_policy` names the scope.unowned rule temporal_scope_policy reports
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
        if name == POPULATION_QUERY:
            return self.population_rows(params)
        if name == CUTOFF_QUERY:
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
        assert REQUEST_KEYS.holds(len(keys))
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
        assert set(params) == POPULATION_PARAMETERS, set(params) ^ POPULATION_PARAMETERS
        labels = bool(params["include_observed"])
        withheld = {} if labels else {"observed_positive": False, "known_from_ms": 0}
        page = [
            {**row, **withheld}
            for row in self.population
            if str(row["account_id"]) > params["after_id"]
        ]
        return [{"status": "ok"}, {"accounts": page[: params["batch_size"]]}]


def scope_counts(policy: str) -> dict[str, int]:
    """temporal_scope_policy counts of a small scope created with one scope.unowned rule."""
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


class ContextServer:
    """Fake temporal_training_context endpoint with per-request statuses."""

    def __init__(
        self,
        statuses: dict[ContextKey, str] | None = None,
        *,
        delay: float = 0.0,
        corrupt: bool = False,
        omit_encodings: bool = False,
    ) -> None:
        self.statuses = statuses or {}
        self.delay, self.corrupt, self.omit_encodings = delay, corrupt, omit_encodings
        self.calls: list[dict[str, Any]] = []
        self.requested: Counter[tuple[int, ContextKey]] = Counter()
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        assert name == "temporal_training_context"
        hop = 1 if params["k_assoc"] else 2
        keys = request_keys(params)
        with self.lock:
            self.calls.append(params)
            self.requested.update((hop, key) for key in keys)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            emit = params["emit_encodings"] and params["include_time_encoding"]
            rows = []
            for index, key in enumerate(keys):
                if key in self.statuses:
                    rows.append({"status": self.statuses[key], "request_index": index})
                    continue
                messages = [event(key.cutoff_seq - 1, key), event(key.cutoff_seq - 3, key, gap=0)]
                row = context_row(key, messages, encodings=emit and not self.omit_encodings)
                if emit and self.corrupt:
                    first = next(iter(row["age_encoding"]))
                    row["age_encoding"][first][3] += 0.01
                rows.append({**row, "request_index": index})
            return rows
        finally:
            with self.lock:
                self.active -= 1


class Runner:
    """A QueryExecutor backed by a function."""

    def __init__(self, respond: Callable[[str, dict[str, Any]], list[dict[str, Any]]]) -> None:
        self.respond = respond

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        return self.respond(name, params)


LINKED_COUNTS = {
    "shared_internal": 0,
    "shared_external": 40,
    "independent_internal": 7,
    "independent_external": 0,
    "linked_internal": 90,
    "linked_external": 0,
    "shared_ledger": 4,
    "ledger_accounts": 4,
    "members": 500,
}


def policy_counts(policy: str) -> dict[str, int]:
    """Membership classes a scope created with `policy` reports."""
    if policy == "linked":
        return dict(LINKED_COUNTS)
    if policy == "shared":
        return {**LINKED_COUNTS, "independent_internal": 97, "linked_internal": 0}
    if policy == "independent":  # every scope created before the policy, strict_mule_v1
        return {
            **LINKED_COUNTS,
            "shared_external": 0,
            "independent_external": 40,
            "independent_internal": 97,
            "linked_internal": 0,
            "shared_ledger": 0,
        }
    return {**LINKED_COUNTS, "shared_internal": 97, "linked_internal": 0}  # a retired draft


class ScopeServer:
    """A TigerGraph fake for scope headers, scope creation and temporal_scope_policy."""

    def __init__(self, header: dict[str, Any] | None, policy: str) -> None:
        self.header, self.policy = header, policy
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.client = SimpleNamespace(
            conn=SimpleNamespace(getVerticesById=self.vertices), graphname="Mule_Pattern_Learner"
        )

    def call(self, operation: Callable[[Any], Any], *, what: str) -> Any:
        """A connection operation, run once (the executor would retry it)."""
        return operation(self.client.conn)

    def vertices(self, *args: Any) -> list[dict[str, Any]]:
        if self.header is None:
            raise TigerGraphException("vertex not found", "601")
        return [{"attributes": dict(self.header)}]

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        if name == "temporal_scope_policy":
            return [{"status": "ok", "scope_id": params["scope_id"], **policy_counts(self.policy)}]
        if name == "temporal_create_training_scope":
            self.policy = params["unowned_policy"]
            return [{"status": "ok", "expected_members": 3}]
        if name == "temporal_finalize_training_scope":
            self.header = {"ready": True, "source_id": "unit_snapshot", "split_seed": 42}
        return [{"status": "ok"}]


class FakeStore:
    """`ContextReader` with `fetch(keys, *, hop)`; None marks a rejected context."""

    plan: FeaturePlan
    sampler: SamplerPlan
    query_calls: int
    rejections: Counter[str]
    rejections_by_hop: dict[int, Counter[str]]
    counts: ContextCounts

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
        self.rejections_by_hop = {}
        self.counts = ContextCounts()
        self.query_calls = 0

    def row(self, key: ContextKey, hop: int = 1) -> dict[str, Any]:
        if (hop, key) not in self.rows:
            pool = self.sampler.pool(hop)
            self.rows[hop, key] = synthetic_row(key, pool, encodings=self.encodings)
        return self.rows[hop, key]

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        self.calls.append((hop, list(keys)))
        self.counts.ask(dict.fromkeys(keys), hop)
        self.query_calls += 1
        out: list[dict[str, Any] | None] = []
        for key in keys:
            if key in self.reject:
                self.rejections["history_capacity_exceeded"] += 1
                self.rejections_by_hop.setdefault(hop, Counter())["history_capacity_exceeded"] += 1
                out.append(None)
            else:
                out.append(self.row(key, hop))
        return out

    def close(self, *, wait: bool = True) -> None: ...


class FakeSource:
    """Thread-safe in-memory ContextReader with optional rejections and failures."""

    def __init__(
        self,
        config: RunConfig,
        *,
        reject: frozenset[str] = frozenset(),
        fail: Callable[[list[ContextKey], int, Counter[str]], bool] | None = None,
    ) -> None:
        self.plan = extraction_plan(config.feature_plan())
        self.sampler = config.sampler
        self.reject, self.fail = reject, fail
        self.query_calls = 0
        self.rejections: Counter[str] = Counter()
        self.rejections_by_hop: dict[int, Counter[str]] = {}
        self.counts = ContextCounts()
        self.calls: Counter[str] = Counter()
        self.lock = threading.Lock()
        self.closed = False

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        with self.lock:
            phases = {k.visibility_phase for k in keys}
            for phase in phases:
                self.calls[f"hop{hop}_phase{phase}"] += 1
            if self.fail is not None and self.fail(keys, hop, self.calls):
                raise RuntimeError("injected source failure")
            self.query_calls += 1
            self.counts.ask(dict.fromkeys(keys), hop)
            rows: list[dict[str, Any] | None] = []
            for key in keys:
                if key.node_id in self.reject:
                    self.rejections["missing_entity"] += 1
                    self.rejections_by_hop.setdefault(hop, Counter())["missing_entity"] += 1
                    rows.append(None)
                else:
                    rows.append(fake_context(key))
            return rows

    def close(self, *, wait: bool = True) -> None:
        self.closed = True


class ScoringExecutor:
    def __init__(self, accounts: pd.DataFrame | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.accounts = accounts

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params))
        if name == CUTOFF_QUERY:
            return [{"status": "ok", "last_visible_seqs": {str(params["cutoff_times"][0]): 29_999}}]
        if name == HUB_QUERY:
            (cutoff,) = params["cutoff_seqs"]
            # Score-new is unscoped: phase-3 rows over all visible history.
            assert not params.get("scope_id")
            echo = {k: params[k] for k in ("threshold", "cutoff_seqs", "scope_id") if k in params}
            return [
                {
                    "status": "ok",
                    **echo,
                    "hubs": [
                        {
                            "account_id": HUB,
                            "cutoff_seq": cutoff,
                            "visibility_phase": 3,
                            "max_visible": 9000,
                            "max_degree": 9000,
                            "reason": "visible_history",
                        }
                    ],
                }
            ]
        if name == POPULATION_QUERY:
            assert params["include_observed"] is False and self.accounts is not None
            page = [
                {"account_id": a, "partition": 3, "first_seen_ts_ms": 1}
                for a in self.accounts.account_id
                if a > params["after_id"]
            ]
            return [{"status": "ok", "accounts": page}]
        raise AssertionError(name)
