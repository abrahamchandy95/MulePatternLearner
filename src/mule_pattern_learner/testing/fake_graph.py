"""In-memory stand-ins for TigerGraph and for context sources.

`FakeTigerGraph` is a ConnectionExecutor of the repository's queries. `run` answers them
the way the repository GSQL does: context requests carry 1..64 keys and exactly the
query's parameters, messages are cut to the requested hop pool, Fourier vectors are
printed only when `emit_encodings` is set, every request index gets exactly one row,
per-request failures are status rows, account queries page by account id, and the
queries that write run with one attempt. `call` and `gsql` run on its connection
(FakeConnection), which answers SHOW QUERY with the installed text, lists the installed
endpoints, creates, installs and drops queries and reports the schema, the vertex counts
and the scope headers; every GSQL statement that writes must run with one attempt too.
It is the one fake executor; a few tests script a connection of their own behind the
real executor (testing.fake_connection) for a query it does not answer. FakeStore and
FakeSource stand in for context sources, for
the tests that need no graph behind them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any, Literal

from pyTigerGraph.common.exception import TigerGraphException

from mule_pattern_learner.config import DEFAULT_CONFIG, RunConfig, ScopeConfig
from mule_pattern_learner.contract.analytics_features import ANALYTICS_GROUPS
from mule_pattern_learner.contract.bounds import REQUEST_KEYS
from mule_pattern_learner.contract.feature_groups import FeaturePlan, extraction_plan
from mule_pattern_learner.contract.graph_schema import PAYMENT_RELATIONS, ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import (
    ANALYTICS_CONTEXT_FILE,
    ANALYTICS_CONTEXT_QUERY,
    ANALYTICS_CONTRACT,
    ANALYTICS_QUERY_FILES,
    CONTEXT_QUERY,
    CONTEXT_QUERY_FILE,
    CREATE_SCOPE_QUERY,
    CUTOFF_QUERY,
    FINALIZE_SCOPE_QUERY,
    GRAPH_NAME,
    HUB_QUERY,
    POPULATION_QUERY,
    REVEAL_QUERY,
    SCOPE_POLICY_QUERY,
    SCOPE_VERTEX,
    TRAINING_QUERY_FILES,
    TRUTH_QUERY,
)
from mule_pattern_learner.data.contexts import ContextCounts
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.data.preparation import prepare
from mule_pattern_learner.paths import GSQL_DIR, DatasetPaths
from mule_pattern_learner.testing.builders import (
    UNIT_SOURCE,
    context,
    encode,
    fake_context,
    ground_truth_rows,
    neighbourhood,
    scope_population,
    synthetic_row,
)
from mule_pattern_learner.testing.fake_connection import FakeClient
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.gsql_text import (
    calls,
    definitions,
    parameter_names,
    repository_queries,
)
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabelReader
from mule_pattern_learner.tigergraph.oracle import REVEAL_INPUTS_QUERY
from mule_pattern_learner.tigergraph.scope import SHARE_ATTRIBUTES, TigerGraphScopeReader
from mule_pattern_learner.tigergraph.scope_types import (
    ScopeTypes,
    declared_scope_types,
    parsed_scope_types,
    uses_scope_types,
)


def signature(path: str, name: str) -> frozenset[str]:
    """Parameter names of one repository query; path is relative to GSQL_DIR."""
    return frozenset(parameter_names(definitions((GSQL_DIR / path).read_text())[name]))


CONTEXT_PARAMETERS = signature(CONTEXT_QUERY_FILE, CONTEXT_QUERY)
ANALYTICS_PARAMETERS = signature(ANALYTICS_CONTEXT_FILE, ANALYTICS_CONTEXT_QUERY)
# The parameters an analytics request must name: the keys, the pool and the scope; the
# include flags and emit_encodings have defaults.
ANALYTICS_REQUIRED = frozenset(
    name for name in ANALYTICS_PARAMETERS if not name.startswith("include_")
) - {"emit_encodings"}
# The analytics message fields of a message the fake graph's analytics query prints: the
# pair window counts and the device and IP ages, none of them seen.
ANALYTICS_MESSAGE_FIELDS: dict[str, float | bool] = {
    name: (False if name.endswith("_present") else 0.0)
    for spec in ANALYTICS_GROUPS.values()
    if spec.path == "message"
    for name in spec.names
}
HUB_PARAMETERS = signature("queries/hub_accounts.gsql", HUB_QUERY)
SCOPE_POLICY_PARAMETERS = signature("queries/training_scope.gsql", SCOPE_POLICY_QUERY)
POPULATION_PARAMETERS = signature("queries/training_scope.gsql", POPULATION_QUERY)
CREATE_SCOPE_PARAMETERS = signature("queries/training_scope.gsql", CREATE_SCOPE_QUERY)
FINALIZE_SCOPE_PARAMETERS = signature("queries/training_scope.gsql", FINALIZE_SCOPE_QUERY)
TRUTH_PARAMETERS = signature("evaluation/ground_truth.gsql", TRUTH_QUERY)
# The queries FakeTigerGraph answers that write to the graph: they must run once.
WRITE_QUERIES = frozenset({CREATE_SCOPE_QUERY, FINALIZE_SCOPE_QUERY})
# The scope types of a current graph: those gsql/schema/scope_vertex.gsql declares.
SCOPE_TYPES = declared_scope_types()
# The scope types of a graph created before the scope vertex recorded its split shares.
EARLIER_SCOPE_TYPES = replace(
    SCOPE_TYPES,
    vertex_attributes=tuple(
        (name, kind) for name, kind in SCOPE_TYPES.vertex_attributes if name not in SHARE_ATTRIBUTES
    ),
)


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


def pool_of(params: dict[str, Any]) -> tuple[int, ...]:
    """The candidate pool a context request asks for: one per hop."""
    return tuple(
        int(params[k]) for k in ("per_relation", "k_old", "k_div", "k_assoc", "max_history")
    )


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


class FakeTigerGraph:
    """In-memory ConnectionExecutor of the repository's queries (see the module docstring).

    `rows` maps keys to fixed contexts; other keys come from `factory` (default: an
    ok context without messages). `statuses` maps a ContextKey or a node ID to a
    per-request status such as history_capacity_exceeded. `hubs` lists
    (account_id, cutoff_seq) pairs that the hub query reports: once at
    phase 3 for an unscoped call, once per phase 1, 2 and 3 for a scoped one.
    `last_visible(index, cutoff_ms)` answers the cutoff query.
    `scope_policy` names the scope.unowned rule the scope policy query reports
    for every scope (default "linked", the configuration default; policy_counts).
    `population` holds the rows the scope population query pages through;
    without include_observed their labels are withheld. `truth` holds the rows the
    ground-truth query pages through, with its field names (ground_truth_rows).
    `reveal` holds what the reveal's interpreted inputs query prints (reveal_inputs).
    `other_reveal` is the count of mules whose label another reveal wrote, which the
    reveal's dry run, the label check before a streamed run, reports (default 0: the
    labels are the dataset's reveal's).
    The analytics context query answers each key with its context row (as the context
    query would, cut to the requested pool), relabelled with ANALYTICS_CONTRACT, its
    messages carrying the analytics message fields unseen and its node features joined
    by `analytics(key)`, the analytics node features of the key (default: none).

    The connection's state: `scopes` maps scope ids to the attributes of their scope
    vertex, which the scope creation queries add; `counts` are the vertex counts by
    type (default: the population's accounts); `queries` maps the installed queries to
    their text (default: every repository query), except that `stale` names queries
    whose installed text differs until they are created again; and `scope_types` are
    the schema's scope vertex and edge types (default: SCOPE_TYPES, those
    gsql/schema/scope_vertex.gsql declares; EARLIER_SCOPE_TYPES lack the split shares;
    None, neither type). `writes` records the GSQL statements that change it and the
    install requests, in order.

    A context request waits `delay` seconds before it is answered, and `encodings`
    injects the Fourier faults of context_rows.

    Tests change what a query does with two hooks. `before(name, params)` runs before
    each query is answered: it may check the parameters, wait or raise. `answers` maps a
    query name to a function of its parameters that answers it instead, for scripted or
    malformed responses. Every call is recorded in `calls` either way.
    """

    graph_name = GRAPH_NAME

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
        truth: Iterable[dict[str, Any]] = (),
        reveal: list[dict[str, Any]] | None = None,
        other_reveal: int = 0,
        analytics: Callable[[ContextKey], dict[str, float]] | None = None,
        scopes: dict[str, dict[str, Any]] | None = None,
        counts: dict[str, int] | None = None,
        queries: Mapping[str, str] | None = None,
        stale: Iterable[str] = (),
        scope_types: ScopeTypes | None = SCOPE_TYPES,
        delay: float = 0.0,
        encodings: Literal["exact", "perturbed", "omitted"] = "exact",
        before: Callable[[str, dict[str, Any]], None] | None = None,
        answers: Mapping[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.rows = rows or {}
        self.scope_policy = scope_policy
        self.population = sorted(population, key=lambda row: str(row["account_id"]))
        self.truth = sorted(truth, key=lambda row: str(row["account_id"]))
        self.reveal = reveal
        self.other_reveal = other_reveal
        self.analytics = analytics or no_analytics
        self.factory = factory or context
        self.statuses = statuses or {}
        self.hubs = list(hubs)
        self.last_visible = last_visible
        self.scopes = {scope_id: dict(header) for scope_id, header in (scopes or {}).items()}
        self.counts = dict(counts) if counts is not None else {"Account": len(self.population)}
        self.stale = frozenset(stale)
        self.scope_types = scope_types
        self.writes: list[str] = []
        self.delay, self.encodings = delay, encodings
        self.before = before
        self.answers = dict(answers or {})
        self.client = FakeClient(FakeConnection(self, repository(queries)))
        self.requested: list[ContextKey] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.pools: Counter[tuple[int, ...]] = Counter()
        self.encoded_requests = 0
        # Context requests being answered now, and the most at once.
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def run(self, name: str, params: dict[str, Any], **options: Any) -> list[dict[str, Any]]:
        with self.lock:
            self.calls.append((name, deepcopy(params)))
        if self.before is not None:
            self.before(name, params)
        if name in self.answers:
            return self.answers[name](params)
        if name in WRITE_QUERIES:
            assert options.get("attempts") == 1, f"{name} writes, so it must run once"
        if name == CONTEXT_QUERY:
            return self.context_rows(params)
        if name == ANALYTICS_CONTEXT_QUERY:
            return self.analytics_rows(params)
        if name == HUB_QUERY:
            return self.hub_rows(params)
        if name == SCOPE_POLICY_QUERY:
            return self.scope_policy_rows(params)
        if name == POPULATION_QUERY:
            return self.population_rows(params)
        if name == TRUTH_QUERY:
            return self.truth_rows(params)
        if name == REVEAL_QUERY:
            return self.reveal_check(params, options)
        if name == CREATE_SCOPE_QUERY:
            return self.create_scope(params)
        if name == FINALIZE_SCOPE_QUERY:
            return self.finalize_scope(params)
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

    def call(
        self,
        operation: Callable[[Any], Any],
        *,
        what: str,
        attempts: int | None = None,
        **_: Any,
    ) -> Any:
        """A connection operation, run once (the real executor would retry it).

        The install request writes, so it must run with one attempt, as GSQL writes must.
        """
        if what == "installQueries":
            assert attempts == 1, f"{what} writes, so it must run once"
        return operation(self.client.conn)

    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str:
        if "SHOW QUERY" not in text:
            assert attempts == 1, f"{what} writes, so it must run once"
        return self.client.conn.gsql(text)

    def reveal_check(self, params: dict[str, Any], options: dict[str, Any]) -> list[dict[str, Any]]:
        """The reveal's dry run on a graph with known labels: the label check's answer.

        Preparation's reveal, which writes, is never answered here (tests replace
        pipeline.prepare.ensure_revealed_labels).
        """
        assert params["apply"] is False and not params.get("force"), params
        assert options.get("attempts") == 1
        record = f"phantomledger_role;reveal_v1;salt={params['salt']};budget={params['budget']}"
        status = "revealed_differently" if self.other_reveal else "already_revealed"
        return [
            {
                "status": status,
                "known_labels": len(self.population),
                "revealed_labels": 0,
                "other_reveal": self.other_reveal,
                "reveal_record": record,
            }
        ]

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def status(self, key: ContextKey) -> str | None:
        return self.statuses.get(key) or self.statuses.get(key.node_id)

    def context_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One row per requested key, answered after `delay` seconds.

        `encodings` "perturbed" shifts one printed Fourier value and "omitted" prints
        none although they were asked for: the faults the spot checks catch.
        """
        assert set(params) == CONTEXT_PARAMETERS, set(params) ^ CONTEXT_PARAMETERS
        keys = request_keys(params)
        assert REQUEST_KEYS.holds(len(keys))
        emit = bool(params["emit_encodings"] and params["include_time_encoding"])
        with self.lock:
            self.requested.extend(keys)
            self.pools[pool_of(params)] += 1
            self.encoded_requests += int(params["emit_encodings"])
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            rows = []
            for index, key in enumerate(keys):
                status = self.status(key)
                if status is not None:
                    rows.append({"status": status, "request_index": index})
                    continue
                row = deepcopy(self.rows[key]) if key in self.rows else self.factory(key)
                row["messages"] = pooled(row["messages"], params)
                row["age_encoding"], row["gap_encoding"] = {}, {}
                if emit and self.encodings != "omitted":
                    encode(row)
                if emit and self.encodings == "perturbed" and row["age_encoding"]:
                    first = next(iter(row["age_encoding"]))
                    row["age_encoding"][first][3] += 0.01
                rows.append({**row, "request_index": index})
            return rows
        finally:
            with self.lock:
                self.active -= 1

    def analytics_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One row per requested key, as fetch_analytics_context prints it (see the class)."""
        missing = ANALYTICS_REQUIRED - set(params)
        assert not missing and set(params) <= ANALYTICS_PARAMETERS, set(params) ^ missing
        keys = request_keys(params)
        assert REQUEST_KEYS.holds(len(keys))
        with self.lock:
            self.requested.extend(keys)
        rows = []
        for index, key in enumerate(keys):
            status = self.status(key)
            if status is not None:
                rows.append({"status": status, "request_index": index})
                continue
            row = deepcopy(self.rows[key]) if key in self.rows else self.factory(key)
            row["messages"] = [
                {**item, **ANALYTICS_MESSAGE_FIELDS} for item in pooled(row["messages"], params)
            ]
            row["features"] = {**row["features"], **self.analytics(key)}
            row["contract_version"] = ANALYTICS_CONTRACT
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
        counts = policy_counts(self.scope_policy)
        return [{"status": "ok", "scope_id": params["scope_id"], **counts}]

    def population_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One page of `population` after after_id; labels only with include_observed."""
        assert set(params) == POPULATION_PARAMETERS, set(params) ^ POPULATION_PARAMETERS
        labels = bool(params["include_observed"])
        withheld = {} if labels else {"observed_positive": False, "known_from_ms": 0}
        rows = [{**row, **withheld} for row in self.population]
        return [{"status": "ok"}, {"accounts": page(rows, params)}]

    def truth_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One page of `truth` after after_id, in the ground-truth query's fields."""
        assert set(params) == TRUTH_PARAMETERS, set(params) ^ TRUTH_PARAMETERS
        return [{"status": "ok"}, {"accounts": page(self.truth, params)}]

    def create_scope(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """A scope vertex that is not ready yet, with the rule its membership follows.

        Split shares that are not all positive, or do not add up to 1 within 1e-6, are
        invalid_parameters, and nothing is written, as create_training_scope refuses them.
        """
        # max_iterations has a default, which the pipeline keeps.
        assert set(params) <= CREATE_SCOPE_PARAMETERS, set(params) - CREATE_SCOPE_PARAMETERS
        shares = [params[name] for name in SHARE_ATTRIBUTES]
        if min(shares) <= 0 or not 0.999999 <= sum(shares) <= 1.000001:
            return [{"status": "invalid_parameters"}]
        if params["scope_id"] in self.scopes:
            return [{"status": "scope_already_exists"}]
        self.scope_policy = params["unowned_policy"]
        header = {name: params[name] for name in ("source_id", "split_seed", *SHARE_ATTRIBUTES)}
        self.scopes[params["scope_id"]] = {"ready": False, **header}
        return [{"status": "ok", "expected_members": len(self.population)}]

    def finalize_scope(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Mark a created scope ready once its membership count is the expected one."""
        assert set(params) == FINALIZE_SCOPE_PARAMETERS, set(params) ^ FINALIZE_SCOPE_PARAMETERS
        members = len(self.population)
        if params["scope_id"] not in self.scopes or params["expected_members"] != members:
            return [{"status": "invalid_scope"}]
        self.scopes[params["scope_id"]]["ready"] = True
        return [{"status": "ok", "members": members}]


# The accounts of the scope prepared_graph builds.
PREPARED_ACCOUNTS = 200


def ready_scope(
    source_id: str, split_seed: int = 42, scope: ScopeConfig = DEFAULT_CONFIG.scope
) -> dict[str, Any]:
    """The attributes of a ready scope vertex: its source, split seed and scope's shares."""
    shares = dict(zip(SHARE_ATTRIBUTES, scope.shares, strict=True))
    return {"ready": True, "source_id": source_id, "split_seed": split_seed, **shares}


def prepared_graph(
    data: Path, config: RunConfig, **options: Any
) -> tuple[FakeTigerGraph, DatasetPaths]:
    """A fake graph with a scope and its ground truth, and config's dataset prepared from it.

    The scope holds PREPARED_ACCOUNTS accounts (scope_population), every label known
    (ground_truth_rows); each context is a deterministic neighbourhood, and N3 is a hub
    at the three cutoffs. The dataset is prepared in data from UNIT_SOURCE. ``options``
    are FakeTigerGraph's (statuses, analytics, reveal, ...).
    """
    population = scope_population(PREPARED_ACCOUNTS)
    header = ready_scope(UNIT_SOURCE, config.dataset.split_seed, config.scope)
    settings: dict[str, Any] = {
        "factory": neighbourhood,
        "hubs": [("N3", cutoff) for cutoff in (101, 102, 103)],
        "population": population,
        "truth": ground_truth_rows(population),
        "scopes": {config.scope.id: header},
    }
    graph = FakeTigerGraph(**{**settings, **options})
    dataset = DatasetPaths.of(dataset_id(UNIT_SOURCE, config), data)
    prepare(
        config,
        UNIT_SOURCE,
        dataset,
        {"Account": PREPARED_ACCOUNTS},
        TigerGraphObservedLabelReader(),
        scope=TigerGraphScopeReader(graph),
        cutoffs=TigerGraphCutoffReader(graph),
        hub_reader=TigerGraphHubReader(graph),
    )
    return graph, dataset


def no_analytics(key: ContextKey) -> dict[str, float]:
    """The analytics node features of a key when a test names none: none."""
    return {}


def page(rows: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows after params' after_id, at most its batch_size: one page of an account query."""
    after = [row for row in rows if str(row["account_id"]) > params["after_id"]]
    return after[: params["batch_size"]]


def repository(queries: Mapping[str, str] | None) -> dict[str, str]:
    """The installed text of each query: queries, or every repository query by default."""
    if queries is None:
        files = (*TRAINING_QUERY_FILES, *ANALYTICS_QUERY_FILES)
        return {name: text for name, (_, text) in repository_queries(files).items()}
    return dict(queries)


# The calls between the retired queries, as the GSQL before the rename made them (commit
# f203226): the Fourier wrapper, both pair encoders and the context query called the
# Fourier values, and the reveal called its uniforms.
RETIRED_CALLS = {
    "temporal_training_context": "temporal_fourier64_values",
    "temporal_fourier64": "temporal_fourier64_values",
    "zelle_pair_time64": "temporal_fourier64_values",
    "payment_pair_time64": "temporal_fourier64_values",
    "temporal_reveal_mule_labels": "temporal_reveal_uniforms",
}


def retired_query(name: str, calls: str | None = None) -> str:
    """A query as the code before the rename installed it: `name`, calling what it called.

    It calls `calls` instead when given, which makes a query of someone else's that calls
    a retired one.
    """
    callee = calls or RETIRED_CALLS.get(name)
    body = f"{callee}(0); PRINT 1;" if callee else "PRINT 1;"
    header = f"CREATE QUERY {name}(INT unused = 0) FOR GRAPH {GRAPH_NAME} SYNTAX V2"
    return f"{header} {{ {body} }}"


def graph_schema(scope_types: ScopeTypes | None) -> dict[str, Any]:
    """What getSchema answers of a graph with these scope types (None: neither type).

    Account and Party are named only; the scope types are given in full, in
    TigerGraph's form: the vertex type's primary id apart from its other attributes, and
    the edge type's vertex type pairs and reverse edge.
    """
    vertices: list[dict[str, Any]] = [{"Name": "Account"}, {"Name": "Party"}]
    edges: list[dict[str, Any]] = []
    if scope_types is not None:
        (key, key_type), *attributes = scope_types.vertex_attributes
        vertices.append(
            {
                "Name": scope_types.vertex,
                "PrimaryId": {
                    "AttributeName": key,
                    "AttributeType": {"Name": key_type},
                    "PrimaryIdAsAttribute": True,
                },
                "Attributes": [schema_attribute(*item) for item in attributes],
            }
        )
        edges.append(
            {
                "Name": scope_types.edge,
                "IsDirected": True,
                "FromVertexTypeName": "*",
                "ToVertexTypeName": scope_types.vertex,
                "EdgePairs": [{"From": a, "To": b} for a, b in scope_types.endpoints],
                "Attributes": [schema_attribute(*item) for item in scope_types.edge_attributes],
                "Config": {"REVERSE_EDGE": scope_types.reverse_edge},
            }
        )
    return {"GraphName": GRAPH_NAME, "VertexTypes": vertices, "EdgeTypes": edges}


def schema_attribute(name: str, kind: str) -> dict[str, Any]:
    """An attribute as getSchema gives it."""
    return {"AttributeName": name, "AttributeType": {"Name": kind}}


class FakeConnection:
    """The pyTigerGraph connection of a FakeTigerGraph: schema, counts and queries.

    GSQL runs SHOW QUERY, CREATE OR REPLACE QUERY (which disables the endpoint until the
    query is installed again), DROP QUERY (refused, as TigerGraph refuses it, while another
    query calls the query) and the scope schema changes; any other GSQL fails. The
    schema change of gsql/schema/scope_vertex.gsql adds the types it declares, refused
    while the graph has them; a job that drops them is refused while a query on the
    graph names one, and deletes every scope. installQueries installs at once; the
    -force flag is recorded. The one interpreted query it runs is the reveal's inputs
    query, which prints the graph's `reveal` rows.
    """

    def __init__(self, graph: FakeTigerGraph, queries: dict[str, str]) -> None:
        self.graph = graph
        self.shown = queries
        self.enabled = dict.fromkeys(queries, True)

    def getVerticesById(self, vertex_type: str, ids: list[str]) -> list[dict[str, Any]]:
        assert vertex_type == SCOPE_VERTEX, vertex_type
        found = [self.graph.scopes[i] for i in ids if i in self.graph.scopes]
        if not found:
            raise TigerGraphException("vertex not found", "601")
        return [{"attributes": dict(header)} for header in found]

    def getVertexCount(self, vertex_type: str, realtime: bool = False) -> dict[str, int]:
        assert vertex_type == "*", vertex_type
        return {**self.graph.counts, SCOPE_VERTEX: len(self.graph.scopes)}

    def runInterpretedQuery(self, text: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert text == REVEAL_INPUTS_QUERY and set(params) == {"scope_id"}, text[:80]
        assert self.graph.reveal is not None, "the fake graph has no reveal inputs"
        self.graph.calls.append(("reveal inputs", dict(params)))
        return deepcopy(self.graph.reveal)

    def getSchema(self, force: bool = False) -> dict[str, Any]:
        return graph_schema(self.graph.scope_types)

    def text(self, name: str) -> str:
        """What SHOW QUERY prints of a query: a stale one's text differs from its file's."""
        text = self.shown[name]
        return text.replace("{", "{ INT stale_marker = 0;", 1) if name in self.graph.stale else text

    def getInstalledQueries(self) -> dict[str, dict[str, Any]]:
        builtin = {"query", "read_committed"}
        return {
            f"GET /query/{GRAPH_NAME}/{name}": {
                "enabled": True,
                "parameters": {key: {} for key in parameter_names(self.text(name)) | builtin},
            }
            for name in self.shown
            if self.enabled[name]
        }

    def installQueries(
        self, names: list[str], wait: bool = False, flag: str | None = None
    ) -> dict[str, Any]:
        forced = "-FORCE " if flag == "-force" else ""
        self.graph.writes.append(f"INSTALL QUERY {forced}" + ", ".join(names))
        for name in names:
            self.enabled[name] = True
        return {"error": False, "message": "Query installation finished: SUCCESS"}

    def gsql(self, text: str) -> str:
        if "SHOW QUERY" in text:
            name = text.rsplit(" ", 1)[1]
            return self.text(name) if name in self.shown else "Query not found"
        self.graph.writes.append(text)
        if "DROP QUERY" in text:
            name = text.rsplit(" ", 1)[1]
            callers = [
                other for other, body in self.shown.items() if other != name and calls(body, name)
            ]
            if callers:
                return f"Query {name} cannot be dropped: {', '.join(callers)} call it."
            self.shown.pop(name)
            self.enabled.pop(name)
            return f"Successfully dropped queries on the graph '{GRAPH_NAME}': [{name}]."
        if "SCHEMA_CHANGE JOB" in text:
            return self.schema_change(text)
        created = definitions(text)
        assert created, "the fake graph runs no other GSQL"
        for name, definition in created.items():
            self.shown[name] = definition
            self.enabled[name] = False
        self.graph.stale = self.graph.stale - set(created)
        return f"Successfully created queries: [{', '.join(created)}]."

    def schema_change(self, text: str) -> str:
        """Add the scope types a job declares, or drop the graph's, as TigerGraph would."""
        graph = self.graph
        if "ADD VERTEX" in text:
            if graph.scope_types is not None:
                return f"Failed: the vertex type {SCOPE_VERTEX} already exists."
            graph.scope_types = parsed_scope_types(text)
            return "Local schema change succeeded."
        assert "DROP VERTEX" in text or "DROP EDGE" in text, text[:80]
        assert graph.scope_types is not None, "the graph has no scope types to drop"
        users = [n for n, body in self.shown.items() if uses_scope_types(body, graph.scope_types)]
        if users:
            return f"Failed: the queries {', '.join(users)} use the types to drop."
        graph.scope_types = None
        graph.scopes.clear()
        return "Local schema change succeeded."


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
    if policy == "independent":
        return {
            **LINKED_COUNTS,
            "shared_external": 0,
            "independent_external": 40,
            "independent_internal": 97,
            "linked_internal": 0,
            "shared_ledger": 0,
        }
    # Any other name: internal customer accounts shared, which no rule does.
    return {**LINKED_COUNTS, "shared_internal": 97, "linked_internal": 0}


class FakeStore:
    """`ContextReader` with `fetch(keys, *, hop)`; None marks a rejected context."""

    plan: FeaturePlan
    sampler: SamplerPlan
    database_calls: int
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
        self.database_calls = 0

    def row(self, key: ContextKey, hop: int = 1) -> dict[str, Any]:
        if (hop, key) not in self.rows:
            pool = self.sampler.pool(hop)
            self.rows[hop, key] = synthetic_row(key, pool, encodings=self.encodings)
        return self.rows[hop, key]

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        self.calls.append((hop, list(keys)))
        self.counts.ask(dict.fromkeys(keys), hop)
        self.database_calls += 1
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
        self.database_calls = 0
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
            self.database_calls += 1
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
