"""Installing and verifying only the stale training queries."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from mule_pattern_learner.contract.server import (
    ANALYTICS_QUERY_FILES,
    CONTEXT_QUERY,
    CUTOFF_QUERY,
    FOURIER_QUERY,
    GRAPH_NAME,
    HUB_QUERY,
    QUERY_FILES,
    RETIRED_QUERIES,
    REVEAL_QUERY,
    REVEAL_UNIFORMS_QUERY,
    TRAINING_QUERY_FILES,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.fake_connection import executor
from mule_pattern_learner.testing.fake_graph import RETIRED_CALLS, FakeTigerGraph, retired_query
from mule_pattern_learner.tigergraph import gsql_text, installer
from mule_pattern_learner.tigergraph.executor import TransientQueryError

INSTALL_FILES = ("queries/label_reveal.gsql", "queries/split_cutoffs.gsql")


def endpoint(parameters: set[str], enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "parameters": {name: {} for name in parameters | {"query", "read_committed"}},
    }


def test_verify_sources_requires_matching_text_and_enabled_endpoints() -> None:
    files = ("queries/split_cutoffs.gsql", "queries/fourier64.gsql")
    expected: dict[str, str] = {}
    for path in files:
        expected.update(gsql_text.definitions((GSQL_DIR / path).read_text()))
    params = {name: gsql_text.parameter_names(text) for name, text in expected.items()}
    assert params[CUTOFF_QUERY] == {"cutoff_times"}

    def conn(
        text_for: Callable[[str], str], endpoints: dict[str, dict[str, Any]]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            gsql=lambda text: text_for(text.rsplit(" ", 1)[1]),
            getInstalledQueries=lambda: {
                f"GET /query/{GRAPH_NAME}/{name}": value for name, value in endpoints.items()
            },
        )

    good = {name: endpoint(value) for name, value in params.items()}
    assert set(installer.verify_sources(executor(conn(expected.__getitem__, good)), files)) == set(
        expected
    )
    disabled = {**good, CUTOFF_QUERY: endpoint({"cutoff_times"}, False)}
    renamed = {**good, CUTOFF_QUERY: endpoint({"other"})}
    cases = [
        (conn(expected.__getitem__, disabled), "not installed"),
        (conn(lambda n: expected[n].replace("24", "25"), good), "differs"),
        (conn(expected.__getitem__, renamed), "parameters differ"),
        (conn(lambda n: "", good), "missing"),
    ]
    for fake, message in cases:
        with pytest.raises(ValueError, match=message):
            installer.verify_sources(executor(fake), files)


class InstallServer:
    """A GSQL server fake: SHOW QUERY, endpoint listing, CREATE OR REPLACE and install.

    `mode` decides how GET /gsql/v1/queries/install answers: "sync" (TigerGraph
    4.2.5: the reply arrives when compilation is done), "timeout" (the client read
    times out; queries become enabled after `ready_after` endpoint listings) or
    "async" (a requestId polled through getQueryInstallationStatus).
    """

    def __init__(
        self, *, stale: tuple[str, ...] = (), mode: str = "sync", ready_after: int = 0
    ) -> None:
        self.queries = gsql_text.repository_queries(INSTALL_FILES)
        self.shown = {name: text for name, (_, text) in self.queries.items()}
        for name in stale:  # an older definition is installed
            self.shown[name] = self.shown[name].replace("{", "{ INT stale_marker = 0;", 1)
        self.enabled = dict.fromkeys(self.queries, True)
        self.mode, self.ready_after = mode, ready_after
        self.created: list[str] = []
        self.installs: list[tuple[list[str], bool]] = []
        self.listings = 0
        self.statuses: list[dict[str, Any]] = []
        self.pending: list[str] = []

    def getSchema(self, force: bool) -> dict[str, Any]:
        return {"VertexTypes": [{"Name": "Temporal_Training_Scope"}]}

    def gsql(self, text: str) -> str:
        if "SHOW QUERY" in text:
            return self.shown.get(text.rsplit(" ", 1)[1], "Query not found")
        self.created.append(text)
        names = list(gsql_text.definitions(text))
        for name in names:
            self.shown[name] = gsql_text.definitions(text)[name]
            self.enabled[name] = False  # CREATE OR REPLACE disables the endpoint
        return f"Successfully created queries: [{', '.join(names)}]."

    def _enable(self) -> None:
        for name in self.pending:
            self.enabled[name] = True

    def installQueries(self, names: list[str], wait: bool) -> dict[str, Any]:
        self.installs.append((list(names), wait))
        self.pending = list(names)
        if self.mode == "timeout":
            raise requests.ReadTimeout("no reply while the server compiles")
        if self.mode == "async":
            return {"requestId": "r1"}
        self._enable()
        return {"error": False, "message": "Query installation finished: SUCCESS"}

    def getQueryInstallationStatus(self, request: str) -> dict[str, Any]:
        status = self.statuses.pop(0)
        if "SUCCESS" in status["message"]:
            self._enable()
        return status

    def getInstalledQueries(self) -> dict[str, Any]:
        self.listings += 1
        if self.pending and self.mode == "timeout" and self.listings > self.ready_after:
            self._enable()
        return {
            f"GET /query/{GRAPH_NAME}/{name}": endpoint(
                gsql_text.parameter_names(text), self.enabled[name]
            )
            for name, (_, text) in self.queries.items()
        }


def test_install_creates_and_installs_only_stale_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "TRAINING_QUERY_FILES", INSTALL_FILES)
    names = list(gsql_text.repository_queries(INSTALL_FILES))
    assert names == [REVEAL_UNIFORMS_QUERY, REVEAL_QUERY, CUTOFF_QUERY]
    # Everything current: nothing is created or installed.
    server = InstallServer()
    logs = installer.install(executor(server))
    assert logs["installed"] == [] and logs["verified"] == names
    assert not server.created and not server.installs
    # A stale subquery is reinstalled together with its caller, nothing else.
    server = InstallServer(stale=(REVEAL_UNIFORMS_QUERY,))
    logs = installer.install(executor(server))
    assert logs["installed"] == [REVEAL_UNIFORMS_QUERY, REVEAL_QUERY]
    assert logs["up_to_date"] == [CUTOFF_QUERY]
    assert server.installs == [([REVEAL_UNIFORMS_QUERY, REVEAL_QUERY], False)]
    assert len(server.created) == 1 and CUTOFF_QUERY not in server.created[0]
    assert server.created[0].startswith(f"USE GRAPH {GRAPH_NAME}\n")
    assert logs["verified"] == names
    # A disabled endpoint is stale even when the text matches.
    server = InstallServer()
    server.enabled[CUTOFF_QUERY] = False
    assert installer.install(executor(server))["installed"] == [CUTOFF_QUERY]
    # Callers are found in the repository queries too.
    queries = gsql_text.repository_queries(QUERY_FILES)
    assert CONTEXT_QUERY in installer._with_callers({FOURIER_QUERY}, queries)
    assert installer._with_callers({HUB_QUERY}, queries) == {HUB_QUERY}


def test_install_polls_endpoints_when_the_install_request_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer, "TRAINING_QUERY_FILES", INSTALL_FILES)
    # Listing 1 finds the stale query; listings 2 and 3 still see it compiling.
    server = InstallServer(stale=(CUTOFF_QUERY,), mode="timeout", ready_after=3)
    tg = executor(server)
    logs = installer.install(tg, sleep=tg.clock.sleep, clock=tg.clock.time, poll_s=30)
    assert logs["installed"] == [CUTOFF_QUERY] and logs["install"] is None
    assert tg.sleeps == [30, 30] and all(server.enabled.values())
    # The install request waited up to the deadline for its answer.
    assert tg.client.timeouts == [installer.INSTALL_DEADLINE_S]
    # Still compiling at the deadline: an actionable timeout, and a later run installs
    # only what is still stale.
    server = InstallServer(stale=(CUTOFF_QUERY,), mode="timeout", ready_after=99)
    tg = executor(server)
    with pytest.raises(TimeoutError, match="still not installed.*re-run `mule install`"):
        installer.install(tg, sleep=tg.clock.sleep, clock=tg.clock.time, poll_s=30, deadline_s=100)
    # Other failures of the install request propagate.
    server = InstallServer(stale=(CUTOFF_QUERY,))
    server.installQueries = lambda names, wait: (_ for _ in ()).throw(KeyError("bad"))
    with pytest.raises(KeyError):
        installer.install(executor(server))


def test_install_follows_an_asynchronous_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "TRAINING_QUERY_FILES", INSTALL_FILES)
    server = InstallServer(stale=(CUTOFF_QUERY,), mode="async")
    server.statuses = [{"message": "RUNNING"}, {"message": "Query installation SUCCESS"}]
    sleeps: list[float] = []
    logs = installer.install(executor(server), sleep=sleeps.append, poll_s=5)
    assert logs["install"]["message"].endswith("SUCCESS") and sleeps == [5, 5]
    server = InstallServer(stale=(CUTOFF_QUERY,), mode="async")
    server.statuses = [{"message": "FAILED: type check"}]
    with pytest.raises(RuntimeError, match="failed"):
        installer.install(executor(server), sleep=sleeps.append)
    server = InstallServer(stale=(CUTOFF_QUERY,), mode="async")
    server.statuses = [{"message": "RUNNING"}] * 5
    clock = iter([0.0, 10.0, 99999.0])
    with pytest.raises(TimeoutError, match="still running"):
        installer.install(
            executor(server), sleep=sleeps.append, deadline_s=60, clock=lambda: next(clock)
        )
    server = InstallServer(stale=(CUTOFF_QUERY,))
    server.gsql = lambda text: (
        "Semantic Check Error" if "SHOW QUERY" not in text else "Query not found"
    )
    with pytest.raises(RuntimeError, match="Semantic Check"):
        installer.install(executor(server))


def test_install_writes_run_once_through_the_executor() -> None:
    # A write that fails is not repeated behind the caller's back, even when its failure
    # would clear: the one attempt's error names the operation.
    server = InstallServer(stale=(CUTOFF_QUERY,))
    real = server.gsql
    writes: list[str] = []

    def flaky(text: str) -> str:
        if "SHOW QUERY" in text:
            return real(text)
        writes.append(text)
        raise requests.ConnectionError("connection reset")

    server.gsql = flaky
    tg = executor(server)
    with pytest.raises(TransientQueryError, match=r"CREATE QUERY .* 1 attempt"):
        installer.install(tg)
    assert len(writes) == 1 and tg.sleeps == [] and not server.installs


def installed_repository() -> dict[str, str]:
    """The text of every training query, as a current graph has it installed."""
    return {
        name: text for name, (_, text) in gsql_text.repository_queries(TRAINING_QUERY_FILES).items()
    }


def drops(graph: FakeTigerGraph) -> list[str]:
    return [write.rsplit(" ", 1)[1] for write in graph.writes if "DROP QUERY" in write]


def test_install_leaves_the_retired_queries_installed() -> None:
    # A graph as the code before the rename left it: every old name installed and none of
    # the new ones. Code of that time may still run, so only `mule install` drops them.
    graph = FakeTigerGraph(queries={name: retired_query(name) for name in RETIRED_QUERIES})
    logs = installer.install(graph)
    assert logs["installed"] == logs["verified"] == list(installed_repository())
    assert "dropped" not in logs and drops(graph) == []
    assert installer.retired_installed(graph) == list(RETIRED_QUERIES)


def test_drop_retired_drops_the_retired_queries_callers_first_and_nothing_else() -> None:
    # Every renamed query installed beside every old name, and a query that is neither
    # the repository's nor retired.
    old = {name: retired_query(name) for name in RETIRED_QUERIES}
    graph = FakeTigerGraph(
        queries={**installed_repository(), **old, "match_parties": retired_query("match_parties")}
    )
    assert installer.drop_retired(graph) == drops(graph) == list(RETIRED_QUERIES)
    assert all(write.startswith(f"USE GRAPH {GRAPH_NAME}\n") for write in graph.writes)
    names = list(installed_repository())
    assert set(installer.installed_endpoints(graph)) == {*names, "match_parties"}
    assert installer.undefined_queries(graph) == ["match_parties"]
    # A second run finds nothing to drop.
    graph.writes.clear()
    assert installer.drop_retired(graph) == [] and graph.writes == []


def test_only_the_retired_queries_still_installed_are_dropped_in_their_order() -> None:
    left = ("temporal_training_population", "temporal_fourier64", "temporal_fourier64_values")
    graph = FakeTigerGraph(
        queries={**installed_repository(), **{n: retired_query(n) for n in left}}
    )
    assert installer.drop_retired(graph) == drops(graph)
    assert drops(graph) == [name for name in RETIRED_QUERIES if name in left]
    assert installer.retired_installed(graph) == []


def test_a_drop_tigergraph_refuses_stops_the_drops() -> None:
    # A query that is not retired calls a retired one, so TigerGraph refuses that drop.
    graph = FakeTigerGraph(
        queries={
            **installed_repository(),
            "temporal_fourier64_values": retired_query("temporal_fourier64_values"),
            "match_parties": retired_query("match_parties", calls="temporal_fourier64_values"),
        }
    )
    with pytest.raises(RuntimeError, match="left it installed.*match_parties call it"):
        installer.drop_retired(graph)


def test_the_retired_queries_are_listed_before_the_queries_they_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for caller, callee in RETIRED_CALLS.items():
        assert RETIRED_QUERIES.index(caller) < RETIRED_QUERIES.index(callee)
    # TigerGraph refuses to drop a query another installed query calls, and so does the
    # fake graph: the list in another order would stop at its first callee.
    graph = FakeTigerGraph(queries={name: retired_query(name) for name in RETIRED_QUERIES})
    monkeypatch.setattr(installer, "RETIRED_QUERIES", tuple(reversed(RETIRED_QUERIES)))
    with pytest.raises(RuntimeError, match="DROP QUERY temporal_reveal_uniforms left it"):
        installer.drop_retired(graph)


def test_the_analytics_queries_are_installed_only_when_asked_for() -> None:
    # A graph with every training query installed and no analytics query.
    graph = FakeTigerGraph(queries=installed_repository())
    analytics = list(gsql_text.repository_queries(ANALYTICS_QUERY_FILES))
    logs = installer.install(graph)
    assert logs["installed"] == [] and graph.writes == []
    assert not set(analytics) & set(installer.installed_endpoints(graph))
    # With analytics, exactly those are created and installed, and the training queries
    # stay as they were; a second install finds them in place.
    logs = installer.install(graph, analytics=True)
    assert logs["installed"] == analytics
    assert [w for w in graph.writes if w.startswith("INSTALL")] == [
        "INSTALL QUERY " + ", ".join(analytics)
    ]
    graph.writes.clear()
    assert installer.install(graph, analytics=True)["installed"] == [] and graph.writes == []
