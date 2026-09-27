"""Installing and verifying only the stale training queries."""

# Tests inspect transport internals (in-flight map, cadence, sessions) on purpose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.testing.fake_connection import executor
from mule_pattern_learner.tigergraph import gsql_text, installer

INSTALL_FILES = ("gsql/features/temporal_fourier64.gsql", "gsql/temporal/training_cutoffs.gsql")


def endpoint(parameters: set[str], enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "parameters": {name: {} for name in parameters | {"query", "read_committed"}},
    }


def test_verify_sources_requires_matching_text_and_enabled_endpoints() -> None:
    files = ("gsql/temporal/training_cutoffs.gsql", "gsql/features/temporal_fourier64.gsql")
    expected: dict[str, str] = {}
    for path in files:
        expected.update(gsql_text.definitions((REPOSITORY_ROOT / path).read_text()))
    params = {name: gsql_text.parameter_names(text) for name, text in expected.items()}
    assert params["temporal_training_cutoffs"] == {"cutoff_times"}

    def conn(
        text_for: Callable[[str], str], endpoints: dict[str, dict[str, Any]]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            gsql=lambda text: text_for(text.rsplit(" ", 1)[1]),
            getInstalledQueries=lambda: {
                f"GET /query/Mule_Pattern_Learner/{name}": value
                for name, value in endpoints.items()
            },
        )

    good = {name: endpoint(value) for name, value in params.items()}
    assert set(installer.verify_sources(executor(conn(expected.__getitem__, good)), files)) == set(
        expected
    )
    disabled = {**good, "temporal_training_cutoffs": endpoint({"cutoff_times"}, False)}
    renamed = {**good, "temporal_fourier64": endpoint({"other"})}
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
            f"GET /query/Mule_Pattern_Learner/{name}": endpoint(
                gsql_text.parameter_names(text), self.enabled[name]
            )
            for name, (_, text) in self.queries.items()
        }


def test_install_creates_and_installs_only_stale_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "TRAINING_QUERY_FILES", INSTALL_FILES)
    names = list(gsql_text.repository_queries(INSTALL_FILES))
    assert names == ["temporal_fourier64_values", "temporal_fourier64", "temporal_training_cutoffs"]
    # Everything current: nothing is created or installed.
    server = InstallServer()
    logs = installer.install(executor(server))
    assert logs["installed"] == [] and logs["verified"] == names
    assert not server.created and not server.installs
    # A stale subquery is reinstalled together with its caller, nothing else.
    server = InstallServer(stale=("temporal_fourier64_values",))
    logs = installer.install(executor(server))
    assert logs["installed"] == ["temporal_fourier64_values", "temporal_fourier64"]
    assert logs["up_to_date"] == ["temporal_training_cutoffs"]
    assert server.installs == [(["temporal_fourier64_values", "temporal_fourier64"], False)]
    assert len(server.created) == 1 and "temporal_training_cutoffs" not in server.created[0]
    assert server.created[0].startswith("USE GRAPH Mule_Pattern_Learner\n")
    assert logs["verified"] == names
    # force reinstalls every query.
    server = InstallServer()
    logs = installer.install(executor(server), force=True)
    assert server.installs == [(names, False)] and len(server.created) == 2
    # A disabled endpoint is stale even when the text matches.
    server = InstallServer()
    server.enabled["temporal_training_cutoffs"] = False
    assert installer.install(executor(server))["installed"] == ["temporal_training_cutoffs"]
    # Callers are found in the repository queries too.
    queries = gsql_text.repository_queries(installer.QUERY_FILES)
    assert "temporal_training_context" in installer._with_callers(
        {"temporal_fourier64_values"}, queries
    )
    assert installer._with_callers({"temporal_hub_registry"}, queries) == {"temporal_hub_registry"}


def test_install_polls_endpoints_when_the_install_request_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(installer, "TRAINING_QUERY_FILES", INSTALL_FILES)
    # Listing 1 finds the stale query; listings 2 and 3 still see it compiling.
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="timeout", ready_after=3)
    tg = executor(server)
    logs = installer.install(tg, sleep=tg.clock.sleep, clock=tg.clock.time, poll_s=30)
    assert logs["installed"] == ["temporal_training_cutoffs"] and logs["install"] is None
    assert tg.sleeps == [30, 30] and all(server.enabled.values())
    # Still compiling at the deadline: an actionable timeout, and a later run installs
    # only what is still stale.
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="timeout", ready_after=99)
    tg = executor(server)
    with pytest.raises(TimeoutError, match="still not installed.*re-run `mule-temporal install`"):
        installer.install(tg, sleep=tg.clock.sleep, clock=tg.clock.time, poll_s=30, deadline_s=100)
    # Other failures of the install request propagate.
    server = InstallServer(stale=("temporal_training_cutoffs",))
    server.installQueries = lambda names, wait: (_ for _ in ()).throw(KeyError("bad"))
    with pytest.raises(KeyError):
        installer.install(executor(server))


def test_install_follows_an_asynchronous_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "TRAINING_QUERY_FILES", INSTALL_FILES)
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="async")
    server.statuses = [{"message": "RUNNING"}, {"message": "Query installation SUCCESS"}]
    sleeps: list[float] = []
    logs = installer.install(executor(server), sleep=sleeps.append, poll_s=5)
    assert logs["install"]["message"].endswith("SUCCESS") and sleeps == [5, 5]
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="async")
    server.statuses = [{"message": "FAILED: type check"}]
    with pytest.raises(RuntimeError, match="failed"):
        installer.install(executor(server), sleep=sleeps.append)
    server = InstallServer(stale=("temporal_training_cutoffs",), mode="async")
    server.statuses = [{"message": "RUNNING"}] * 5
    clock = iter([0.0, 10.0, 99999.0])
    with pytest.raises(TimeoutError, match="still running"):
        installer.install(
            executor(server), sleep=sleeps.append, deadline_s=60, clock=lambda: next(clock)
        )
    server = InstallServer(stale=("temporal_training_cutoffs",))
    server.gsql = lambda text: (
        "Semantic Check Error" if "SHOW QUERY" not in text else "Query not found"
    )
    with pytest.raises(RuntimeError, match="Semantic Check"):
        installer.install(executor(server))
