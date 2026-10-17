"""The installer, on a fake server that keeps query texts and endpoints like TigerGraph."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pytest
import requests

from mule_pattern_learner.tigergraph.installer import (
    definitions,
    install,
    normalized,
    parameter_names,
    repository_queries,
    verify,
)
from mule_pattern_learner.tigergraph.queries import FETCH_SUBGRAPHS, GRAPH_NAME, QUERY_FILES

QUERIES = repository_queries()


class FakeConn:
    """The connection operations the installer calls, on the server's catalogue."""

    def __init__(self, server: FakeServer) -> None:
        self.server = server

    def getInstalledQueries(self) -> dict[str, Any]:
        self.server.listings += 1
        if self.server.listings > self.server.compiling_until:
            self.server.enabled |= self.server.compiling
        return {
            f"GET /query/{GRAPH_NAME}/{name}": {
                "enabled": name in self.server.enabled,
                "parameters": {key: {} for key in self.server.parameters(name)},
            }
            for name in self.server.texts
        }

    def installQueries(self, names: list[str]) -> Any:
        self.server.installs.append(list(names))
        if self.server.install_answer is not None:
            return self.server.install_answer
        if self.server.compiling_until:
            self.server.compiling = set(names)
            raise requests.ReadTimeout("read timed out")
        self.server.enabled |= set(names)
        return {"message": "Query installation finished: SUCCESS"}


class FakeServer:
    """A graph's queries: each text as SHOW QUERY gives it, and the enabled endpoints."""

    def __init__(self, texts: dict[str, str] | None = None, enabled: set[str] | None = None):
        self.texts = dict(texts or {})
        self.enabled = set(enabled or set())
        self.created: list[list[str]] = []
        self.installs: list[list[str]] = []
        self.install_answer: Any = None
        self.create_answer = "Successfully created queries: [{}]."
        # The endpoint listings after which a compilation the client stopped waiting for ends.
        self.compiling_until = 0
        self.compiling: set[str] = set()
        self.listings = 0
        self.extra_parameters: dict[str, list[str]] = {}

    def parameters(self, name: str) -> list[str]:
        return [*parameter_names(self.texts[name]), "query", *self.extra_parameters.get(name, [])]


class FakeExecutor:
    """A ConnectionExecutor over the fake server: one attempt for everything."""

    graph_name = GRAPH_NAME

    def __init__(self, server: FakeServer) -> None:
        self.server = server
        self.conn = FakeConn(server)
        self.client = self
        self.timeouts: list[float] = []

    @contextmanager
    def request_timeout(self, read_s: float) -> Generator[None, None, None]:
        self.timeouts.append(read_s)
        yield

    def call[T](
        self, operation: Callable[[Any], T], *, what: str, attempts: int | None = None
    ) -> T:
        return operation(self.conn)

    def gsql(self, text: str, *, what: str = "gsql", attempts: int | None = None) -> str:
        if "SHOW QUERY" in text:
            name = text.split()[-1]
            return self.server.texts.get(name, f"Query {name} cannot be found.")
        assert attempts == 1, "a write runs once"
        made = definitions(text)
        self.server.created.append(list(made))
        if "Successfully" in self.server.create_answer:
            for name, body in made.items():
                self.server.texts[name] = f"CREATE {body}"
                self.server.enabled.discard(name)
        return self.server.create_answer.format(", ".join(made))

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        raise AssertionError("the installer runs no query")


def _current() -> FakeServer:
    """A server holding every repository query, installed, as SHOW QUERY reprints it."""
    texts = {name: f"CREATE {' '.join(text.split())}" for name, text in QUERIES.items()}
    return FakeServer(texts, set(QUERIES))


def test_a_fresh_graph_gets_every_query_created_and_installed() -> None:
    server = FakeServer()
    executor = FakeExecutor(server)
    assert install(executor) == list(QUERY_FILES)
    assert server.created == [list(QUERY_FILES)] and server.installs == [list(QUERY_FILES)]
    verify(executor)


def test_current_queries_are_left_alone() -> None:
    server = _current()
    assert install(FakeExecutor(server)) == []
    assert server.created == [] and server.installs == []


def test_only_a_changed_query_is_created_again() -> None:
    server = _current()
    server.texts[FETCH_SUBGRAPHS] = server.texts[FETCH_SUBGRAPHS].replace(
        "seed_payments < 1", "seed_payments < 2"
    )
    assert install(FakeExecutor(server)) == [FETCH_SUBGRAPHS]
    assert server.created == [[FETCH_SUBGRAPHS]]
    assert definitions(server.texts[FETCH_SUBGRAPHS]) == {FETCH_SUBGRAPHS: QUERIES[FETCH_SUBGRAPHS]}


def test_a_query_whose_compilation_did_not_finish_is_installed_without_a_create() -> None:
    server = _current()
    server.enabled.discard("write_split")
    assert install(FakeExecutor(server)) == ["write_split"]
    assert server.created == [] and server.installs == [["write_split"]]


def test_an_install_the_client_stops_waiting_for_is_awaited_on_the_endpoints() -> None:
    server = FakeServer()
    server.compiling_until = 3
    executor = FakeExecutor(server)
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    assert install(executor, deadline_s=600, poll_s=30, sleep=sleep, clock=lambda: now[0])
    assert executor.timeouts == [600] and sleeps == [30, 30]
    verify(executor)


def test_an_install_that_never_finishes_runs_out_of_time() -> None:
    server = FakeServer()
    server.compiling_until = 1_000
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(TimeoutError, match="still not installed"):
        install(FakeExecutor(server), deadline_s=100, poll_s=30, sleep=sleep, clock=lambda: now[0])


def test_a_failed_create_or_install_raises_with_what_tigergraph_said() -> None:
    server = FakeServer()
    server.create_answer = "Semantic Check Error: unknown attribute"
    with pytest.raises(RuntimeError, match="Semantic Check Error"):
        install(FakeExecutor(server))
    server = FakeServer()
    server.install_answer = {"error": True, "message": "Query installation FAILED"}
    with pytest.raises(RuntimeError, match="installation failed"):
        install(FakeExecutor(server))


def test_verify_names_every_query_that_is_not_current() -> None:
    server = _current()
    server.texts["list_population"] = server.texts["list_population"].replace("10000", "20000")
    server.extra_parameters["read_supervision"] = ["cutoff"]
    del server.texts["read_answer_key"]
    with pytest.raises(ValueError) as refused:
        verify(FakeExecutor(server))
    message = str(refused.value)
    assert "list_population differs from the repository" in message
    assert "read_supervision has other parameters" in message
    assert "read_answer_key is missing on the server" in message


def test_install_refuses_another_graph() -> None:
    executor = FakeExecutor(FakeServer())
    executor.graph_name = "Other"
    with pytest.raises(ValueError, match="Other"):
        install(executor)


def test_query_texts_compare_without_comments_layout_or_keyword_case() -> None:
    text = "USE GRAPH G\n/* note */\nCREATE OR REPLACE QUERY q(LIST<STRING> a, INT b = 2) {\n"
    text += '  PRINT "Ok" AS status; // done\n}\nUSE GRAPH G\n'
    assert definitions(text) == {"q": text[text.index("QUERY") : text.rindex("}") + 1]}
    assert parameter_names(definitions(text)["q"]) == ["a", "b"]
    same = 'create query q(list<string> a, int b = 2) { print "Ok" as status; }'
    assert normalized(definitions(same)["q"]) == normalized(definitions(text)["q"])
    other = same.replace('"Ok"', '"ok"')
    assert normalized(definitions(other)["q"]) != normalized(definitions(text)["q"])
