"""The installer, on a fake server that keeps query texts and endpoints like TigerGraph, and
on three small queries in a temporary gsql/ folder, so the tests do not depend on which
queries the repository holds."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import requests

from mule_pattern_learner.paths import parity_record
from mule_pattern_learner.tigergraph import installer
from mule_pattern_learner.tigergraph.installer import (
    definitions,
    install,
    normalized,
    parameter_names,
    parity_versions,
    repository_queries,
    require_parity,
    verify,
)
from mule_pattern_learner.tigergraph.queries import GRAPH_NAME

LIST_ITEMS = "list_items"
READ_ITEMS = "read_items"
READ_ANSWERS = "read_answers"
# Each query of the temporary folder: its file under gsql/ and its text.
FILES = {
    LIST_ITEMS: (
        "queries/list_items.gsql",
        """USE GRAPH Mule_Pattern_Learner

/* Lists accounts a page at a time. */
CREATE OR REPLACE QUERY list_items(
    UINT t_ms,
    INT page = 10000
) FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {
    IF t_ms == 0 OR page < 1 THEN
        PRINT "invalid_parameters" AS status;
        RETURN;
    END;
    PRINT "ok" AS status;
}
""",
    ),
    READ_ITEMS: (
        "queries/read_items.gsql",
        """USE GRAPH Mule_Pattern_Learner

CREATE OR REPLACE QUERY read_items(
    UINT t_ms,
    SET<VERTEX<Account>> hubs
) FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {
    PRINT "ok" AS status, hubs.size() AS hubs;
}
""",
    ),
    READ_ANSWERS: (
        "evaluation/read_answers.gsql",
        """USE GRAPH Mule_Pattern_Learner

CREATE OR REPLACE QUERY read_answers(STRING after_id) FOR GRAPH Mule_Pattern_Learner {
    PRINT "ok" AS status;
}
""",
    ),
}
NAMES = list(FILES)


@pytest.fixture(autouse=True)
def query_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A gsql/ folder holding the three queries, installed as the repository's."""
    for relative, text in FILES.values():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    monkeypatch.setattr(installer, "GSQL_DIR", tmp_path)
    monkeypatch.setattr(
        installer, "QUERY_FILES", {name: relative for name, (relative, _) in FILES.items()}
    )
    return tmp_path


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
    queries = repository_queries()
    texts = {name: f"CREATE {' '.join(text.split())}" for name, text in queries.items()}
    return FakeServer(texts, set(queries))


def test_the_repository_queries_are_read_from_their_files() -> None:
    queries = repository_queries()
    assert list(queries) == NAMES
    assert queries[READ_ANSWERS] == (
        "QUERY read_answers(STRING after_id) FOR GRAPH Mule_Pattern_Learner {\n"
        '    PRINT "ok" AS status;\n}'
    )
    assert parameter_names(queries[LIST_ITEMS]) == ["t_ms", "page"]
    assert parameter_names(queries[READ_ITEMS]) == ["t_ms", "hubs"]


def test_a_registered_query_whose_file_is_missing_is_named(query_folder: Path) -> None:
    (query_folder / FILES[READ_ITEMS][0]).unlink()
    with pytest.raises(FileNotFoundError, match="the query read_items is registered but not"):
        repository_queries()


def test_a_file_must_define_its_query_and_no_other(query_folder: Path) -> None:
    relative, text = FILES[READ_ITEMS]
    (query_folder / relative).write_text(text.replace("read_items(", "read_other("))
    with pytest.raises(ValueError, match="must define the query read_items and no other"):
        repository_queries()
    (query_folder / relative).write_text(text + FILES[READ_ANSWERS][1])
    with pytest.raises(ValueError, match="must define the query read_items and no other"):
        repository_queries()


def test_a_fresh_graph_gets_every_query_created_and_installed() -> None:
    server = FakeServer()
    executor = FakeExecutor(server)
    assert install(executor) == NAMES
    assert server.created == [NAMES] and server.installs == [NAMES]
    verify(executor)


def test_current_queries_are_left_alone() -> None:
    server = _current()
    assert install(FakeExecutor(server)) == []
    assert server.created == [] and server.installs == []


def test_only_a_changed_query_is_created_again() -> None:
    server = _current()
    server.texts[LIST_ITEMS] = server.texts[LIST_ITEMS].replace("page < 1", "page < 2")
    assert install(FakeExecutor(server)) == [LIST_ITEMS]
    assert server.created == [[LIST_ITEMS]]
    assert definitions(server.texts[LIST_ITEMS]) == {LIST_ITEMS: repository_queries()[LIST_ITEMS]}


def test_a_query_whose_compilation_did_not_finish_is_installed_without_a_create() -> None:
    server = _current()
    server.enabled.discard(READ_ITEMS)
    assert install(FakeExecutor(server)) == [READ_ITEMS]
    assert server.created == [] and server.installs == [[READ_ITEMS]]


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
    server.texts[LIST_ITEMS] = server.texts[LIST_ITEMS].replace("10000", "20000")
    server.extra_parameters[READ_ITEMS] = ["cutoff"]
    del server.texts[READ_ANSWERS]
    with pytest.raises(ValueError) as refused:
        verify(FakeExecutor(server))
    message = str(refused.value)
    assert "list_items differs from the repository" in message
    assert "read_items has other parameters" in message
    assert "read_answers is missing on the server" in message


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


def test_a_comment_in_the_parameter_list_is_no_parameter() -> None:
    text = (
        "CREATE OR REPLACE QUERY q(\n"
        "    UINT t_ms, // the scoring time: a whole second, after (90) days\n"
        "    LIST<STRING> account_ids /* the accounts, each once */,\n"
        '    STRING note = "a // b, c" // a default that holds a slash\n'
        ") {\n}\n"
    )
    assert parameter_names(definitions(text)["q"]) == ["t_ms", "account_ids", "note"]


# One record of gsql/parity.toml: a query version that passed parity.
PASSED = """
[[passed]]
query = "list_accounts"
version = "list_accounts-0123456789ab"
tigergraph = "4.2.5"
world = "0123456789abcdef0123456789abcdef"
date = 2026-10-17
"""


def test_the_parity_record_holds_the_versions_that_passed(tmp_path: Path) -> None:
    record = tmp_path / "parity.toml"
    record.write_text(PASSED + PASSED.replace("list_accounts", "read_labels"))
    assert parity_versions(record) == {
        ("list_accounts", "list_accounts-0123456789ab"),
        ("read_labels", "read_labels-0123456789ab"),
    }
    require_parity({"list_accounts": "list_accounts-0123456789ab"}, record)
    # Every version without a record is named: one of another text, one never run.
    versions = {
        "list_accounts": "list_accounts-ba9876543210",
        "read_labels": "read_labels-0123456789ab",
        "read_hubs": "read_hubs-0123456789ab",
    }
    with pytest.raises(ValueError) as refused:
        require_parity(versions, record)
    assert str(refused.value).startswith(
        "No parity record for list_accounts list_accounts-ba9876543210, "
        "read_hubs read_hubs-0123456789ab in "
    )
    assert "run `mule parity`" in str(refused.value)


def test_without_a_record_no_version_has_passed(tmp_path: Path) -> None:
    missing = tmp_path / "parity.toml"
    assert parity_versions(missing) == set()
    with pytest.raises(ValueError, match="No parity record for list_accounts"):
        require_parity({"list_accounts": "list_accounts-0123456789ab"}, missing)
    require_parity({}, missing)
    # The record is committed with the queries, as TOML since git ignores JSON.
    assert parity_record().name == "parity.toml" and parity_record().parent.name == "gsql"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('tigergraph = "4.2.5"\n', "", "without exactly"),
        ('tigergraph = "4.2.5"', 'tigergraph = "4.2.5"\nsplit = 1', "without exactly"),
        ("date = 2026-10-17", 'date = "2026-10-17"', "date that is not a date"),
        ('"list_accounts-0123456789ab"', '"read_labels-0123456789ab"', "records the version"),
        ("[[passed]]", "[passed]", "must hold"),
    ],
)
def test_a_malformed_parity_record_is_refused(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    record = tmp_path / "parity.toml"
    record.write_text(PASSED.replace(old, new))
    with pytest.raises(ValueError, match=message):
        parity_versions(record)
