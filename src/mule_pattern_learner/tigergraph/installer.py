"""Install and verify the query definitions the training pipeline uses; drop the retired.

Every write (the scope schema change, CREATE, the install request and each DROP) runs
through the executor with one attempt, so it is never repeated behind the caller's back.
"""

from __future__ import annotations

from collections.abc import Callable
import re
import time
from typing import Any

from ..contract.server import (
    ANALYTICS_QUERY_FILES,
    GRAPH_NAME,
    RETIRED_QUERIES,
    SCOPE_VERTEX,
    TRAINING_QUERY_FILES,
)
from ..paths import GSQL_DIR
from ..runtime.progress import emit
from .executor import (
    AVAILABILITY,
    SERVER_TIMEOUT,
    ConnectionExecutor,
    TransientQueryError,
    failure_class,
)
from .gsql_text import calls, definitions, normalized, parameter_names, repository_queries

BUILTIN_ENDPOINT_PARAMETERS = frozenset({"query", "read_committed"})
# How long an install waits for compilation: well over the about 50 minutes that
# compiling every query of the context query's size took on the reference graph.
INSTALL_DEADLINE_S = 90 * 60.0
# The problems of query_problems that only a new CREATE resolves.
MISSING = "is missing on the server"
DIFFERS = "differs from repository source"


def _show_query(executor: ConnectionExecutor, name: str) -> str:
    return str(
        executor.gsql(f"USE GRAPH {GRAPH_NAME}\nSHOW QUERY {name}", what="SHOW QUERY " + name)
    )


def undefined_queries(executor: ConnectionExecutor) -> list[str]:
    """Installed queries that no GSQL file of the repository defines (read-only).

    `mule install` lists them after dropping the retired ones (drop_retired): what is
    left belongs to someone else, and nothing drops it.
    """
    defined = repository_queries((*TRAINING_QUERY_FILES, *ANALYTICS_QUERY_FILES))
    return sorted(set(installed_endpoints(executor)) - set(defined))


def installed_endpoints(executor: ConnectionExecutor) -> dict[str, dict[str, Any]]:
    """Installed-query endpoint metadata by query name (includes `enabled`)."""
    raw = executor.call(lambda conn: conn.getInstalledQueries(), what="getInstalledQueries")
    if not isinstance(raw, dict):
        raise ValueError("TigerGraph did not return installed query endpoints")
    prefix = f"GET /query/{GRAPH_NAME}/"
    return {
        endpoint[len(prefix) :]: info
        for endpoint, info in raw.items()
        if endpoint.startswith(prefix) and isinstance(info, dict)
    }


def query_problems(
    executor: ConnectionExecutor, files: tuple[str, ...] = TRAINING_QUERY_FILES
) -> dict[str, list[str]]:
    """Per query: why the server copy is not the installed repository query (empty if it is).

    SHOW QUERY also returns created but uninstalled text, so the REST endpoint is
    checked separately, including its parameter names.
    """
    endpoints = installed_endpoints(executor)
    problems: dict[str, list[str]] = {}
    for name, (_, source) in repository_queries(files).items():
        actual = definitions(_show_query(executor, name)).get(name)
        if actual is None:
            problems[name] = [MISSING]
            continue
        issues = []
        if normalized(actual) != normalized(source):
            issues.append(DIFFERS)
        endpoint = endpoints.get(name)
        if endpoint is None or endpoint.get("enabled") is not True:
            issues.append("is not installed (REST endpoint disabled)")
        else:
            served = set(endpoint.get("parameters", {})) - BUILTIN_ENDPOINT_PARAMETERS
            if served != parameter_names(source):
                issues.append("endpoint parameters differ from repository source")
        if issues:
            problems[name] = issues
    return problems


def verify_sources(
    executor: ConnectionExecutor, files: tuple[str, ...] = TRAINING_QUERY_FILES
) -> list[str]:
    """Every training query must match the repository text and be installed and enabled."""
    problems = query_problems(executor, files)
    if problems:
        raise ValueError(
            "Installed query differs from repository source or is not installed: "
            + "; ".join(f"{name} {issue}" for name, issues in problems.items() for issue in issues)
            + ". Run `mule install`."
        )
    return list(repository_queries(files))


def has_scope_vertex(executor: ConnectionExecutor) -> bool:
    """Whether the graph schema has the scope vertex type (read-only)."""
    schema = executor.call(lambda conn: conn.getSchema(force=True), what="getSchema")
    return SCOPE_VERTEX in {v["Name"] for v in schema["VertexTypes"]}


def _installation_state(status: Any) -> str:
    if not isinstance(status, dict):
        return "running"
    message = str(status.get("message", ""))
    if status.get("error") in (True, "true") or "FAILED" in message.upper():
        return "failed"
    return "success" if "SUCCESS" in message.upper() else "running"


def _with_callers(stale: set[str], queries: dict[str, tuple[str, str]]) -> set[str]:
    """Add every repository query that calls a stale query (subqueries are linked in)."""
    result = set(stale)
    changed = True
    while changed:
        changed = False
        for name, (_, text) in queries.items():
            if name not in result and any(calls(text, callee) for callee in result):
                result.add(name)
                changed = True
    return result


def _created(output: str) -> bool:
    return "Successfully created queries" in output and not re.search(
        r"(?:[1-9]\d* syntax error|(?:Type Check|Semantic Check|Syntax) Error|draft query)",
        output,
        re.I,
    )


def retired_installed(executor: ConnectionExecutor) -> list[str]:
    """The RETIRED_QUERIES still installed, in their order, callers first (read-only)."""
    installed = installed_endpoints(executor)
    return [name for name in RETIRED_QUERIES if name in installed]


def drop_retired(executor: ConnectionExecutor) -> list[str]:
    """Drop every installed query of RETIRED_QUERIES, callers first; the names dropped.

    The names are the ones the queries were installed under before they were renamed
    (the owner's decision in docs/architecture.md), which code from before the rename
    still calls, so only `mule install` drops them, after its install has passed. A name
    that is not installed is skipped, and no other query is ever touched. TigerGraph
    refuses to drop a query another installed query calls, so each drop is checked
    against the endpoint listing, and one that leaves its query installed raises with
    TigerGraph's answer.
    """
    names = retired_installed(executor)
    for name in names:
        output = executor.gsql(
            f"USE GRAPH {GRAPH_NAME}\nDROP QUERY {name}", what="DROP QUERY " + name, attempts=1
        )
        if name in installed_endpoints(executor):
            raise RuntimeError(f"DROP QUERY {name} left it installed: {output}")
    emit({"event": "drop_retired", "dropped": names})
    return names


def _unanswered(error: Exception) -> bool:
    """Whether the install request failed because the client gave up waiting.

    The executor wraps the client's error after its one attempt; the client's own error
    (its cause) says whether the server may still be compiling.
    """
    cause = error.__cause__ if isinstance(error, TransientQueryError) else error
    return failure_class(cause or error) in (AVAILABILITY, SERVER_TIMEOUT)


def install(
    executor: ConnectionExecutor,
    *,
    analytics: bool = False,
    deadline_s: float = INSTALL_DEADLINE_S,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Create and install only the stale queries; the retired ones stay (drop_retired).

    A query is stale when SHOW QUERY differs from the repository, its endpoint is
    missing or disabled, or its endpoint parameters differ (see query_problems);
    ``analytics`` also installs the analytics queries. Queries that call a stale query
    (the context query calls the Fourier query) are installed with it. Only the
    definitions whose text is missing or differs, and the queries that call them, are
    created again, because CREATE OR REPLACE disables an installed endpoint until the
    query is installed again. A query whose text is current but whose endpoint is not (a
    compilation that has not finished or failed) is only installed, so a run after a
    timeout sends no CREATE for what the last one created.

    TigerGraph 4.2.5 answers GET /gsql/v1/queries/install only when compilation
    finishes and returns no requestId, so that request gets a read timeout of
    `deadline_s`. When the client gives up first (read timeout, dropped
    connection, gateway error), the endpoint listing is polled every `poll_s`
    seconds until every installed query is enabled or the deadline passes. A
    requestId, when a server returns one, is polled with getQueryInstallationStatus.
    Success is decided by verify_sources, not by a status message.
    """
    logs: dict[str, Any] = {}
    if not has_scope_vertex(executor):
        migration = GSQL_DIR / "schema/scope_vertex.gsql"
        result = executor.gsql(migration.read_text(), what="scope schema change", attempts=1)
        if "Local schema change succeeded" not in result:
            raise RuntimeError(result)
        logs["scope_schema"] = result
    files = (*TRAINING_QUERY_FILES, *(ANALYTICS_QUERY_FILES if analytics else ()))
    queries = repository_queries(files)
    problems = query_problems(executor, files)
    stale = _with_callers(set(problems), queries)
    names = [name for name in queries if name in stale]
    texts = {name for name, issues in problems.items() if {MISSING, DIFFERS} & set(issues)}
    created = _with_callers(texts, queries)
    logs["installed"] = names
    logs["up_to_date"] = [name for name in queries if name not in stale]
    emit({"event": "install", "stale": names, "up_to_date": logs["up_to_date"]})
    if not names:
        logs["verified"] = verify_sources(executor, files)
        return logs
    for relative in files:
        chosen = [
            queries[name][1] for name in names if name in created and queries[name][0] == relative
        ]
        if not chosen:
            continue
        output = executor.gsql(
            f"USE GRAPH {GRAPH_NAME}\n" + "\n\n".join(chosen) + "\n",
            what="CREATE QUERY " + relative,
            attempts=1,
        )
        if not _created(output):
            raise RuntimeError(output)
        logs[relative] = output
    started = clock()
    status: Any = None
    try:
        with executor.client.request_timeout(read_s=deadline_s):
            status = executor.call(
                lambda conn: conn.installQueries(names, wait=False),
                what="installQueries",
                attempts=1,
            )
    except Exception as error:
        if not _unanswered(error):
            raise
        emit(
            {
                "event": "install_unanswered",
                "error": type(error).__name__,
                "note": "polling the endpoint listing",
            }
        )
    request_id = status.get("requestId") if isinstance(status, dict) else None
    while request_id and _installation_state(status) == "running":
        elapsed = clock() - started
        if elapsed > deadline_s:
            raise TimeoutError(
                f"Query installation {request_id} still running after {elapsed:.0f}s; "
                "check it with getQueryInstallationStatus before retrying"
            )
        emit(
            {
                "event": "install_wait",
                "installing": len(names),
                "request_id": request_id,
                "elapsed_s": round(elapsed),
            }
        )
        sleep(poll_s)
        status = executor.call(
            lambda conn: conn.getQueryInstallationStatus(str(request_id)),
            what="getQueryInstallationStatus",
        )
    state = _installation_state(status)
    if state == "failed":
        raise RuntimeError(f"Query installation failed: {status}")
    if state != "success":
        _await_enabled(executor, names, started, deadline_s, poll_s, sleep, clock)
    logs["install"] = status
    logs["verified"] = verify_sources(executor, files)
    return logs


def _await_enabled(
    executor: ConnectionExecutor,
    names: list[str],
    started: float,
    deadline_s: float,
    poll_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Poll the endpoint listing until every named query is installed and enabled."""
    while True:
        endpoints = installed_endpoints(executor)
        pending = [name for name in names if endpoints.get(name, {}).get("enabled") is not True]
        elapsed = clock() - started
        if not pending:
            return
        if elapsed > deadline_s:
            raise TimeoutError(
                f"Queries {pending} are still not installed after {elapsed:.0f}s. The server "
                "may still be compiling: once it has finished, run the same command again, "
                "which installs only what is still stale."
            )
        emit({"event": "install_wait", "awaiting": pending, "elapsed_s": round(elapsed)})
        sleep(poll_s)
