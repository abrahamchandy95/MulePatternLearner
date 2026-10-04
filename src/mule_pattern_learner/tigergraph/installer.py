"""Install and verify the query definitions the training pipeline uses; drop the retired.

The scope's types come first: a graph without them gets gsql/schema/scope_vertex.gsql,
and `mule install` replaces types that differ from it (replace_scope_types). Every write
(a schema change, CREATE, the install request and each DROP) runs through the executor
with one attempt, so it is never repeated behind the caller's back. The output of each
GSQL write is recorded in a `gsql` event.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from ..runtime.console import plural
from ..runtime.progress import emit
from .executor import (
    AVAILABILITY,
    SERVER_TIMEOUT,
    ConnectionExecutor,
    TransientQueryError,
    error_summary,
    failure_class,
)
from .gsql_text import calls, definitions, normalized, parameter_names, repository_queries
from .scope_types import (
    SCOPE_TYPES_FILE,
    ScopeTypes,
    declared_scope_types,
    drop_job,
    foreign_scope_edges,
    graph_scope_types,
    scope_type_differences,
    uses_scope_types,
)

BUILTIN_ENDPOINT_PARAMETERS = frozenset({"query", "read_committed"})
# How long an install waits for compilation: well over the about 50 minutes that
# compiling every query of the context query's size took on the reference graph.
INSTALL_DEADLINE_S = 90 * 60.0
# The problems of query_problems that only a new CREATE resolves.
MISSING = "is missing on the server"
DIFFERS = "differs from repository source"
# The states of a graph's scope types (scope_schema).
PRESENT, ABSENT, OUTDATED = "present", "missing", "outdated"
# What TigerGraph answers a schema change job of the graph that succeeded.
SCHEMA_CHANGED = "Local schema change succeeded"


def _write(executor: ConnectionExecutor, text: str, what: str) -> str:
    """Run a GSQL write once and record what TigerGraph answered, in a `gsql` event."""
    output = executor.gsql(text, what=what, attempts=1)
    emit({"event": "gsql", "operation": what, "output": output})
    return output


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


@dataclass(frozen=True)
class ScopeSchema:
    """A graph's scope types against gsql/schema/scope_vertex.gsql (scope_schema).

    ``state`` is "present" when they are the same, "missing" when the graph has neither
    type and "outdated" when they differ, as ``differences`` says in words. ``found``
    holds the graph's types, and ``foreign_edges`` the edge types the file does not
    declare that reach the scope vertex type.
    """

    state: str
    found: ScopeTypes | None
    differences: tuple[str, ...]
    foreign_edges: tuple[str, ...]


def scope_schema(executor: ConnectionExecutor) -> ScopeSchema:
    """The graph's scope types against gsql/schema/scope_vertex.gsql (read-only)."""
    schema = executor.call(lambda conn: conn.getSchema(force=True), what="getSchema")
    declared = declared_scope_types()
    found = graph_scope_types(schema, declared)
    differences = tuple(scope_type_differences(found, declared)) if found else ()
    state = ABSENT if found is None else OUTDATED if differences else PRESENT
    return ScopeSchema(state, found, differences, tuple(foreign_scope_edges(schema, declared)))


def scope_vertices(executor: ConnectionExecutor, schema: ScopeSchema) -> int:
    """How many scope vertices the graph holds (read-only); 0 without the vertex type."""
    if schema.found is None or not schema.found.vertex_attributes:
        return 0
    raw = executor.call(lambda conn: conn.getVertexCount("*", realtime=True), what="getVertexCount")
    if not isinstance(raw, dict) or SCOPE_VERTEX not in raw:
        raise ValueError(f"TigerGraph did not count the {SCOPE_VERTEX} vertices")
    return int(raw[SCOPE_VERTEX])


def outdated_scope(schema: ScopeSchema) -> str:
    """How a graph's outdated scope types differ, as the messages about them begin."""
    return f"The graph's scope types differ from gsql/{SCOPE_TYPES_FILE}: " + "; ".join(
        schema.differences
    )


def _add_scope_types(executor: ConnectionExecutor) -> str:
    """Apply gsql/schema/scope_vertex.gsql; what TigerGraph answered."""
    output = _write(executor, (GSQL_DIR / SCOPE_TYPES_FILE).read_text(), "scope schema change")
    if SCHEMA_CHANGED not in output:
        raise RuntimeError(output)
    return output


def _on_server(executor: ConnectionExecutor, name: str) -> str | None:
    """A query's text on the server (SHOW QUERY), installed or not; None without it."""
    return definitions(_show_query(executor, name)).get(name)


def _callers_first(names: list[str], texts: dict[str, str]) -> list[str]:
    """names in an order that drops each query before every one of them it calls."""
    ordered, left = [], list(names)
    while left:
        free = [name for name in left if not any(calls(texts[o], name) for o in left if o != name)]
        if not free:
            raise RuntimeError(f"The queries {left} call one another")
        ordered += free
        left = [name for name in left if name not in free]
    return ordered


def scope_dependents(
    executor: ConnectionExecutor, declared: ScopeTypes
) -> tuple[list[str], list[str]]:
    """The queries a replacement of the scope types drops, callers first, and those it may not.

    The first are the repository's queries (the training, evaluation and analytics
    files) on the server whose text there names a scope type, with the repository's
    queries on the server that call them. The second are the installed queries that no
    repository file defines and that name a scope type or call one of the first: a
    replacement would break them, and it never touches them (read-only).
    """
    repository = repository_queries((*TRAINING_QUERY_FILES, *ANALYTICS_QUERY_FILES))
    texts = {name: text for name in repository if (text := _on_server(executor, name)) is not None}
    using = {name for name, text in texts.items() if uses_scope_types(text, declared)}
    dependents = _with_callers(using, {name: ("", text) for name, text in texts.items()})
    others = []
    for name in sorted(set(installed_endpoints(executor)) - set(repository)):
        text = _on_server(executor, name) or ""
        if uses_scope_types(text, declared) or any(calls(text, d) for d in dependents):
            others.append(name)
    return _callers_first([name for name in texts if name in dependents], texts), others


def replace_scope_types(executor: ConnectionExecutor, schema: ScopeSchema) -> dict[str, Any]:
    """Replace a graph's outdated scope types with gsql/schema/scope_vertex.gsql's.

    It refuses, and changes nothing, while the graph holds a scope vertex (replacing the
    types would delete every scope), while an edge type that the file does not declare
    reaches the scope vertex type, or while an installed query that no repository file
    defines uses the types or calls a query that does. Otherwise it drops the
    repository's queries that use the types, callers first (scope_dependents), since
    TigerGraph drops no type a query uses; drops the graph's scope edge type and vertex
    type in a schema change job; applies scope_vertex.gsql; and checks that the graph's
    types are now the file's. It returns the differences it found and the queries it
    dropped, which install then installs again with every other training query.
    """
    found = schema.found
    if found is None:
        raise ValueError("The graph has no scope types to replace")
    outdated = outdated_scope(schema)
    scopes = scope_vertices(executor, schema)
    if scopes:
        held = plural(scopes, f"{SCOPE_VERTEX} vertex", f"{SCOPE_VERTEX} vertices")
        raise ValueError(
            f"{outdated}. The graph holds {held}, "
            "which replacing the types would delete, so nothing was changed. Clear the "
            "graph's data and load it again, as docs/how-to/set-up-a-graph.md describes, "
            "then run `mule install` again."
        )
    if schema.foreign_edges:
        raise ValueError(
            f"{outdated}. The edge types {', '.join(schema.foreign_edges)}, which no "
            f"repository file declares, reach {SCOPE_VERTEX}, so nothing was changed: "
            "drop them, then run `mule install` again."
        )
    declared = declared_scope_types()
    dependents, others = scope_dependents(executor, declared)
    if others:
        raise ValueError(
            f"{outdated}. The queries {', '.join(others)}, which no repository file defines, "
            "use the scope types or call a query that does, so nothing was changed: drop or "
            "change them, then run `mule install` again."
        )
    emit(
        {
            "event": "scope_types",
            "replacing": list(schema.differences),
            "dropping": dependents,
        }
    )
    for name in dependents:
        output = _write(
            executor, f"USE GRAPH {GRAPH_NAME}\nDROP QUERY {name}", "DROP QUERY " + name
        )
        if _on_server(executor, name) is not None:
            raise RuntimeError(f"DROP QUERY {name} left it on the server: {output}")
    output = _write(executor, drop_job(found), "scope schema drop")
    if SCHEMA_CHANGED not in output:
        raise RuntimeError(output)
    added = _add_scope_types(executor)
    after = scope_schema(executor)
    if after.state != PRESENT:
        raise RuntimeError(
            f"The scope types still differ from gsql/{SCOPE_TYPES_FILE} after they were "
            f"replaced: {'; '.join(after.differences) or 'they are missing'}"
        )
    emit({"event": "scope_types", "replaced": list(schema.differences), "dropped": dependents})
    return {"differences": list(schema.differences), "dropped": dependents, "output": added}


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
        output = _write(
            executor, f"USE GRAPH {GRAPH_NAME}\nDROP QUERY {name}", "DROP QUERY " + name
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
    replace_scope: bool = False,
    deadline_s: float = INSTALL_DEADLINE_S,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Create and install only the stale queries; the retired ones stay (drop_retired).

    The scope types come first (scope_schema). A graph without them gets
    gsql/schema/scope_vertex.gsql. One whose types differ from the file is refused
    unless ``replace_scope``, which only `mule install` sets: replace_scope_types then
    replaces them, and every query of the files is installed, whatever its text, with
    the -force flag, since TigerGraph otherwise skips a query it has installed.

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
    Success is decided by verify_sources, not by a status message; an `installed` event
    then names the queries installed and the seconds it took.
    """
    logs: dict[str, Any] = {}
    scope = scope_schema(executor)
    if scope.state == OUTDATED:
        if not replace_scope:
            raise ValueError(
                f"{outdated_scope(scope)}. Run `mule install`, which replaces them while the "
                "graph holds no scope vertex."
            )
        logs["scope_replaced"] = replace_scope_types(executor, scope)
    elif scope.state == ABSENT:
        logs["scope_schema"] = _add_scope_types(executor)
    every = "scope_replaced" in logs
    files = (*TRAINING_QUERY_FILES, *(ANALYTICS_QUERY_FILES if analytics else ()))
    queries = repository_queries(files)
    problems = query_problems(executor, files)
    stale = set(queries) if every else _with_callers(set(problems), queries)
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
        output = _write(
            executor,
            f"USE GRAPH {GRAPH_NAME}\n" + "\n\n".join(chosen) + "\n",
            "CREATE QUERY " + relative,
        )
        if not _created(output):
            raise RuntimeError(output)
        logs[relative] = output
    started = clock()
    status: Any = None
    try:
        with executor.client.request_timeout(read_s=deadline_s):
            flag = {"flag": "-force"} if every else {}
            status = executor.call(
                lambda conn: conn.installQueries(names, wait=False, **flag),
                what="installQueries",
                attempts=1,
            )
    except Exception as error:
        if not _unanswered(error):
            raise
        # The request had one attempt, so no retry recorded its error: this record does.
        emit(
            {
                "event": "install_unanswered",
                "error": type(error).__name__,
                "detail": error_summary(error),
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
    emit({"event": "installed", "installed": names, "seconds": round(clock() - started)})
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
