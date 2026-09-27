"""Install and verify only the query definitions the training pipeline uses."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import json
import re
import time
from typing import Any

from ..paths import REPOSITORY_ROOT
from .executor import AVAILABILITY, GRAPH, SERVER_TIMEOUT, connection_call, failure_class
from .gsql_text import definitions, normalized, parameter_names, repository_queries

# The queries preparation runs; a prepared dataset records their source hashes.
QUERY_FILES = (
    "gsql/queries/fourier64.gsql",
    "gsql/queries/training_context.gsql",
    "gsql/queries/training_scope.gsql",
    "gsql/queries/split_cutoffs.gsql",
    "gsql/queries/hub_accounts.gsql",
)
# Preparation queries plus the oracle export for audits, the label-contract validation
# and the one-time reveal job (the first run reveals known mules; see reveal.py).
TRAINING_QUERY_FILES = (
    *QUERY_FILES,
    "gsql/evaluation/ground_truth.gsql",
    "gsql/queries/label_contract.gsql",
    "gsql/queries/label_reveal.gsql",
)
# Analytics queries: parity tools for the persisted pair encodings. Training never calls
# them, so they are installed only when asked for (install --include-optional).
OPTIONAL_QUERY_FILES = (
    "gsql/analytics/zelle_pair_gaps.gsql",
    "gsql/analytics/payment_pair_gaps.gsql",
)
BUILTIN_ENDPOINT_PARAMETERS = frozenset({"query", "read_committed"})
INSTALL_DEADLINE_S = 45 * 60.0


def _show_query(executor: Any, name: str) -> str:
    text = f"USE GRAPH {GRAPH}\nSHOW QUERY {name}"
    gsql = getattr(executor, "gsql", None)
    if gsql is not None:
        return str(gsql(text, what="SHOW QUERY " + name))
    return str(executor.client.conn.gsql(text))


def installed_endpoints(executor: Any) -> dict[str, dict[str, Any]]:
    """Installed-query endpoint metadata by query name (includes `enabled`)."""
    raw = connection_call(executor, "getInstalledQueries", lambda conn: conn.getInstalledQueries())
    if not isinstance(raw, dict):
        raise ValueError("TigerGraph did not return installed query endpoints")
    prefix = f"GET /query/{GRAPH}/"
    return {
        endpoint[len(prefix) :]: info
        for endpoint, info in raw.items()
        if endpoint.startswith(prefix) and isinstance(info, dict)
    }


def query_problems(
    executor: Any, files: tuple[str, ...] = TRAINING_QUERY_FILES
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
            problems[name] = ["is missing on the server"]
            continue
        issues = []
        if normalized(actual) != normalized(source):
            issues.append("differs from repository source")
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


def verify_sources(executor: Any, files: tuple[str, ...] = TRAINING_QUERY_FILES) -> list[str]:
    """Every training query must match the repository text and be installed and enabled."""
    problems = query_problems(executor, files)
    if problems:
        raise ValueError(
            "Installed query differs from repository source or is not installed: "
            + "; ".join(f"{name} {issue}" for name, issues in problems.items() for issue in issues)
            + ". Run `mule-temporal install`."
        )
    return list(repository_queries(files))


def _installation_state(status: Any) -> str:
    if not isinstance(status, dict):
        return "running"
    message = str(status.get("message", ""))
    if status.get("error") in (True, "true") or "FAILED" in message.upper():
        return "failed"
    return "success" if "SUCCESS" in message.upper() else "running"


def _with_callers(stale: set[str], queries: dict[str, tuple[str, str]]) -> set[str]:
    """Add every repository query that calls a stale query (subqueries are linked in)."""
    bodies = {
        name: re.sub(r"/\*.*?\*/|//[^\n]*|#[^\n]*", "", text, flags=re.S)
        for name, (_, text) in queries.items()
    }
    result = set(stale)
    changed = True
    while changed:
        changed = False
        for name, body in bodies.items():
            if name not in result and any(
                re.search(rf"\b{re.escape(callee)}\s*\(", body) for callee in result
            ):
                result.add(name)
                changed = True
    return result


def _created(output: str) -> bool:
    return "Successfully created queries" in output and not re.search(
        r"(?:[1-9]\d* syntax error|(?:Type Check|Semantic Check|Syntax) Error|draft query)",
        output,
        re.I,
    )


def install(
    executor: Any,
    *,
    force: bool = False,
    include_optional: bool = False,
    deadline_s: float = INSTALL_DEADLINE_S,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Create and install only the training queries that are stale on the server.

    A query is stale when SHOW QUERY differs from the repository, its endpoint is
    missing or disabled, or its endpoint parameters differ (see query_problems);
    `force=True` treats every query as stale. Queries that call a stale query
    (temporal_training_context calls temporal_fourier64_values) are installed
    with it. Only stale definitions are re-created, because CREATE OR REPLACE
    disables an installed endpoint until the query is installed again.

    TigerGraph 4.2.5 answers GET /gsql/v1/queries/install only when compilation
    finishes and returns no requestId, so that request gets a read timeout of
    `deadline_s`. When the client gives up first (read timeout, dropped
    connection, gateway error), the endpoint listing is polled every `poll_s`
    seconds until every installed query is enabled or the deadline passes. A
    requestId, when a server returns one, is polled with getQueryInstallationStatus.
    Success is decided by verify_sources, not by a status message.
    """
    conn = executor.client.conn
    logs: dict[str, Any] = {}
    schema = conn.getSchema(force=True)
    if "Temporal_Training_Scope" not in {v["Name"] for v in schema["VertexTypes"]}:
        migration = REPOSITORY_ROOT / "gsql/schema/scope_vertex.gsql"
        result = str(conn.gsql(migration.read_text()))
        if "Local schema change succeeded" not in result:
            raise RuntimeError(result)
        logs["scope_schema"] = result
    files = (*TRAINING_QUERY_FILES, *(OPTIONAL_QUERY_FILES if include_optional else ()))
    queries = repository_queries(files)
    stale = set(queries) if force else _with_callers(set(query_problems(executor, files)), queries)
    names = [name for name in queries if name in stale]
    logs["installed"] = names
    logs["up_to_date"] = [name for name in queries if name not in stale]
    print(json.dumps({"install": names, "up_to_date": logs["up_to_date"]}), flush=True)
    if not names:
        logs["verified"] = verify_sources(executor, files)
        return logs
    for relative in files:
        chosen = [queries[name][1] for name in names if queries[name][0] == relative]
        if not chosen:
            continue
        output = str(conn.gsql(f"USE GRAPH {GRAPH}\n" + "\n\n".join(chosen) + "\n"))
        if not _created(output):
            raise RuntimeError(output)
        logs[relative] = output
    started = clock()
    status: Any = None
    timeout = getattr(executor.client, "request_timeout", None)
    try:
        with timeout(read_s=deadline_s) if timeout is not None else nullcontext():
            status = conn.installQueries(names, wait=False)
    except Exception as error:
        if failure_class(error) not in (AVAILABILITY, SERVER_TIMEOUT):
            raise
        print(
            json.dumps(
                {
                    "install_request": "no answer; polling the endpoint listing",
                    "error": type(error).__name__,
                }
            ),
            flush=True,
        )
    request_id = status.get("requestId") if isinstance(status, dict) else None
    while request_id and _installation_state(status) == "running":
        elapsed = clock() - started
        if elapsed > deadline_s:
            raise TimeoutError(
                f"Query installation {request_id} still running after {elapsed:.0f}s; "
                "check it with getQueryInstallationStatus before retrying"
            )
        print(
            json.dumps(
                {"installing": len(names), "request_id": request_id, "elapsed_s": round(elapsed)}
            ),
            flush=True,
        )
        sleep(poll_s)
        status = connection_call(
            executor,
            "getQueryInstallationStatus",
            lambda conn: conn.getQueryInstallationStatus(str(request_id)),
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
    executor: Any,
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
                "may still be compiling: re-run `mule-temporal install` later, which installs "
                "only what is still stale."
            )
        print(json.dumps({"awaiting": pending, "elapsed_s": round(elapsed)}), flush=True)
        sleep(poll_s)
