"""Install and verify only the query definitions used by temporal training.

verify_frozen_source also rechecks the live provenance of a prepared dataset (vertex
counts and, for strict runs, its scope; see scope.py) before a streamed run.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import json
import re
import time
from typing import Any

from mule_pattern_learner.configuration import REPOSITORY_ROOT

from .executor import AVAILABILITY, GRAPH, SERVER_TIMEOUT, connection_call, failure_class
from .scope import verify_scope

# The queries preparation runs; a prepared dataset records their source hashes.
QUERY_FILES = (
    "gsql/features/temporal_fourier64.gsql",
    "gsql/temporal/training_context.gsql",
    "gsql/temporal/training_population.gsql",
    "gsql/temporal/training_scope.gsql",
    "gsql/temporal/training_cutoffs.gsql",
    "gsql/temporal/hub_registry.gsql",
)
# Preparation queries plus the label contract: the oracle audit export/validation and the
# one-time reveal job (the first run reveals known mules; see labels.py).
TRAINING_QUERY_FILES = (
    *QUERY_FILES,
    "gsql/temporal/account_supervision.gsql",
    "gsql/temporal/label_reveal.gsql",
)
# Parity tools for the persisted pair encodings; training never calls them.
OPTIONAL_QUERY_FILES = (
    "gsql/features/zelle_pair_time64.gsql",
    "gsql/features/payment_pair_time64.gsql",
)
# Experiment metadata written by preparation itself; never part of source identity.
EXPERIMENT_METADATA_TYPES = frozenset({"Temporal_Training_Scope"})
BUILTIN_ENDPOINT_PARAMETERS = frozenset({"query", "read_committed"})
INSTALL_DEADLINE_S = 45 * 60.0


def _show_query(executor: Any, name: str) -> str:
    text = f"USE GRAPH {GRAPH}\nSHOW QUERY {name}"
    gsql = getattr(executor, "gsql", None)
    if gsql is not None:
        return str(gsql(text, what="SHOW QUERY " + name))
    return str(executor.client.conn.gsql(text))


def normalized(source: str) -> str:
    source = re.sub(r"/\*.*?\*/|//[^\n]*|#[^\n]*", "", source, flags=re.S)
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[^\s"]+', source)
    return "".join(token if token.startswith('"') else token.lower() for token in tokens)


def definitions(source: str) -> dict[str, str]:
    starts = list(re.finditer(r"CREATE (?:OR REPLACE )?QUERY (\w+)", source, re.I))
    result = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(source)
        result[match[1]] = source[match.start() : end].split("USE GRAPH")[0].strip()
    return result


def parameter_names(definition: str) -> set[str]:
    """Names in `CREATE QUERY name(TYPE a, TYPE b = default, ...)`."""
    match = re.search(r"QUERY\s+\w+\s*\(", definition, re.I)
    if match is None:
        raise ValueError("Query definition has no parameter list")
    depth, quoted, current, parts = 1, False, "", []
    for char in definition[match.end() :]:
        if quoted:
            quoted = char != '"'
        elif char == '"':
            quoted = True
        elif char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            parts.append(current)
            current = ""
            continue
        current += char
    else:
        raise ValueError("Unterminated query parameter list")
    parts.append(current)
    names = set()
    for part in parts:
        declaration = part.split("=", 1)[0].split()
        if declaration:
            names.add(declaration[-1])
    return names


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


def repository_queries(files: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Query name -> (repository file, definition text), in file order."""
    result: dict[str, tuple[str, str]] = {}
    for path in files:
        for name, text in definitions((REPOSITORY_ROOT / path).read_text()).items():
            result[name] = (path, text)
    return result


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


def source_counts(executor: Any) -> dict[str, int]:
    """Live vertex counts by type, excluding experiment metadata vertex types."""
    raw = connection_call(
        executor, "getVertexCount", lambda conn: conn.getVertexCount("*", realtime=True)
    )
    if not isinstance(raw, dict):
        raise ValueError("TigerGraph did not return counts by vertex type")
    return {
        str(name): int(count)
        for name, count in raw.items()
        if str(name) not in EXPERIMENT_METADATA_TYPES
    }


def verify_frozen_source(executor: Any, manifest: dict[str, Any]) -> None:
    """Recheck live provenance on every streamed run, including prepared-data reuse.

    Counts and headers catch drift, but cannot prove absence of same-count edits.
    The experiment still requires an operationally frozen source. Scope vertices
    are experiment metadata, so creating another scope does not invalidate data.
    """
    verify_sources(executor)
    recorded = {
        name: count
        for name, count in manifest["source"]["source_counts"].items()
        if name not in EXPERIMENT_METADATA_TYPES
    }
    if source_counts(executor) != recorded:
        raise ValueError("Live graph counts changed; freeze the source and prepare a new dataset")
    config = manifest["config"]
    if config["evaluation_protocol"] == "strict_inductive":
        try:
            verify_scope(executor, config)
        except ValueError as error:
            raise ValueError(f"Prepared experiment scope is no longer valid: {error}") from None


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
        migration = REPOSITORY_ROOT / "gsql/schema/migrations/temporal_training_scope.gsql"
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
