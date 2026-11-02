"""Install the queries of gsql/ on the graph, check that the installed ones are current, and
refuse versions the parity check has not passed.

A query is current when SHOW QUERY gives the repository's text (comments, whitespace and
case outside string literals aside) and its REST endpoint is enabled with the same
parameters. install creates and installs only the queries that are not. Every write runs
once through the executor, so it is never repeated behind the caller's back. The schema
(gsql/schema/) is not installed here: the loader creates it on an empty graph.

No query is fetched until its version has passed parity on the 5,000-person test world,
which mule parity records in gsql/parity.toml, committed with the code (docs/design.md,
"The pipeline").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
import hashlib
from pathlib import Path
import re
import time
import tomllib
from typing import Any

from ..paths import GSQL_DIR, parity_record
from .executor import (
    AVAILABILITY,
    SERVER_TIMEOUT,
    ConnectionExecutor,
    TransientQueryError,
    failure_class,
)
from .queries import GRAPH_NAME, QUERY_FILES

# GSQL comments, which neither a compared text nor a version includes.
COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*|#[^\n]*", re.S)
# Parameters an endpoint lists beyond the query's own.
BUILTIN_PARAMETERS = frozenset({"query", "read_committed"})
# How long the install request may wait for compilation before the endpoints are polled.
INSTALL_DEADLINE_S = 60 * 60.0
# The problems of query_problems that only a new CREATE resolves.
MISSING = "is missing on the server"
DIFFERS = "differs from the repository"


def normalized(text: str) -> str:
    """A query's text as compared and hashed: no comments or whitespace, and lower case
    outside string literals."""
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[^\s"]+', COMMENTS.sub("", text))
    return "".join(token if token.startswith('"') else token.lower() for token in tokens)


def text_version(prefix: str, definition: str) -> str:
    """A query's version: prefix and 12 hex digits of the sha256 of its normalized definition,
    so no comment or layout changes it. No query prints it; an old query that still prints
    its version as a string literal has that literal left out."""
    text = re.sub(rf'"{re.escape(prefix)}[0-9a-f]{{12}}"', '""', definition)
    return prefix + hashlib.sha256(normalized(text).encode()).hexdigest()[:12]


def definitions(text: str) -> dict[str, str]:
    """Each query a GSQL text defines, by name: from QUERY to its closing brace.

    CREATE and OR REPLACE are left out, so a file and SHOW QUERY compare equal whichever
    of the two the server echoes.
    """
    starts = list(re.finditer(r"CREATE\s+(?:OR\s+REPLACE\s+)?(QUERY\s+(\w+))", text, re.I))
    found: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[match.start(1) : end].split("USE GRAPH")[0]
        found[match[2]] = body[: body.rfind("}") + 1]
    return found


def parameter_names(definition: str) -> list[str]:
    """The parameter names of `QUERY name(TYPE a, TYPE b = default, ...)`, in order. A comment
    in the list, such as `UINT t_ms // the scoring time, a whole second`, is skipped."""
    match = re.search(r"QUERY\s+\w+\s*\(", definition, re.I)
    if match is None:
        raise ValueError("A query definition has no parameter list")
    depth, quoted, current, parts = 1, False, "", list[str]()
    rest = definition[match.end() :]
    skip_to = 0
    for index, char in enumerate(rest):
        if index < skip_to:
            continue
        if not quoted and rest.startswith(("//", "/*"), index):
            end = rest.find("\n" if rest.startswith("//", index) else "*/", index + 2)
            skip_to = len(rest) if end < 0 else end + (0 if rest[end] == "\n" else 2)
            continue
        if quoted:
            quoted = char != '"'
        elif char == '"':
            quoted = True
        elif char in "(<":
            depth += 1
        elif char in ")>":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            parts.append(current)
            current = ""
            continue
        current += char
    else:
        raise ValueError("A query's parameter list is not closed")
    parts.append(current)
    return [part.split("=", 1)[0].split()[-1] for part in parts if part.strip()]


def repository_queries() -> dict[str, str]:
    """Every query the code installs, by name: its definition in its file under gsql/."""
    found: dict[str, str] = {}
    for name, relative in QUERY_FILES.items():
        path = GSQL_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"gsql/{relative} is missing: the query {name} is registered but not written"
            )
        defined = definitions(path.read_text())
        if list(defined) != [name]:
            raise ValueError(f"gsql/{relative} must define the query {name} and no other")
        found[name] = defined[name]
    return found


def installed_endpoints(executor: ConnectionExecutor) -> dict[str, dict[str, Any]]:
    """The graph's installed-query endpoints by query name, with `enabled` and `parameters`."""
    raw = executor.call(lambda conn: conn.getInstalledQueries(), what="getInstalledQueries")
    if not isinstance(raw, dict):
        raise ValueError("TigerGraph did not list the installed queries")
    prefix = f"GET /query/{GRAPH_NAME}/"
    return {
        str(endpoint)[len(prefix) :]: info
        for endpoint, info in raw.items()
        if str(endpoint).startswith(prefix) and isinstance(info, dict)
    }


def query_problems(executor: ConnectionExecutor) -> dict[str, list[str]]:
    """Per query that is not current, why: its text is missing or differs, or its endpoint
    is not enabled or has other parameters (read-only).

    SHOW QUERY also shows a query that is created but not installed, so the endpoint is
    checked apart from the text.
    """
    endpoints = installed_endpoints(executor)
    problems: dict[str, list[str]] = {}
    for name, source in repository_queries().items():
        shown = executor.gsql(f"USE GRAPH {GRAPH_NAME}\nSHOW QUERY {name}", what="SHOW QUERY")
        actual = definitions(shown).get(name)
        if actual is None:
            problems[name] = [MISSING]
            continue
        issues: list[str] = []
        if normalized(actual) != normalized(source):
            issues.append(DIFFERS)
        endpoint = endpoints.get(name)
        if endpoint is None or endpoint.get("enabled") is not True:
            issues.append("is not installed")
        elif set(endpoint.get("parameters", {})) - BUILTIN_PARAMETERS != set(
            parameter_names(source)
        ):
            issues.append("has other parameters than the repository")
        if issues:
            problems[name] = issues
    return problems


def verify(executor: ConnectionExecutor) -> None:
    """Raise unless every query is installed, enabled and the repository's text."""
    problems = query_problems(executor)
    if problems:
        raise ValueError(
            "The installed queries are not the repository's: "
            + "; ".join(f"{name} {issue}" for name, issues in problems.items() for issue in issues)
            + ". Run `mule install`."
        )


def _created(output: str) -> bool:
    return "Successfully created queries" in output and not re.search(
        r"[1-9]\d* syntax error|(?:Type Check|Semantic Check|Syntax) Error", output, re.I
    )


def _failed(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    message = str(status.get("message", "")).upper()
    return status.get("error") in (True, "true") or "FAILED" in message


def _unanswered(error: Exception) -> bool:
    """Whether the install request failed because the client stopped waiting, so the
    server may still be compiling. The executor wraps the client's error, its cause."""
    cause = error.__cause__ if isinstance(error, TransientQueryError) else error
    return failure_class(cause or error) in (AVAILABILITY, SERVER_TIMEOUT)


def install(
    executor: ConnectionExecutor,
    *,
    deadline_s: float = INSTALL_DEADLINE_S,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Create and install every query that is not current; the names it installed.

    Only a query whose text is missing or differs is created again, since CREATE OR
    REPLACE disables an installed endpoint until it is installed again; one whose text is
    current but whose endpoint is not (a compilation that did not finish) is only
    installed. The install request waits up to deadline_s. When the client stops waiting
    first, the endpoint listing is polled every poll_s seconds until the queries are
    enabled, and verify decides whether the install worked.
    """
    if executor.graph_name != GRAPH_NAME:
        raise ValueError(f"The queries are for the graph {GRAPH_NAME}, not {executor.graph_name}")
    queries = repository_queries()
    problems = query_problems(executor)
    stale = [name for name in queries if name in problems]
    if not stale:
        return []
    changed = [name for name in stale if {MISSING, DIFFERS} & set(problems[name])]
    if changed:
        text = "\n\n".join(f"CREATE OR REPLACE {queries[name]}" for name in changed)
        output = executor.gsql(f"USE GRAPH {GRAPH_NAME}\n{text}\n", what="CREATE QUERY", attempts=1)
        if not _created(output):
            raise RuntimeError(output)
    started = clock()
    try:
        with executor.client.request_timeout(read_s=deadline_s):
            status = executor.call(
                lambda conn: conn.installQueries(stale), what="installQueries", attempts=1
            )
    except Exception as error:
        if not _unanswered(error):
            raise
    else:
        if _failed(status):
            raise RuntimeError(f"Query installation failed: {status}")
    while True:
        endpoints = installed_endpoints(executor)
        pending = [name for name in stale if endpoints.get(name, {}).get("enabled") is not True]
        if not pending:
            break
        if clock() - started > deadline_s:
            raise TimeoutError(
                f"The queries {pending} are still not installed after {deadline_s:.0f} s. The "
                "server may still be compiling: once it has finished, install again."
            )
        sleep(poll_s)
    verify(executor)
    return stale


# The fields of each record of gsql/parity.toml: the query, the version of its text that
# passed, the TigerGraph version it ran on, and the identity and date of that world's run.
PARITY_FIELDS = {"query": str, "version": str, "tigergraph": str, "world": str, "date": date}


def parity_versions(record: Path | None = None) -> set[tuple[str, str]]:
    """The (query, version) pairs the parity record holds, each one that passed parity.

    The record is TOML, one `[[passed]]` table per query version with the PARITY_FIELDS.
    """
    path = parity_record() if record is None else record
    if not path.is_file():
        return set()
    passed = tomllib.loads(path.read_text()).get("passed", [])
    if not isinstance(passed, list):
        raise ValueError(f"{path} must hold [[passed]] tables")
    found: set[tuple[str, str]] = set()
    for entry in passed:
        if not isinstance(entry, dict) or set(entry) != set(PARITY_FIELDS):
            raise ValueError(f"{path} holds a record without exactly {sorted(PARITY_FIELDS)}")
        for field, kind in PARITY_FIELDS.items():
            if not isinstance(entry[field], kind):
                raise ValueError(f"{path} holds a {field} that is not a {kind.__name__}")
        query, version = str(entry["query"]), str(entry["version"])
        if not version.startswith(f"{query}-"):
            raise ValueError(f"{path} records the version {version} for the query {query}")
        found.add((query, version))
    return found


def require_parity(versions: Mapping[str, str], record: Path | None = None) -> None:
    """Raise unless the parity record holds every version (query name to version), naming
    every one it lacks."""
    passed = parity_versions(record)
    missing = [
        f"{name} {version}" for name, version in versions.items() if (name, version) not in passed
    ]
    if missing:
        path = parity_record() if record is None else record
        raise ValueError(
            f"No parity record for {', '.join(missing)} in {path}: run `mule parity` on the "
            "5,000-person test world and commit the record"
        )
