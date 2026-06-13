"""
Install GSQL queries onto the graph from files in this repo.
"""

import re
import time
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path

_FAILURE_MARKERS: tuple[str, ...] = (
    "error",
    "fail",
    "could not",
    "cannot",
    "not valid",
    "syntax",
)

# CREATE [OR REPLACE] [DISTRIBUTED] QUERY <name> -- captures the installed name.
_CREATE_QUERY_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:DISTRIBUTED\s+)?QUERY\s+([A-Za-z_]\w*)"
)


class GsqlInstallError(RuntimeError):
    pass


def get_query_from_file(registry_name: str) -> str:
    """
     Query name for a gsql_paths registry key.

    Parses `CREATE [OR REPLACE] [DISTRIBUTED] QUERY <name>` from the .gsql file.
    """
    text = gsql_path(registry_name).read_text(encoding="utf-8")
    match = _CREATE_QUERY_RE.search(text)
    if match is None:
        raise GsqlInstallError(
            f"{registry_name}: no CREATE QUERY found in source; not an installable query "
            "(a loading job or schema script installs differently)."
        )
    return match.group(1)


def _run_gsql(client: Client, statement: str) -> str:
    result = client.conn.gsql(statement)
    if not isinstance(result, str):
        raise GsqlInstallError(f"expected text from gsql(), got {type(result).__name__}")
    return result


def _check_output(registry_name: str, action: str, output: str) -> None:
    low = output.lower()
    if any(marker in low for marker in _FAILURE_MARKERS):
        raise GsqlInstallError(
            f"{action} {registry_name!r} reported a problem:\n{output.strip()[:800]}"
        )


def install_query(client: Client, registry_name: str, drop_first: bool = True) -> str:
    """
    Install one query from its .gsql file; return the install log.

    """
    path = gsql_path(registry_name)
    if not path.is_file():
        raise GsqlInstallError(f"{registry_name}: source file not found at {path}")
    name = get_query_from_file(registry_name)
    text = path.read_text(encoding="utf-8")

    if drop_first:
        # a failed drop is tolerated only when the query was simply not installed
        drop_out = _run_gsql(client, f"USE GRAPH {client.graphname}\nDROP QUERY {name}\n")
        low = drop_out.lower()
        if any(m in low for m in _FAILURE_MARKERS) and "not exist" not in low:
            raise GsqlInstallError(f"drop {name!r} failed:\n{drop_out.strip()[:800]}")

    install_out = _run_gsql(client, text + f"\nINSTALL QUERY {name}\n")
    _check_output(registry_name, "install", install_out)
    return install_out


def install_queries(
    client: Client, registry_names: list[str], drop_first: bool = True
) -> dict[str, str]:
    """
    Installs queries in order; return {registry_name: install log}.
    """
    logs: dict[str, str] = {}
    for registry_name in registry_names:
        logs[registry_name] = install_query(client, registry_name, drop_first=drop_first)
    return logs


def _extract_request_id(submitted: object) -> str:
    # runInstalledQuery(runAsync=True) returns the detached-mode request id. It
    # is normally a bare string, but tolerate a 1-element list or a dict with a
    # request id so a pyTigerGraph version change does not silently break this.
    if isinstance(submitted, str):
        return submitted
    if isinstance(submitted, list) and len(submitted) == 1 and isinstance(submitted[0], str):
        return submitted[0]
    if isinstance(submitted, dict):
        for key in ("request_id", "requestid", "requestId"):
            value = submitted.get(key)
            if isinstance(value, str):
                return value
    raise GsqlInstallError(f"could not read a request id from async submission: {submitted!r}")


def _status_of(status_response: object) -> str:
    # checkQueryStatus returns a list of {status: success|running|aborted, ...}.
    # We submit one query at a time, so read the first entry's status. The bare
    # "error"/"fail" markers are deliberately NOT applied here: a "running"
    # status legitimately contains neither, and reusing _check_output would
    # misread an in-progress poll as a failure.
    rows: list[object]
    if isinstance(status_response, list):
        rows = cast(list[object], status_response)
    elif isinstance(status_response, dict):
        results = status_response.get("results")
        rows = cast(list[object], results) if isinstance(results, list) else [status_response]
    else:
        raise GsqlInstallError(f"unexpected query-status payload: {status_response!r}")
    if not rows:
        raise GsqlInstallError("empty query-status payload")
    first = rows[0]
    if not isinstance(first, dict):
        raise GsqlInstallError(f"unexpected query-status row: {first!r}")
    status = cast(dict[str, object], first).get("status")
    if not isinstance(status, str):
        raise GsqlInstallError(f"query-status row has no string status: {first!r}")
    return status


def run_query(
    client: Client,
    registry_name: str,
    params: dict[str, object] | None = None,
    poll_seconds: float = 5.0,
    max_wait_seconds: float = 14_400.0,
    query_timeout_ms: int = 7_200_000,
) -> list[object]:
    """
    Run an installed query by its gsql_paths registry key, in detached (async)
    mode, and block until it finishes.

    On Savanna a synchronous runInstalledQuery holds one HTTPS connection open
    for the query's whole duration; long feature queries on a large graph
    outlive the managed load balancer's idle window and the connection is
    dropped (surfacing as ReadTimeout even with read timeout=None -- the value
    is irrelevant because the socket is severed, not timed out). Detached mode
    avoids this: the query runs server-side and we issue only short calls --
    submit, then poll, then fetch -- so there is no long-lived connection to
    drop. The feature queries are idempotent, so a re-run after any transient
    failure is safe.

    poll_seconds: gap between status checks. max_wait_seconds: hard ceiling on
    total wait (default 4h) so a genuinely stuck query cannot hang forever.
    query_timeout_ms: the SERVER-SIDE limit (milliseconds) TigerGraph applies
    before it aborts the query itself, the same value a GSQL session sets with
    SET query_timeout. This is distinct from max_wait_seconds (client-side poll
    ceiling): heavy feature queries (fastrp, pagerank) far exceed RESTPP's small
    default and are aborted server-side unless this is raised. Default 2h.
    """
    name = get_query_from_file(registry_name)
    submitted: object = client.conn.runInstalledQuery(
        name,
        params if params is not None else {},
        timeout=query_timeout_ms,
        runAsync=True,
    )
    request_id = _extract_request_id(submitted)

    waited = 0.0
    while True:
        status = _status_of(client.conn.checkQueryStatus(request_id))
        if status == "success":
            break
        if status in ("aborted", "timeout"):
            raise GsqlInstallError(
                f"async query {name!r} ended with status {status!r} "
                + f"(request_id {request_id})."
            )
        if waited >= max_wait_seconds:
            raise GsqlInstallError(
                f"async query {name!r} still {status!r} after {max_wait_seconds:.0f}s "
                + f"(request_id {request_id}); aborting the wait."
            )
        time.sleep(poll_seconds)
        waited += poll_seconds

    return cast(list[object], client.conn.getQueryResult(request_id))
