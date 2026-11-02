"""The scored population of each week (docs/design.md, "The scored population" and "Cache"):
list_accounts, read a page at a time by strictly increasing id, and kept in the world's cache
as population.parquet in the scoring time's folder, beside the hubs and the query tables.

The file is written once, through a temporary file and a rename, and read back through
list_accounts' own parser, so cached rows pass the live checks. Its folder is keyed by the
world's identity, the query versions and the settings (tigergraph.world.cache_key), so a
population of another query version is never read.
"""

from __future__ import annotations

from pathlib import Path
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from ..tigergraph.executor import QueryExecutor
from ..tigergraph.queries import LIST_ACCOUNTS, list_accounts
from ..tigergraph.queries.list_accounts import ScorableAccount
from ..tigergraph.queries.rows import PAGE_LIMIT
from ..tigergraph.world import time_folder

POPULATION_FILE = "population.parquet"
SCHEMA = pa.schema([("account_id", pa.string()), ("owner_id", pa.string())])


def list_population(
    executor: QueryExecutor, t_ms: int, page: int = PAGE_LIMIT
) -> tuple[ScorableAccount, ...]:
    """The scorable accounts at the time t, each with its owner, by increasing id: every page
    of list_accounts, until one comes back short."""
    found: list[ScorableAccount] = []
    after_id = ""
    while True:
        rows = executor.run(LIST_ACCOUNTS, list_accounts.params(t_ms, after_id, page))
        accounts = list_accounts.parse(rows, after_id, page)
        found.extend(accounts)
        if len(accounts) < page:
            return tuple(found)
        after_id = accounts[-1].account_id


def population_file(root: Path, t_ms: int) -> Path:
    """The population at the time t in a world's cache folder of query versions and
    settings."""
    return root / time_folder(t_ms) / POPULATION_FILE


def write_population(path: Path, accounts: tuple[ScorableAccount, ...]) -> None:
    """Write a scoring time's population once, through a temporary file and a rename."""
    table = pa.table(
        {
            "account_id": [account.account_id for account in accounts],
            "owner_id": [account.owner_id for account in accounts],
        },
        schema=SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_population(path: Path) -> tuple[ScorableAccount, ...]:
    """A cached population, reparsed by list_accounts' parser as one page of every account."""
    rows = pq.read_table(path, schema=SCHEMA).to_pylist()
    response: list[dict[str, object]] = [
        {"status": "ok"},
        {"accounts": [{"attributes": row} for row in rows]},
    ]
    return list_accounts.parse(response, "", max(1, len(rows)))


def population(
    executor: QueryExecutor, root: Path, t_ms: int, page: int = PAGE_LIMIT
) -> tuple[ScorableAccount, ...]:
    """The scorable accounts at the time t: from the cache when it holds them, else listed
    and cached."""
    path = population_file(root, t_ms)
    if path.is_file():
        return read_population(path)
    accounts = list_population(executor, t_ms, page)
    write_population(path, accounts)
    return accounts
