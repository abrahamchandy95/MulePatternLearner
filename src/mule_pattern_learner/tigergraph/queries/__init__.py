"""The registry of the queries of docs/design.md ("The queries"): every query's name, its file
under gsql/, the version its text prints, the builder of its parameters and the dataclasses of
its rows, and which entry point of the executor may run it.

Every one of the twelve queries is registered here from the start, with its file. The query
steps add the rest as each query is written: its version, stamped by testing.versions, and its
parameter builder and row dataclasses, from a module of this package beside rows.py, which
holds what every query's parser shares. Until its step is done a query's file does not exist,
so `mule install` refuses to run and the live compile test compiles the queries written so far.

read_answer_key alone is registered with its version but without its builder and rows: its
module, read_answer_key.py, is imported by evaluation/ and testing/ only, which an import
contract in pyproject.toml holds (the Labels guard), so nothing that imports the registry
imports the answer key's parser.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...paths import GSQL_DIR
from . import list_accounts, profile_hubs, read_hubs, read_labels, read_money_flow, read_world

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

GRAPH_NAME = "Mule_Pattern_Learner"

# The infrastructure queries.
LIST_ACCOUNTS = "list_accounts"
READ_LABELS = "read_labels"
READ_ANSWER_KEY = "read_answer_key"
READ_HUBS = "read_hubs"
READ_WORLD = "read_world"
PROFILE_HUBS = "profile_hubs"
# The behaviour queries, B1 to B6.
READ_MONEY_FLOW = "read_money_flow"
READ_COUNTERPARTIES = "read_counterparties"
READ_ACTIVITY_CHANGE = "read_activity_change"
READ_PAYMENT_EVENTS = "read_payment_events"
READ_SHARED_IDENTITY = "read_shared_identity"
READ_COMMUNITY = "read_community"

# The folders of gsql/ that hold queries: evaluation/ holds the answer key's alone.
QUERIES_FOLDER = "queries"
EVALUATION_FOLDER = "evaluation"


@dataclass(frozen=True)
class Query:
    """A query: its name, the folder of gsql/ its file is in, and what its step adds: the
    version its text prints, the builder of its parameters and the dataclasses of its rows."""

    name: str
    folder: str
    version: str | None = None
    params: Callable[..., dict[str, Any]] | None = None
    rows: tuple[type[DataclassInstance], ...] = ()

    @property
    def file(self) -> str:
        """Its file, relative to gsql/."""
        return f"{self.folder}/{self.name}.gsql"


REGISTRY: dict[str, Query] = {
    query.name: query
    for query in (
        Query(
            LIST_ACCOUNTS,
            QUERIES_FOLDER,
            "list_accounts-a270fbdf9620",
            list_accounts.params,
            (list_accounts.ScorableAccount,),
        ),
        Query(
            READ_LABELS,
            QUERIES_FOLDER,
            "read_labels-e22e410b15e2",
            read_labels.params,
            (read_labels.IdentifiedMule,),
        ),
        Query(
            READ_HUBS,
            QUERIES_FOLDER,
            "read_hubs-3b28b3433685",
            read_hubs.params,
            (read_hubs.HubAccount, read_hubs.Leaf),
        ),
        Query(
            READ_WORLD,
            QUERIES_FOLDER,
            "read_world-0c985b5f9e81",
            read_world.params,
            (read_world.World,),
        ),
        Query(
            PROFILE_HUBS,
            QUERIES_FOLDER,
            "profile_hubs-a4603a434498",
            profile_hubs.params,
            (profile_hubs.HubProfile,),
        ),
        Query(
            READ_MONEY_FLOW,
            QUERIES_FOLDER,
            "read_money_flow-04c652eeaf53",
            read_money_flow.params,
            (read_money_flow.MoneyFlow, read_money_flow.TrackedInflow),
        ),
        Query(READ_COUNTERPARTIES, QUERIES_FOLDER),
        Query(READ_ACTIVITY_CHANGE, QUERIES_FOLDER),
        Query(READ_PAYMENT_EVENTS, QUERIES_FOLDER),
        Query(READ_SHARED_IDENTITY, QUERIES_FOLDER),
        Query(READ_COMMUNITY, QUERIES_FOLDER),
        Query(READ_ANSWER_KEY, EVALUATION_FOLDER, "read_answer_key-3d5467c23b01"),
    )
}

# Every query the code installs, and its file under gsql/.
QUERY_FILES: dict[str, str] = {name: query.file for name, query in REGISTRY.items()}
# The version each written query prints (installer.text_version). The parity record holds
# the versions that passed, the cache is keyed by them and a model's contract lock stores them.
VERSIONS: dict[str, str] = {
    name: query.version for name, query in REGISTRY.items() if query.version is not None
}

# Which entry point of the executor runs which query (docs/design.md, "Query plumbing"). Fetch
# runs these and nothing else, so neither the answer key nor anything else installed in the
# graph, such as the loader's verify_load, is ever fetched.
FETCH_ALLOWLIST = frozenset(
    {
        LIST_ACCOUNTS,
        READ_LABELS,
        READ_HUBS,
        READ_WORLD,
        READ_MONEY_FLOW,
        READ_COUNTERPARTIES,
        READ_ACTIVITY_CHANGE,
        READ_PAYMENT_EVENTS,
        READ_SHARED_IDENTITY,
        READ_COMMUNITY,
    }
)
# profile_hubs runs only from mule bench.
BENCH_ONLY = frozenset({PROFILE_HUBS})
# read_answer_key runs only from evaluation's executor.
EVALUATION_ONLY = frozenset({READ_ANSWER_KEY})


def written_files() -> dict[str, str]:
    """The registered queries whose files exist, with their files: every query once its step
    is done."""
    return {name: file for name, file in QUERY_FILES.items() if (GSQL_DIR / file).is_file()}
