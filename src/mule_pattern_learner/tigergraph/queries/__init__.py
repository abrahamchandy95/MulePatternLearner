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
from . import (
    list_accounts,
    mark_crowds,
    profile_hubs,
    read_activity_change,
    read_community,
    read_counterparties,
    read_hubs,
    read_labels,
    read_money_flow,
    read_payment_events,
    read_shared_identity,
    read_world,
)

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
# read_payment_events split into queries of one job each: the payment graph's reads, and the
# marks that store what they read on the graph.
READ_NEWEST_PAYMENTS = "read_newest_payments"
READ_PAYMENTS = "read_payments"
READ_ACCOUNTS = "read_accounts"
READ_OTHER_ACCOUNTS = "read_other_accounts"
MARK_HUBS = "mark_hubs"
MARK_PAYMENT_GAPS = "mark_payment_gaps"
MARK_PAIR_HISTORY = "mark_pair_history"
# read_shared_identity split into queries of one job each.
MARK_CROWDS = "mark_crowds"
READ_CUSTOMER_ACCOUNTS = "read_customer_accounts"
READ_CUSTOMER_DEVICES = "read_customer_devices"
READ_ACCOUNT_DEVICES = "read_account_devices"
READ_DEVICE_USERS = "read_device_users"
READ_DEVICE_COUNTS = "read_device_counts"
READ_CUSTOMER_LINK_COUNTS = "read_customer_link_counts"
READ_CUSTOMER_CO_USER_COUNTS = "read_customer_co_user_counts"

# read_money_flow and read_activity_change split into queries of one job each.
READ_PAYMENT_TOTALS = "read_payment_totals"
READ_MONEY_VELOCITY = "read_money_velocity"
READ_PAYMENT_RHYTHM = "read_payment_rhythm"

# read_counterparties split into queries of one job each.
READ_COUNTERPARTY_PAYERS = "read_counterparty_payers"
READ_COUNTERPARTY_PAYEES = "read_counterparty_payees"

# read_community split into queries of one job each.
READ_COMMUNITY_DEVICES = "read_community_devices"
READ_COMMUNITY_DEVICE_USERS = "read_community_device_users"
READ_COMMUNITY_PAYMENTS = "read_community_payments"

# The folders of gsql/ that hold queries: evaluation/ holds the answer key's alone.
QUERIES_FOLDER = "queries"
EVALUATION_FOLDER = "evaluation"
# The finalized queries of one job each, beside the old ones until the refactor removes them.
# read_world stays in queries/: only research on PhantomLedger worlds reads it, never production
# (the owner, 2 November 2026).
FINAL_FOLDER = "final"


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
            FINAL_FOLDER,
            "list_accounts-ff764a4688c6",
            list_accounts.params,
            (list_accounts.ScorableAccount,),
        ),
        Query(
            READ_LABELS,
            FINAL_FOLDER,
            "read_labels-52bb4b4a5df5",
            read_labels.params,
            (read_labels.IdentifiedMule,),
        ),
        Query(
            READ_HUBS,
            QUERIES_FOLDER,
            "read_hubs-13de3a1aab4c",
            read_hubs.params,
            (read_hubs.HubAccount, read_hubs.Leaf),
        ),
        Query(
            READ_WORLD,
            QUERIES_FOLDER,
            "read_world-7264bd8030e8",
            read_world.params,
            (read_world.World,),
        ),
        Query(
            PROFILE_HUBS,
            FINAL_FOLDER,
            "profile_hubs-5afd8c9e5601",
            profile_hubs.params,
            (profile_hubs.HubProfile,),
        ),
        Query(
            READ_MONEY_FLOW,
            QUERIES_FOLDER,
            "read_money_flow-5c7b4142fc25",
            read_money_flow.params,
            (read_money_flow.MoneyFlow, read_money_flow.TrackedInflow),
        ),
        Query(
            READ_COUNTERPARTIES,
            QUERIES_FOLDER,
            "read_counterparties-d1144ba6fca5",
            read_counterparties.params,
            (read_counterparties.Counterparties,),
        ),
        Query(
            READ_ACTIVITY_CHANGE,
            QUERIES_FOLDER,
            "read_activity_change-1615ec14e0f2",
            read_activity_change.params,
            (read_activity_change.ActivityChange,),
        ),
        Query(
            READ_PAYMENT_EVENTS,
            QUERIES_FOLDER,
            "read_payment_events-011b7ee36654",
            read_payment_events.params,
            (
                read_payment_events.EventAccount,
                read_payment_events.PaymentEvent,
                read_payment_events.OwnedBy,
            ),
        ),
        Query(
            READ_SHARED_IDENTITY,
            QUERIES_FOLDER,
            "read_shared_identity-59f0ed612e93",
            read_shared_identity.params,
            (
                read_shared_identity.CustomerRow,
                read_shared_identity.NodeRow,
                read_shared_identity.IdentityAccount,
                read_shared_identity.Enrolled,
                read_shared_identity.OwnerEnrolled,
                read_shared_identity.Seen,
            ),
        ),
        Query(
            READ_COMMUNITY,
            QUERIES_FOLDER,
            "read_community-04e5fd86e262",
            read_community.params,
            (read_community.Community,),
        ),
        Query(READ_ANSWER_KEY, EVALUATION_FOLDER, "read_answer_key-7efe564ea071"),
        Query(READ_NEWEST_PAYMENTS, FINAL_FOLDER, "read_newest_payments-0b75a96e6c3a"),
        Query(READ_PAYMENTS, FINAL_FOLDER, "read_payments-0c683c8e6406"),
        Query(READ_ACCOUNTS, FINAL_FOLDER, "read_accounts-1b9facb47b88"),
        Query(READ_OTHER_ACCOUNTS, QUERIES_FOLDER, "read_other_accounts-1757ed3d0993"),
        Query(MARK_HUBS, FINAL_FOLDER, "mark_hubs-d41d68e819d7"),
        Query(MARK_PAYMENT_GAPS, FINAL_FOLDER, "mark_payment_gaps-a6cef942330c"),
        Query(MARK_PAIR_HISTORY, FINAL_FOLDER, "mark_pair_history-1bae90bdfce4"),
        Query(
            MARK_CROWDS,
            FINAL_FOLDER,
            "mark_crowds-8a193ea9b67c",
            mark_crowds.params,
            (mark_crowds.Crowd,),
        ),
        Query(READ_CUSTOMER_ACCOUNTS, FINAL_FOLDER, "read_customer_accounts-0b6630ec2f5f"),
        Query(READ_CUSTOMER_DEVICES, FINAL_FOLDER, "read_customer_devices-66e0b1e8fc96"),
        Query(READ_ACCOUNT_DEVICES, FINAL_FOLDER, "read_account_devices-ebe7e275c834"),
        Query(READ_DEVICE_USERS, FINAL_FOLDER, "read_device_users-0938bb0b2960"),
        Query(READ_DEVICE_COUNTS, FINAL_FOLDER, "read_device_counts-4785da1c511a"),
        Query(READ_CUSTOMER_LINK_COUNTS, FINAL_FOLDER, "read_customer_link_counts-cf32f9df822e"),
        Query(
            READ_CUSTOMER_CO_USER_COUNTS,
            FINAL_FOLDER,
            "read_customer_co_user_counts-ce87d067be77",
        ),
        Query(READ_PAYMENT_TOTALS, FINAL_FOLDER, "read_payment_totals-91e3358b33f7"),
        Query(READ_MONEY_VELOCITY, FINAL_FOLDER, "read_money_velocity-9c1b28e9c436"),
        Query(READ_PAYMENT_RHYTHM, FINAL_FOLDER, "read_payment_rhythm-310dc9fdcdc5"),
        Query(READ_COUNTERPARTY_PAYERS, FINAL_FOLDER, "read_counterparty_payers-2a91b4691286"),
        Query(READ_COUNTERPARTY_PAYEES, FINAL_FOLDER, "read_counterparty_payees-9fa110e16b6a"),
        Query(READ_COMMUNITY_DEVICES, FINAL_FOLDER, "read_community_devices-5d655005c1d4"),
        Query(
            READ_COMMUNITY_DEVICE_USERS, FINAL_FOLDER, "read_community_device_users-7158523c79b8"
        ),
        Query(READ_COMMUNITY_PAYMENTS, FINAL_FOLDER, "read_community_payments-388eca70b7ae"),
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
        MARK_HUBS,
        READ_NEWEST_PAYMENTS,
        READ_PAYMENTS,
        READ_ACCOUNTS,
        READ_OTHER_ACCOUNTS,
        MARK_CROWDS,
        READ_CUSTOMER_ACCOUNTS,
        READ_CUSTOMER_DEVICES,
        READ_ACCOUNT_DEVICES,
        READ_DEVICE_USERS,
        READ_DEVICE_COUNTS,
        READ_CUSTOMER_LINK_COUNTS,
        READ_CUSTOMER_CO_USER_COUNTS,
        READ_PAYMENT_TOTALS,
        READ_MONEY_VELOCITY,
        READ_PAYMENT_RHYTHM,
        READ_COUNTERPARTY_PAYERS,
        READ_COUNTERPARTY_PAYEES,
        READ_COMMUNITY_DEVICES,
        READ_COMMUNITY_DEVICE_USERS,
        READ_COMMUNITY_PAYMENTS,
    }
)
# profile_hubs runs only from mule bench.
BENCH_ONLY = frozenset({PROFILE_HUBS})
# read_answer_key runs only from evaluation's executor.
EVALUATION_ONLY = frozenset({READ_ANSWER_KEY})
# The marks of every payment's timing run once per loaded world, from mule mark.
MARK_ONLY = frozenset({MARK_PAYMENT_GAPS, MARK_PAIR_HISTORY})


def written_files() -> dict[str, str]:
    """The registered queries whose files exist, with their files: every query once its step
    is done."""
    return {name: file for name, file in QUERY_FILES.items() if (GSQL_DIR / file).is_file()}
