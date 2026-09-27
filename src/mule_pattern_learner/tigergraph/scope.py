"""Experiment scopes: the frozen Temporal_Training_Scope a strict run samples in.

ensure_scope creates a scope on first use; every later preparation and every
streamed run verifies its header (ready, source, split seed) and the scope.unowned
rule its membership was created with. TigerGraphScope reads the scope's accounts.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from typing import Any

from ..config import ScopeConfig
from ..contract.server import SCOPE_VERTEX
from ..runtime.progress import emit
from .executor import (
    ConnectionExecutor,
    QueryExecutor,
    account_pages,
    checked_rows,
    merged_rows,
    printed,
)

POPULATION_QUERY = "temporal_scope_population"


class TigerGraphScope:
    """The ScopeReader of data.ports: temporal_scope_population, paged by account."""

    def __init__(self, executor: QueryExecutor) -> None:
        self.executor = executor

    def population_pages(
        self, scope_id: str, *, include_observed: bool
    ) -> Iterator[list[dict[str, Any]]]:
        """Pages of the scope's accounts; rows carry labels only with include_observed."""
        params = {"scope_id": scope_id, "include_observed": include_observed}
        return account_pages(self.executor, POPULATION_QUERY, params)


def scope_header(executor: ConnectionExecutor, scope_id: str) -> dict[str, Any] | None:
    """Attributes of the Temporal_Training_Scope vertex, or None when it does not exist."""
    from pyTigerGraph.common.exception import TigerGraphException

    try:
        rows = executor.call(
            lambda conn: conn.getVerticesById(SCOPE_VERTEX, [scope_id]),
            what="getVerticesById",
        )
    except TigerGraphException as error:
        if str(error.code) != "601":
            raise
        return None
    if not isinstance(rows, list) or len(rows) > 1:
        raise ValueError("Unexpected scope metadata response")
    return dict(rows[0]["attributes"]) if rows else None


def check_scope(
    attrs: dict[str, Any] | None, scope_id: str, *, source_id: str, split_seed: int
) -> None:
    if attrs is None:
        raise ValueError(f"Prepared experiment scope is missing: {scope_id}")
    if not attrs["ready"] or attrs["source_id"] != source_id or attrs["split_seed"] != split_seed:
        raise ValueError("Scope is incomplete or belongs to a different source/partition")


SCOPE_POLICY_QUERY = "temporal_scope_policy"
# Unowned member Accounts by membership class and side, as temporal_scope_policy prints them.
SCOPE_POLICY_COUNTS = (
    "shared_internal",
    "shared_external",
    "independent_internal",
    "independent_external",
    "linked_internal",
    "linked_external",
    # Unowned bank ledger accounts (account_type "gl"), and how many of them are shared.
    "shared_ledger",
    "ledger_accounts",
)


def scope_policy_counts(executor: QueryExecutor, scope_id: str) -> dict[str, int]:
    """Membership classes of the scope's unowned Accounts, plus `members` (read-only)."""
    merged = merged_rows(
        checked_rows(executor.run(SCOPE_POLICY_QUERY, {"scope_id": scope_id}, timeout_s=900.0))
    )
    names = (*SCOPE_POLICY_COUNTS, "members")
    missing = [name for name in names if name not in merged]
    if missing:
        raise ValueError(
            f"{SCOPE_POLICY_QUERY} response lacks {missing}; install the current queries "
            "(mule-temporal install)"
        )
    return {name: int(merged[name]) for name in names}


def inferred_scope_policy(counts: dict[str, int]) -> str | None:
    """The scope.unowned rule a scope was created with, from its membership classes.

    Under every rule an unowned internal customer account is never shared and an
    unowned external account is never linked. Unowned bank ledger accounts (the
    bank's own "gl" books) are shared exactly when external accounts are: all of them
    under "shared" and "linked", none under "independent". "independent": nothing shared or linked
    (every scope created before the policy existed, such as strict_mule_v1).
    "shared": every unowned external account shared, nothing linked. "linked":
    every unowned external account shared (possibly none exist) and at least one
    internal account linked to its sole owned deposit counterparty. None: no rule
    produces this membership (for example an early draft that also shared
    internal accounts). Rules that wrote identical membership read as the
    simplest of them: "independent" without unowned external accounts or links,
    "shared" when no internal account was linked.
    """
    if counts["shared_internal"] or counts["linked_external"]:
        return None
    shared, linked = counts["shared_external"], counts["linked_internal"]
    ledger, shared_ledger = counts["ledger_accounts"], counts["shared_ledger"]
    if counts["independent_external"] and (shared or linked or shared_ledger):
        return None  # shared and linked scopes share every unowned external account
    if (shared or linked or shared_ledger) and shared_ledger != ledger:
        return None  # ... and every unowned ledger account
    if linked:
        return "linked"
    return "shared" if shared or shared_ledger else "independent"


def check_scope_policy(counts: dict[str, int], scope_id: str, configured: str) -> str:
    """Raise unless the scope was created with the configured scope.unowned rule."""
    stored = inferred_scope_policy(counts)
    if stored is not None and stored == configured:
        return stored
    if stored is None:
        raise ValueError(
            f"Scope {scope_id!r} matches no scope.unowned rule (unowned member classes "
            f"{json.dumps(counts, sort_keys=True)}). Create a new scope: set a new scope.id "
            "in config.ScopeConfig; the next `mule-temporal train` (or `prepare`) creates it."
        )
    raise ValueError(
        f"Scope {scope_id!r} was created with scope.unowned = {stored!r}, but the "
        f"configuration says {configured!r} (unowned member classes "
        f"{json.dumps(counts, sort_keys=True)}). Set scope.unowned = {stored!r} to use this "
        "scope, or set a new scope.id; the next `mule-temporal train` (or `prepare`) "
        "creates it."
    )


def verify_scope(
    executor: ConnectionExecutor,
    scope_id: str,
    *,
    unowned: str,
    source_id: str,
    split_seed: int,
) -> dict[str, int]:
    """Header (ready, source, split seed) and scope.unowned rule of an existing scope."""
    check_scope(
        scope_header(executor, scope_id), scope_id, source_id=source_id, split_seed=split_seed
    )
    counts = scope_policy_counts(executor, scope_id)
    check_scope_policy(counts, scope_id, unowned)
    return counts


def ensure_scope(
    executor: ConnectionExecutor, scope: ScopeConfig, *, source_id: str, split_seed: int
) -> None:
    """Use the frozen scope, creating it on first use unless scope.create is false.

    Creation writes a Temporal_Training_Scope vertex and one membership edge per
    Account and Party, recording the source id and the split seed that partitions it.
    scope.unowned decides the accounts without an owning Party:
    - "independent": each is its own ownership group with a hashed partition.
    - "shared": unowned external accounts and unowned bank ledger ("gl") accounts
      are visible in every phase (partition 1, group "shared:<component>"); other
      unowned internal accounts stay independent.
    - "linked" (the default): as "shared", and an unowned internal account whose
      only owned internal deposit counterparty is one account joins that
      account's ownership group and partition.
    An existing scope must have been created with the configured rule; it is
    inferred from the membership (temporal_scope_policy) and a mismatch raises.
    """
    scope_id = scope.id

    def verified() -> dict[str, int]:
        return verify_scope(
            executor, scope_id, unowned=scope.unowned, source_id=source_id, split_seed=split_seed
        )

    attrs = scope_header(executor, scope_id)
    if attrs is not None:
        counts = verified()
        emit({"scope": scope_id, "unowned_members": counts})
        return
    if not scope.create:
        raise ValueError(
            f"Scope {scope_id!r} does not exist on TigerGraph and scope.create is false; "
            "set it to true to let the first run create it"
        )
    emit({"scope": scope_id, "creating": True})
    created = checked_rows(
        executor.run(
            "temporal_create_training_scope",
            {
                "scope_id": scope_id,
                "source_id": source_id,
                "split_seed": split_seed,
                "unowned_policy": scope.unowned,
            },
            timeout_s=3600.0,
            attempts=1,
        )
    )
    expected = printed(created, "expected_members")
    checked_rows(
        executor.run(
            "temporal_finalize_training_scope",
            {"scope_id": scope_id, "expected_members": expected},
            timeout_s=3600.0,
            attempts=1,
        )
    )
    counts = verified()
    emit({"scope": scope_id, "created": True, "unowned_members": counts})
