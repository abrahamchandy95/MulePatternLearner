"""Experiment scopes: the frozen Temporal_Training_Scope a strict run samples in.

ensure_scope creates a scope on first use; every later preparation and every
streamed run verifies its header (ready, source, split seed and split shares) and the
scope.unowned rule its membership was created with. TigerGraphScopeReader reads the
scope's accounts.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import math
from typing import Any

from ..config import ScopeConfig
from ..contract.server import (
    CREATE_SCOPE_QUERY,
    FINALIZE_SCOPE_QUERY,
    POPULATION_QUERY,
    SCOPE_POLICY_QUERY,
    SCOPE_VERTEX,
)
from ..runtime.progress import emit
from .executor import (
    ConnectionExecutor,
    QueryExecutor,
    account_pages,
    checked_rows,
    merged_rows,
    printed,
)


class TigerGraphScopeReader:
    """The ScopeReader of data.ports: the scope population query, paged by account."""

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


# The split shares a scope vertex records, in the order of ScopeConfig.shares.
SHARE_ATTRIBUTES = ("train_share", "validation_share", "test_share")


def check_scope(
    attrs: dict[str, Any] | None,
    scope_id: str,
    *,
    source_id: str,
    split_seed: int,
    shares: tuple[float, float, float],
) -> None:
    """Refuse a missing scope, one not ready, or one of another source or partition.

    The partition is the split seed and the train, validation and test shares. A scope
    vertex of the earlier schema records no shares, so it is refused as another
    partition: its accounts were split 70, 15 and 15%.
    """
    if attrs is None:
        raise ValueError(f"Prepared experiment scope is missing: {scope_id}")
    if not attrs["ready"] or attrs["source_id"] != source_id or attrs["split_seed"] != split_seed:
        raise ValueError("Scope is incomplete or belongs to a different source/partition")
    recorded = [attrs.get(name) for name in SHARE_ATTRIBUTES]
    if not all(
        isinstance(have, int | float) and math.isclose(have, want)
        for have, want in zip(recorded, shares, strict=True)
    ):
        raise ValueError(
            f"Scope {scope_id!r} records the split shares {recorded}, not {list(shares)}: it "
            "belongs to a different partition. Set a new scope.id; the next `mule train` "
            "creates it."
        )


# Unowned member Accounts by membership class and side, as the scope policy query prints them.
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
            "(mule install)"
        )
    return {name: int(merged[name]) for name in names}


def inferred_scope_policy(counts: dict[str, int]) -> str | None:
    """The scope.unowned rule a scope was created with, from its membership classes.

    The scope vertex stores no rule, so this is the check that an existing scope was
    built with the configured one (check_scope_policy). Under every rule an unowned
    internal customer account is never shared and an unowned external account is never
    linked. Unowned bank ledger accounts (the bank's own "gl" books) are shared exactly
    when external accounts are: all of them under "shared" and "linked", none under
    "independent". "independent": nothing shared or linked. "shared": every unowned
    external account shared, nothing linked. "linked": every unowned external account
    shared (possibly none exist) and at least one internal account linked to its sole
    owned deposit counterparty. None: no rule produces this membership (for example
    internal customer accounts shared). Rules that wrote identical membership read as
    the simplest of them: "independent" without unowned external accounts or links,
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
            "in config.ScopeConfig; the next `mule train` creates it."
        )
    raise ValueError(
        f"Scope {scope_id!r} was created with scope.unowned = {stored!r}, but the "
        f"configuration says {configured!r} (unowned member classes "
        f"{json.dumps(counts, sort_keys=True)}). Set scope.unowned = {stored!r} to use this "
        "scope, or set a new scope.id; the next `mule train` creates it."
    )


def verify_scope(
    executor: ConnectionExecutor,
    scope_id: str,
    *,
    unowned: str,
    source_id: str,
    split_seed: int,
    shares: tuple[float, float, float],
) -> dict[str, int]:
    """Header (ready, source, split seed, shares) and scope.unowned rule of an existing scope."""
    check_scope(
        scope_header(executor, scope_id),
        scope_id,
        source_id=source_id,
        split_seed=split_seed,
        shares=shares,
    )
    counts = scope_policy_counts(executor, scope_id)
    check_scope_policy(counts, scope_id, unowned)
    return counts


def ensure_scope(
    executor: ConnectionExecutor, scope: ScopeConfig, *, source_id: str, split_seed: int
) -> None:
    """Use the frozen scope, creating it on first use unless scope.create is false.

    Creation writes a Temporal_Training_Scope vertex and one membership edge per
    Account and Party, recording the source id, and the split seed and scope.train_share,
    scope.validation_share and scope.test_share that partition it.
    scope.unowned decides the accounts without an owning Party:
    - "independent": each is its own ownership group with a hashed partition.
    - "shared": unowned external accounts and unowned bank ledger ("gl") accounts
      are visible in every phase (partition 1, group "shared:<component>"); other
      unowned internal accounts stay independent.
    - "linked" (the default): as "shared", and an unowned internal account whose
      only owned internal deposit counterparty is one account joins that
      account's ownership group and partition.
    An existing scope must have been created with the configured rule; it is
    inferred from the membership (the scope policy query) and a mismatch raises.
    """
    scope_id = scope.id

    def verified() -> dict[str, int]:
        return verify_scope(
            executor,
            scope_id,
            unowned=scope.unowned,
            source_id=source_id,
            split_seed=split_seed,
            shares=scope.shares,
        )

    attrs = scope_header(executor, scope_id)
    if attrs is not None:
        counts = verified()
        emit({"event": "scope", "scope": scope_id, "unowned_members": counts})
        return
    if not scope.create:
        raise ValueError(
            f"Scope {scope_id!r} does not exist on TigerGraph and scope.create is false; "
            "set it to true to let the first run create it"
        )
    emit({"event": "scope", "scope": scope_id, "creating": True})
    created = checked_rows(
        executor.run(
            CREATE_SCOPE_QUERY,
            {
                "scope_id": scope_id,
                "source_id": source_id,
                "split_seed": split_seed,
                **dict(zip(SHARE_ATTRIBUTES, scope.shares, strict=True)),
                "unowned_policy": scope.unowned,
            },
            timeout_s=3600.0,
            attempts=1,
        )
    )
    expected = printed(created, "expected_members")
    checked_rows(
        executor.run(
            FINALIZE_SCOPE_QUERY,
            {"scope_id": scope_id, "expected_members": expected},
            timeout_s=3600.0,
            attempts=1,
        )
    )
    counts = verified()
    emit({"event": "scope", "scope": scope_id, "created": True, "unowned_members": counts})
