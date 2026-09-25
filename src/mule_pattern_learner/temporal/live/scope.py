"""Experiment scopes: the frozen Temporal_Training_Scope a strict run samples in.

ensure_scope creates a scope on first use; every later preparation and every
streamed run verifies its header (ready, source, split seed) and the scope_unowned
rule its membership was created with.
"""

from __future__ import annotations

import json
from typing import Any

from .config_schema import OPERATIONAL_DEFAULTS, split_seed
from .executor import QueryExecutor, checked_rows, connection_call, merged_rows, printed


def scope_header(executor: Any, scope_id: str) -> dict[str, Any] | None:
    """Attributes of the Temporal_Training_Scope vertex, or None when it does not exist."""
    from pyTigerGraph.common.exception import TigerGraphException

    try:
        rows = connection_call(
            executor,
            "getVerticesById",
            lambda conn: conn.getVerticesById("Temporal_Training_Scope", [scope_id]),
        )
    except TigerGraphException as error:
        if str(error.code) != "601":
            raise
        return None
    if not isinstance(rows, list) or len(rows) > 1:
        raise ValueError("Unexpected scope metadata response")
    return dict(rows[0]["attributes"]) if rows else None


def check_scope(attrs: dict[str, Any] | None, config: dict[str, Any]) -> None:
    if attrs is None:
        raise ValueError(f"Prepared experiment scope is missing: {config['scope_id']}")
    if (
        not attrs["ready"]
        or attrs["source_id"] != config["dataset_id"]
        or attrs["split_seed"] != split_seed(config)
    ):
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
)


def scope_policy_counts(executor: Any, scope_id: str) -> dict[str, int]:
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
    """The scope_unowned rule a scope was created with, from its membership classes.

    Under every rule an unowned internal account is never shared and an unowned
    external account is never linked. "independent": nothing shared or linked
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
    if counts["independent_external"] and (shared or linked):
        return None  # shared and linked scopes share every unowned external account
    if linked:
        return "linked"
    return "shared" if shared else "independent"


def check_scope_policy(counts: dict[str, int], config: dict[str, Any]) -> str:
    """Raise unless the scope was created with the configured scope_unowned rule."""
    configured = config.get("scope_unowned", OPERATIONAL_DEFAULTS["scope_unowned"])
    stored = inferred_scope_policy(counts)
    if stored is not None and stored == configured:
        return stored
    scope_id = config["scope_id"]
    if stored is None:
        raise ValueError(
            f"Scope {scope_id!r} matches no scope_unowned rule (unowned member classes "
            f"{json.dumps(counts, sort_keys=True)}). Create a new scope: set a new scope_id "
            "(for example in an --config overrides file); the next `mule-temporal train` "
            "(or `prepare`) creates it."
        )
    raise ValueError(
        f"Scope {scope_id!r} was created with scope_unowned = {stored!r}, but the "
        f"configuration says {configured!r} (unowned member classes "
        f"{json.dumps(counts, sort_keys=True)}). Set scope_unowned = {stored!r} to use this "
        "scope, or set a new scope_id; the next `mule-temporal train` (or `prepare`) "
        "creates it."
    )


def verify_scope(executor: Any, config: dict[str, Any]) -> dict[str, int]:
    """Header (ready, source, split seed) and scope_unowned rule of an existing scope."""
    check_scope(scope_header(executor, config["scope_id"]), config)
    counts = scope_policy_counts(executor, config["scope_id"])
    check_scope_policy(counts, config)
    return counts


def ensure_scope(executor: QueryExecutor, config: dict[str, Any]) -> None:
    """Use the frozen scope, creating it on first use unless create_scope = false.

    Creation writes a Temporal_Training_Scope vertex and one membership edge per
    Account and Party. scope_unowned decides the accounts without an owning Party:
    - "independent": each is its own ownership group with a hashed partition.
    - "shared": unowned external accounts are visible in every phase (partition
      1, group "shared:<component>"); unowned internal accounts stay independent.
    - "linked" (the default): as "shared", and an unowned internal account whose
      only owned internal deposit counterparty is one account joins that
      account's ownership group and partition.
    An existing scope must have been created with the configured rule; it is
    inferred from the membership (temporal_scope_policy) and a mismatch raises.
    """
    scope_id = config["scope_id"]
    attrs = scope_header(executor, scope_id)
    if attrs is not None:
        counts = verify_scope(executor, config)
        print(json.dumps({"scope": scope_id, "unowned_members": counts}), flush=True)
        return
    if config.get("create_scope", OPERATIONAL_DEFAULTS["create_scope"]) is not True:
        raise ValueError(
            f"Scope {scope_id!r} does not exist on TigerGraph and create_scope = false; "
            "remove that override to let the first run create it"
        )
    print(json.dumps({"scope": scope_id, "creating": True}), flush=True)
    created = checked_rows(
        executor.run(
            "temporal_create_training_scope",
            {
                "scope_id": scope_id,
                "source_id": config["dataset_id"],
                "split_seed": split_seed(config),
                "unowned_policy": config.get(
                    "scope_unowned", OPERATIONAL_DEFAULTS["scope_unowned"]
                ),
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
    counts = verify_scope(executor, config)
    print(json.dumps({"scope": scope_id, "created": True, "unowned_members": counts}), flush=True)
