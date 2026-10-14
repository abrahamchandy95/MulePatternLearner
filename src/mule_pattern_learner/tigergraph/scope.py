"""The graph's one split: the Temporal_Training_Scope vertex a strict run samples in.

A graph holds one train, validation and test split. ensure_scope creates it on first
use; every later preparation and every streamed run verifies its header (ready, split
seed and split shares). TigerGraphScopeReader reads the split's accounts.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Any

from ..config import ScopeConfig
from ..contract.server import (
    CREATE_SCOPE_QUERY,
    FINALIZE_SCOPE_QUERY,
    GRAPH_NAME,
    POPULATION_QUERY,
    SCOPE_VERTEX,
)
from ..runtime.progress import emit
from .executor import ConnectionExecutor, QueryExecutor, account_pages, checked_rows, printed


class TigerGraphScopeReader:
    """The ScopeReader of data.ports: the split's population query, paged by account."""

    def __init__(self, executor: QueryExecutor) -> None:
        self.executor = executor

    def population_pages(self) -> Iterator[list[dict[str, Any]]]:
        """Pages of the split's accounts, with their observed labels."""
        return account_pages(self.executor, POPULATION_QUERY, {})


# How the owner removes the graph's split, by hand: nothing here deletes one. Deleting a
# scope vertex deletes its membership edges with it. Through the GSQL port (14240) or
# TigerGraph Cloud, the RESTPP path starts with /restpp and needs a token.
REMOVE_SPLIT = (
    f"delete the {SCOPE_VERTEX} vertices by hand, which deletes their membership edges "
    f'with them (pyTigerGraph: delVertices("{SCOPE_VERTEX}"); RESTPP: DELETE '
    f"/graph/{GRAPH_NAME}/vertices/{SCOPE_VERTEX}, /restpp/graph/... with a token through "
    "port 14240 or TigerGraph Cloud)"
)


def scope_header(executor: ConnectionExecutor) -> dict[str, Any] | None:
    """Attributes of the graph's one Temporal_Training_Scope vertex; None without one."""
    rows = executor.call(lambda conn: conn.getVertices(SCOPE_VERTEX), what="getVertices")
    if not isinstance(rows, list):
        raise ValueError("Unexpected scope metadata response")
    if len(rows) > 1:
        raise ValueError(
            f"The graph holds {len(rows)} {SCOPE_VERTEX} vertices, not one split: "
            f"{REMOVE_SPLIT}; the next `mule train` creates the split again"
        )
    return dict(rows[0]["attributes"]) if rows else None


# The split shares a scope vertex records, in the order of ScopeConfig.shares.
SHARE_ATTRIBUTES = ("train_share", "validation_share", "test_share")


def check_scope(
    attrs: dict[str, Any] | None, *, split_seed: int, shares: tuple[float, float, float]
) -> None:
    """Refuse a missing split, one not ready, or one of another partition.

    The partition is the split seed and the train, validation and test shares.
    """
    if attrs is None:
        raise ValueError("The graph has no split; the next `mule train` creates it")
    if not attrs["ready"]:
        raise ValueError(
            f"The graph's split is incomplete, its creation never finished: {REMOVE_SPLIT}; "
            "the next `mule train` creates it again"
        )
    recorded = [attrs.get(name) for name in SHARE_ATTRIBUTES]
    if attrs["split_seed"] != split_seed or not all(
        isinstance(have, int | float) and math.isclose(have, want)
        for have, want in zip(recorded, shares, strict=True)
    ):
        raise ValueError(
            f"The graph's split has the split seed {attrs['split_seed']} and the shares "
            f"{recorded}, not {split_seed} and {list(shares)}: set dataset.split_seed and "
            f"the scope shares to the split's, or {REMOVE_SPLIT}; the next `mule train` "
            "creates the configured one"
        )


def verify_scope(
    executor: ConnectionExecutor, *, split_seed: int, shares: tuple[float, float, float]
) -> None:
    """The graph's split must be ready and have this split seed and these shares."""
    check_scope(scope_header(executor), split_seed=split_seed, shares=shares)


def ensure_scope(executor: ConnectionExecutor, scope: ScopeConfig, *, split_seed: int) -> bool:
    """Use the graph's split, creating it when there is none unless scope.create is false.

    Creation writes the Temporal_Training_Scope vertex and one membership edge per
    Account and Party, recording the split seed and scope.train_share,
    scope.validation_share and scope.test_share that partition it. Accounts no Party
    owns follow one rule (gsql/queries/training_scope.gsql): unowned external and bank
    ledger ("gl") accounts are visible in every phase (partition 1), and an unowned
    internal account whose only owned internal deposit counterparty is one account
    joins that account's ownership group and partition. An existing split must have
    the configured split seed and shares. Nothing here deletes a split (REMOVE_SPLIT).
    Whether it created the split.
    """
    if scope_header(executor) is not None:
        verify_scope(executor, split_seed=split_seed, shares=scope.shares)
        emit({"event": "scope", "found": True})
        return False
    if not scope.create:
        raise ValueError(
            "The graph has no split and scope.create is false; set it to true to let the "
            "first run create it"
        )
    emit({"event": "scope", "creating": True})
    created = checked_rows(
        executor.run(
            CREATE_SCOPE_QUERY,
            {"split_seed": split_seed, **dict(zip(SHARE_ATTRIBUTES, scope.shares, strict=True))},
            timeout_s=3600.0,
            attempts=1,
        )
    )
    expected = printed(created, "expected_members")
    checked_rows(
        executor.run(
            FINALIZE_SCOPE_QUERY, {"expected_members": expected}, timeout_s=3600.0, attempts=1
        )
    )
    verify_scope(executor, split_seed=split_seed, shares=scope.shares)
    emit({"event": "scope", "created": True})
    return True
