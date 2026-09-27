"""Experiment scopes: creation, the unowned-account rule and the policy query."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.config import ScopeConfig
from mule_pattern_learner.contract.server import (
    CREATE_SCOPE_QUERY,
    FINALIZE_SCOPE_QUERY,
    SCOPE_POLICY_QUERY,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.builders import SNAPSHOT_SOURCE, scope_population, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, Runner, policy_counts
from mule_pattern_learner.tigergraph import gsql_text, scope


def ensure(server: FakeTigerGraph, config: ScopeConfig) -> None:
    """ensure_scope for the unit snapshot and the built-in split seed."""
    scope.ensure_scope(server, config, source_id=SNAPSHOT_SOURCE, split_seed=42)


def test_missing_scope_is_created_unless_forbidden() -> None:
    config = unit_config().scope
    # A graph of three accounts without the scope; the fake refuses a write that could
    # run twice, so the creation queries run with one attempt.
    server = FakeTigerGraph(scope_policy="independent", population=scope_population(3))
    with pytest.raises(ValueError, match="scope.create is false"):
        ensure(server, replace(config, create=False))
    assert not server.calls
    ensure(server, config)
    assert server.names() == [CREATE_SCOPE_QUERY, FINALIZE_SCOPE_QUERY, SCOPE_POLICY_QUERY]
    _, create = server.calls[0]
    assert create["unowned_policy"] == "linked"
    assert (create["source_id"], create["split_seed"]) == (SNAPSHOT_SOURCE, 42)
    assert "shared_unowned" not in create
    assert server.calls[1][1] == {"scope_id": "unit_scope", "expected_members": 3}
    assert server.scopes["unit_scope"] == {
        "ready": True,
        "source_id": SNAPSHOT_SOURCE,
        "split_seed": 42,
    }
    for policy in ("independent", "shared"):
        server = FakeTigerGraph(scope_policy="independent", population=scope_population(3))
        ensure(server, replace(config, unowned=policy))
        assert server.calls[0][1]["unowned_policy"] == policy


def existing(header: dict[str, Any], policy: str) -> FakeTigerGraph:
    """A graph whose unit scope has this header and was created under this rule."""
    return FakeTigerGraph(scope_policy=policy, scopes={"unit_scope": header})


def test_existing_scope_must_have_the_configured_unowned_policy() -> None:
    header = {"ready": True, "source_id": SNAPSHOT_SOURCE, "split_seed": 42}
    config = unit_config().scope
    for policy in ("independent", "shared", "linked"):
        assert scope.inferred_scope_policy(policy_counts(policy)) == policy
        server = FakeTigerGraph(scope_policy=policy, scopes={"unit_scope": header})
        ensure(server, replace(config, unowned=policy))
        assert server.names() == [SCOPE_POLICY_QUERY]
    assert scope.inferred_scope_policy(policy_counts("retired")) is None
    # Without unowned external accounts a linked scope is still recognised by its links.
    no_external = {**policy_counts("linked"), "shared_external": 0, "independent_external": 0}
    assert scope.inferred_scope_policy(no_external) == "linked"
    alone = {**no_external, "linked_internal": 0, "shared_ledger": 0}
    assert scope.inferred_scope_policy(alone) == "independent"
    # Bank ledger accounts are shared exactly when external accounts are: all or none.
    assert scope.inferred_scope_policy({**alone, "shared_ledger": 4}) == "shared"
    for ledger_shared in (0, 3):  # a linked scope that left some ledger books partitioned
        mixed = {**policy_counts("linked"), "shared_ledger": ledger_shared}
        assert scope.inferred_scope_policy(mixed) is None
    # Linking without sharing the external accounts is no rule.
    unshared = {**policy_counts("linked"), "shared_external": 0, "independent_external": 3}
    assert scope.inferred_scope_policy(unshared) is None
    partly = {**policy_counts("shared"), "independent_external": 1}
    assert scope.inferred_scope_policy(partly) is None
    # A pre-policy scope (strict_mule_v1) under the default "linked" configuration.
    with pytest.raises(ValueError, match=r"created with scope.unowned = 'independent'.*set a new"):
        ensure(existing(header, "independent"), config)
    with pytest.raises(ValueError, match="matches no scope.unowned rule"):
        ensure(existing(header, "retired"), config)
    with pytest.raises(ValueError, match="different source"):
        ensure(existing({**header, "split_seed": 7}, "linked"), config)
    with pytest.raises(ValueError, match="different source"):
        ensure(existing({**header, "source_id": "another"}, "linked"), config)
    with pytest.raises(ValueError, match="lacks"):
        scope.scope_policy_counts(Runner(lambda n, p: [{"status": "ok", "members": 3}]), "s")


def test_scope_policy_query_prints_what_the_client_reads() -> None:
    text = (GSQL_DIR / "queries/training_scope.gsql").read_text()
    queries = gsql_text.definitions(text)
    assert SCOPE_POLICY_QUERY in queries
    query = queries[SCOPE_POLICY_QUERY]
    assert gsql_text.parameter_names(query) == {"scope_id"}
    for name in (*scope.SCOPE_POLICY_COUNTS, "members"):
        assert f"AS {name}" in query, name
    create = gsql_text.parameter_names(queries[CREATE_SCOPE_QUERY])
    assert "unowned_policy" in create and "shared_unowned" not in create
