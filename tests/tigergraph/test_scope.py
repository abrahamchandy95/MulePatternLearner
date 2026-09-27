"""Experiment scopes: creation, the unowned-account rule and the policy query."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from mule_pattern_learner.config import ScopeConfig
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.builders import SNAPSHOT_SOURCE, unit_config
from mule_pattern_learner.testing.fake_graph import Runner, ScopeServer, policy_counts
from mule_pattern_learner.tigergraph import gsql_text, scope


def ensure(server: ScopeServer, config: ScopeConfig) -> None:
    """ensure_scope for the unit snapshot and the built-in split seed."""
    scope.ensure_scope(cast(Any, server), config, source_id=SNAPSHOT_SOURCE, split_seed=42)


def test_missing_scope_is_created_unless_forbidden() -> None:
    config = unit_config().scope
    server = ScopeServer(None, "independent")
    with pytest.raises(ValueError, match="scope.create is false"):
        ensure(server, replace(config, create=False))
    assert not server.calls
    ensure(server, config)
    names = [call[0] for call in server.calls]
    assert names == [
        "temporal_create_training_scope",
        "temporal_finalize_training_scope",
        "temporal_scope_policy",
    ]
    create = server.calls[0]
    assert create[1]["unowned_policy"] == "linked" and create[2]["attempts"] == 1
    assert (create[1]["source_id"], create[1]["split_seed"]) == (SNAPSHOT_SOURCE, 42)
    assert "shared_unowned" not in create[1]
    assert server.calls[1][1] == {"scope_id": "unit_scope", "expected_members": 3}
    for policy in ("independent", "shared"):
        server = ScopeServer(None, "independent")
        ensure(server, replace(config, unowned=policy))
        assert server.calls[0][1]["unowned_policy"] == policy


def test_existing_scope_must_have_the_configured_unowned_policy() -> None:
    header = {"ready": True, "source_id": SNAPSHOT_SOURCE, "split_seed": 42}
    config = unit_config().scope
    for policy in ("independent", "shared", "linked"):
        assert scope.inferred_scope_policy(policy_counts(policy)) == policy
        server = ScopeServer(header, policy)
        ensure(server, replace(config, unowned=policy))
        assert [call[0] for call in server.calls] == ["temporal_scope_policy"]
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
        ensure(ScopeServer(header, "independent"), config)
    with pytest.raises(ValueError, match="matches no scope.unowned rule"):
        ensure(ScopeServer(header, "retired"), config)
    with pytest.raises(ValueError, match="different source"):
        ensure(ScopeServer({**header, "split_seed": 7}, "linked"), config)
    with pytest.raises(ValueError, match="different source"):
        ensure(ScopeServer({**header, "source_id": "another"}, "linked"), config)
    with pytest.raises(ValueError, match="lacks"):
        scope.scope_policy_counts(Runner(lambda n, p: [{"status": "ok", "members": 3}]), "s")


def test_scope_policy_query_prints_what_the_client_reads() -> None:
    text = (GSQL_DIR / "queries/training_scope.gsql").read_text()
    queries = gsql_text.definitions(text)
    assert scope.SCOPE_POLICY_QUERY in queries
    query = queries[scope.SCOPE_POLICY_QUERY]
    assert gsql_text.parameter_names(query) == {"scope_id"}
    for name in (*scope.SCOPE_POLICY_COUNTS, "members"):
        assert f"AS {name}" in query, name
    create = gsql_text.parameter_names(queries["temporal_create_training_scope"])
    assert "unowned_policy" in create and "shared_unowned" not in create
