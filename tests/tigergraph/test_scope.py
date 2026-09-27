"""Experiment scopes: creation, the unowned-account rule and the policy query."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.testing.builders import unit_config
from mule_pattern_learner.testing.fake_graph import Runner, ScopeServer, policy_counts
from mule_pattern_learner.tigergraph import gsql_text, scope


def test_missing_scope_is_created_unless_forbidden(tmp_path: Path) -> None:
    config = unit_config(tmp_path)
    server = ScopeServer(None, "independent")
    with pytest.raises(ValueError, match="create_scope = false"):
        scope.ensure_scope(cast(Any, server), {**config, "create_scope": False})
    assert not server.calls
    scope.ensure_scope(cast(Any, server), config)
    names = [call[0] for call in server.calls]
    assert names == [
        "temporal_create_training_scope",
        "temporal_finalize_training_scope",
        "temporal_scope_policy",
    ]
    create = server.calls[0]
    assert create[1]["unowned_policy"] == "linked" and create[2]["attempts"] == 1
    assert "shared_unowned" not in create[1]
    assert server.calls[1][1] == {"scope_id": "unit_scope", "expected_members": 3}
    for policy in ("independent", "shared"):
        server = ScopeServer(None, "independent")
        changed = {**config, "scope_unowned": policy}
        scope.ensure_scope(cast(Any, server), changed)
        assert server.calls[0][1]["unowned_policy"] == policy


@pytest.mark.legacy
def test_existing_scope_must_have_the_configured_unowned_policy(tmp_path: Path) -> None:
    header = {"ready": True, "source_id": "unit_snapshot", "split_seed": 42}
    config = unit_config(tmp_path)
    for policy in ("independent", "shared", "linked"):
        assert scope.inferred_scope_policy(policy_counts(policy)) == policy
        server = ScopeServer(header, policy)
        scope.ensure_scope(cast(Any, server), {**config, "scope_unowned": policy})
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
    with pytest.raises(ValueError, match=r"created with scope_unowned = 'independent'.*set a new"):
        scope.ensure_scope(cast(Any, ScopeServer(header, "independent")), config)
    with pytest.raises(ValueError, match="matches no scope_unowned rule"):
        scope.ensure_scope(cast(Any, ScopeServer(header, "retired")), config)
    with pytest.raises(ValueError, match="different source"):
        scope.ensure_scope(cast(Any, ScopeServer({**header, "split_seed": 7}, "linked")), config)
    with pytest.raises(ValueError, match="lacks"):
        scope.scope_policy_counts(Runner(lambda n, p: [{"status": "ok", "members": 3}]), "s")


def test_scope_policy_query_prints_what_the_client_reads() -> None:
    text = (REPOSITORY_ROOT / "gsql/queries/training_scope.gsql").read_text()
    queries = gsql_text.definitions(text)
    assert scope.SCOPE_POLICY_QUERY in queries
    query = queries[scope.SCOPE_POLICY_QUERY]
    assert gsql_text.parameter_names(query) == {"scope_id"}
    for name in (*scope.SCOPE_POLICY_COUNTS, "members"):
        assert f"AS {name}" in query, name
    create = gsql_text.parameter_names(queries["temporal_create_training_scope"])
    assert "unowned_policy" in create and "shared_unowned" not in create
