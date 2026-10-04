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
from mule_pattern_learner.testing.builders import UNIT_SOURCE, scope_population, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, policy_counts, ready_scope
from mule_pattern_learner.tigergraph import gsql_text, scope


def ensure(server: FakeTigerGraph, config: ScopeConfig) -> None:
    """ensure_scope for the unit snapshot and the built-in split seed."""
    scope.ensure_scope(server, config, source_id=UNIT_SOURCE, split_seed=42)


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
    assert (create["source_id"], create["split_seed"]) == (UNIT_SOURCE, 42)
    # The split shares are the scope's settings, sent as the query's parameters.
    shares = (create["train_share"], create["validation_share"], create["test_share"])
    assert shares == config.shares == (0.5, 0.25, 0.25)
    assert "shared_unowned" not in create
    assert server.calls[1][1] == {"scope_id": "unit_scope", "expected_members": 3}
    # The scope vertex records them beside the source and the split seed.
    assert server.scopes["unit_scope"] == {
        "ready": True,
        "source_id": UNIT_SOURCE,
        "split_seed": 42,
        "train_share": 0.5,
        "validation_share": 0.25,
        "test_share": 0.25,
    }
    other = replace(config, train_share=0.7, validation_share=0.15, test_share=0.15)
    server = FakeTigerGraph(scope_policy="independent", population=scope_population(3))
    ensure(server, other)
    assert server.scopes["unit_scope"]["train_share"] == 0.7
    for policy in ("independent", "shared"):
        server = FakeTigerGraph(scope_policy="independent", population=scope_population(3))
        ensure(server, replace(config, unowned=policy))
        assert server.calls[0][1]["unowned_policy"] == policy


def existing(header: dict[str, Any], policy: str) -> FakeTigerGraph:
    """A graph whose unit scope has this header and was created under this rule."""
    return FakeTigerGraph(scope_policy=policy, scopes={"unit_scope": header})


def test_existing_scope_must_have_the_configured_unowned_policy() -> None:
    header = ready_scope(UNIT_SOURCE)
    config = unit_config().scope
    for policy in ("independent", "shared", "linked"):
        assert scope.inferred_scope_policy(policy_counts(policy)) == policy
        server = FakeTigerGraph(scope_policy=policy, scopes={"unit_scope": header})
        ensure(server, replace(config, unowned=policy))
        assert server.names() == [SCOPE_POLICY_QUERY]
    assert scope.inferred_scope_policy(policy_counts("no_rule")) is None
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
    # A scope created with "independent", under the default "linked" configuration.
    with pytest.raises(ValueError, match=r"created with scope.unowned = 'independent'.*set a new"):
        ensure(existing(header, "independent"), config)
    with pytest.raises(ValueError, match="matches no scope.unowned rule"):
        ensure(existing(header, "no_rule"), config)
    with pytest.raises(ValueError, match="different source"):
        ensure(existing({**header, "split_seed": 7}, "linked"), config)
    with pytest.raises(ValueError, match="different source"):
        ensure(existing({**header, "source_id": "another"}, "linked"), config)
    # A scope of other shares, or of the earlier schema that recorded none (its accounts
    # split 70, 15 and 15%), is another partition, never reused.
    earlier = {k: v for k, v in header.items() if not k.endswith("_share")}
    for found in ({**header, "train_share": 0.7, "test_share": 0.05}, earlier):
        with pytest.raises(ValueError, match=r"split shares .* Set a new scope.id"):
            ensure(existing(found, "linked"), config)
    with pytest.raises(ValueError, match="lacks"):
        lacking = FakeTigerGraph(
            answers={SCOPE_POLICY_QUERY: lambda p: [{"status": "ok", "members": 3}]}
        )
        scope.scope_policy_counts(lacking, "s")


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
    assert set(scope.SHARE_ATTRIBUTES) <= create
    # The scope vertex records the shares the query inserts.
    schema = (GSQL_DIR / "schema/scope_vertex.gsql").read_text()
    for name in scope.SHARE_ATTRIBUTES:
        assert f"    {name} DOUBLE,\n" in schema
    assert (
        "INSERT INTO Temporal_Training_Scope VALUES (\n"
        "    scope_id, source_id, split_seed, train_share, validation_share, test_share, FALSE);"
    ) in queries[CREATE_SCOPE_QUERY]
