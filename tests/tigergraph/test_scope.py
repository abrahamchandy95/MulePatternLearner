"""The graph's one split: its creation, its checks and its queries."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import pytest

from mule_pattern_learner.config import ScopeConfig
from mule_pattern_learner.contract.server import (
    CREATE_SCOPE_QUERY,
    FINALIZE_SCOPE_QUERY,
    POPULATION_QUERY,
    SCOPE_VERTEX,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.builders import scope_population, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph, ready_scope
from mule_pattern_learner.testing.gsql_blocks import flat
from mule_pattern_learner.tigergraph import gsql_text, scope


def ensure(server: FakeTigerGraph, config: ScopeConfig) -> bool:
    """ensure_scope with the built-in split seed: whether it created the split."""
    return scope.ensure_scope(server, config, split_seed=42)


def test_missing_split_is_created_unless_forbidden() -> None:
    config = unit_config().scope
    # A graph of three accounts without a split; the fake refuses a write that could run
    # twice, so the creation queries run with one attempt.
    server = FakeTigerGraph(population=scope_population(3))
    with pytest.raises(ValueError, match="no split and scope.create is false"):
        ensure(server, replace(config, create=False))
    assert not server.calls
    assert ensure(server, config) is True
    assert server.names() == [CREATE_SCOPE_QUERY, FINALIZE_SCOPE_QUERY]
    # The split seed and the shares (the scope's settings) are all the creation sends:
    # the graph's one split is named by no id.
    _, create = server.calls[0]
    assert create == {
        "split_seed": 42,
        "train_share": 0.5,
        "validation_share": 0.25,
        "test_share": 0.25,
    }
    assert config.shares == (0.5, 0.25, 0.25)
    assert server.calls[1][1] == {"expected_members": 3}
    # The scope vertex records them beside the split seed.
    assert server.scope == {
        "ready": True,
        "split_seed": 42,
        "train_share": 0.5,
        "validation_share": 0.25,
        "test_share": 0.25,
    }
    # The next run uses the split it finds, reading only its header, and creates none.
    assert ensure(server, config) is False
    assert server.names() == [CREATE_SCOPE_QUERY, FINALIZE_SCOPE_QUERY]
    other = replace(config, train_share=0.7, validation_share=0.15, test_share=0.15)
    server = FakeTigerGraph(population=scope_population(3))
    ensure(server, other)
    assert server.scope is not None and server.scope["train_share"] == 0.7


def test_invalid_split_shares_or_a_second_split_create_nothing() -> None:
    # The fake graph refuses them as create_training_scope does: a share that is not
    # positive, or shares that do not add up to 1.
    params = {"split_seed": 42}
    queries = gsql_text.definitions((GSQL_DIR / "queries/training_scope.gsql").read_text())
    create = flat(queries[CREATE_SCOPE_QUERY])
    assert "IF train_share <= 0 OR validation_share <= 0 OR test_share <= 0" in create
    assert "OR train_share + validation_share + test_share < 0.999999" in create
    assert "OR train_share + validation_share + test_share > 1.000001" in create
    for shares in ((0.5, 0.5, 0.0), (0.6, 0.25, 0.25), (0.5, 0.25, 0.2), (1.2, -0.1, -0.1)):
        server = FakeTigerGraph(population=scope_population(3))
        named = dict(zip(scope.SHARE_ATTRIBUTES, shares, strict=True))
        rows = server.run(CREATE_SCOPE_QUERY, {**params, **named}, attempts=1)
        assert rows == [{"status": "invalid_parameters"}] and server.scope is None, shares
    named = dict(zip(scope.SHARE_ATTRIBUTES, (0.5, 0.25, 0.25), strict=True))
    server = FakeTigerGraph(population=scope_population(3))
    assert server.run(CREATE_SCOPE_QUERY, {**params, **named}, attempts=1)[0]["status"] == "ok"
    # A graph holds one split: any scope vertex, ready or not, refuses another.
    assert 'IF Scopes.size() > 0 THEN PRINT "scope_already_exists" AS status;' in create
    for found in (ready_scope(), {**ready_scope(), "ready": False}):
        server = FakeTigerGraph(population=scope_population(3), scope=found)
        rows = server.run(CREATE_SCOPE_QUERY, {**params, **named}, attempts=1)
        assert rows == [{"status": "scope_already_exists"}] and server.scope == found


def existing(header: dict[str, Any]) -> FakeTigerGraph:
    """A graph whose split has this header."""
    return FakeTigerGraph(scope=header)


class SeveralSplits:
    """A connection to a graph that holds two scope vertices, as one of several splits did."""

    def getVertices(self, vertex_type: str) -> list[dict[str, Any]]:
        assert vertex_type == SCOPE_VERTEX
        return [{"v_id": name, "attributes": ready_scope()} for name in ("one", "two")]


def test_an_existing_split_must_be_ready_and_have_the_configured_partition() -> None:
    header = ready_scope()
    config = unit_config().scope
    ensure(existing(header), config)
    # The owner removes a split by hand: every refusal says how, and nothing deletes it.
    removal = f"delete the {SCOPE_VERTEX} vertices by hand"
    unready = existing({**header, "ready": False})
    with pytest.raises(ValueError, match=f"split is incomplete.*{removal}"):
        ensure(unready, config)
    assert unready.scope is not None and not unready.writes
    with pytest.raises(ValueError, match=rf"split seed 7 and the shares .* not 42 .*{removal}"):
        ensure(existing({**header, "split_seed": 7}), config)
    # A split of other shares, or one whose vertex records none, is another partition,
    # never reused.
    unshared = {k: v for k, v in header.items() if not k.endswith("_share")}
    for found in ({**header, "train_share": 0.7, "test_share": 0.05}, unshared):
        with pytest.raises(ValueError, match=r"shares \[.*\], not 42 and \[0.5, 0.25, 0.25\]"):
            ensure(existing(found), config)
    # A graph that holds more than one scope vertex does not hold exactly one split.
    several = existing(header)
    several.client.conn = SeveralSplits()
    with pytest.raises(ValueError, match=f"holds 2 {SCOPE_VERTEX} vertices, not one split"):
        scope.scope_header(several)
    assert scope.scope_header(FakeTigerGraph()) is None


def test_the_scope_queries_name_the_one_split_and_print_what_the_client_reads() -> None:
    text = (GSQL_DIR / "queries/training_scope.gsql").read_text()
    queries = gsql_text.definitions(text)
    assert set(queries) == {CREATE_SCOPE_QUERY, POPULATION_QUERY, FINALIZE_SCOPE_QUERY}
    # No query takes a scope id or a source id: each reads the graph's one split.
    assert gsql_text.parameter_names(queries[FINALIZE_SCOPE_QUERY]) == {"expected_members"}
    create = gsql_text.parameter_names(queries[CREATE_SCOPE_QUERY])
    assert create == {*scope.SHARE_ATTRIBUTES, "split_seed"}
    assert gsql_text.parameter_names(queries[POPULATION_QUERY]) == {"after_id", "batch_size"}
    assert "scope_id" not in text and "source_id" not in text
    population = flat(queries[POPULATION_QUERY])
    assert "ReadyScope = SELECT r FROM Scopes:r WHERE r.ready;" in population
    assert 'IF ReadyScope.size() != 1 THEN PRINT "scope_not_ready"' in population
    finalize = flat(queries[FINALIZE_SCOPE_QUERY])
    assert "IF Scopes.size() != 1 OR expected_members == 0 THEN" in finalize
    assert "IF e.partition < 1 OR e.partition > 3 THEN @@invalid += 1 END;" in finalize
    # Each prints only what the client reads.
    assert 'PRINT "ok" AS status, Entities.size() AS expected_members;' in flat(
        queries[CREATE_SCOPE_QUERY]
    )
    printed = population.split("PRINT Accounts[", 1)[1]
    assert re.findall(r" AS (\w+)", printed) == [
        "account_id",
        "first_seen_ts_ms",
        "partition",
        "observed_positive",
        "known_from_ms",
        "accounts",
    ]
    # The scope vertex records the shares the query inserts, under the split's fixed id,
    # and every membership edge points at it and records only its partition.
    schema = flat((GSQL_DIR / "schema/scope_vertex.gsql").read_text())
    assert "PRIMARY_ID id STRING, split_seed UINT," in schema and "source_id" not in schema
    assert "partition INT DEFAULT 0 ) WITH" in schema and "group_id" not in schema
    for name in scope.SHARE_ATTRIBUTES:
        assert f" {name} DOUBLE," in schema
    insert = flat(queries[CREATE_SCOPE_QUERY])
    assert (
        "INSERT INTO Temporal_Training_Scope VALUES ( "
        '"split", split_seed, train_share, validation_share, test_share, FALSE);'
    ) in insert
    assert insert.count(" Temporal_Training_Scope,") == 1
    assert insert.count('"split" Temporal_Training_Scope,') == 1
