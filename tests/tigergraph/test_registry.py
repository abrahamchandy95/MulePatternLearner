"""The query registry: the design's twelve queries with their files, the entry point each runs
from, and, for every query written so far, its file, its version and the rules of its text."""

from __future__ import annotations

from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import name_rule, version_rule
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    BENCH_ONLY,
    EVALUATION_ONLY,
    FETCH_ALLOWLIST,
    GRAPH_NAME,
    QUERY_FILES,
    REGISTRY,
    VERSIONS,
    written_files,
)

# The queries of docs/design.md ("The queries"), in its table's order.
DESIGN = (
    "list_accounts",
    "read_labels",
    "read_answer_key",
    "read_hubs",
    "read_world",
    "profile_hubs",
    "read_money_flow",
    "read_counterparties",
    "read_activity_change",
    "read_payment_events",
    "read_shared_identity",
    "read_community",
)


def test_the_registry_holds_the_designs_twelve_queries() -> None:
    assert sorted(REGISTRY) == sorted(DESIGN) and len(DESIGN) == 12
    assert all(query.name == name for name, query in REGISTRY.items())
    for name in DESIGN:
        assert name_rule(name) == [], name


def test_each_query_has_its_file_under_queries_or_the_answer_key_under_evaluation() -> None:
    expected = {name: f"queries/{name}.gsql" for name in DESIGN}
    expected["read_answer_key"] = "evaluation/read_answer_key.gsql"
    assert QUERY_FILES == expected


def test_the_queries_are_for_the_graph_the_schema_creates() -> None:
    schema = (GSQL_DIR / "schema" / "schema.gsql").read_text()
    assert f"CREATE GRAPH {GRAPH_NAME} ()" in schema


def test_each_query_runs_from_one_entry_point_of_the_design() -> None:
    # Fetch runs the population, the labels, the hubs, the world and B1 to B6.
    assert FETCH_ALLOWLIST == {
        "list_accounts",
        "read_labels",
        "read_hubs",
        "read_world",
        "read_money_flow",
        "read_counterparties",
        "read_activity_change",
        "read_payment_events",
        "read_shared_identity",
        "read_community",
    }
    assert BENCH_ONLY == {"profile_hubs"} and EVALUATION_ONLY == {"read_answer_key"}
    # The three are disjoint and hold every query.
    assert not FETCH_ALLOWLIST & BENCH_ONLY and not FETCH_ALLOWLIST & EVALUATION_ONLY
    assert not BENCH_ONLY & EVALUATION_ONLY
    assert FETCH_ALLOWLIST | BENCH_ONLY | EVALUATION_ONLY == set(REGISTRY)


def _gsql_files(folder: str) -> set[str]:
    return {path.relative_to(GSQL_DIR).as_posix() for path in (GSQL_DIR / folder).glob("*.gsql")}


def test_every_query_file_is_registered() -> None:
    files = _gsql_files("queries") | _gsql_files("evaluation")
    assert files == set(written_files().values())
    assert set(written_files().values()) <= set(QUERY_FILES.values())


def test_every_written_query_defines_itself_and_prints_its_registered_version() -> None:
    assert set(VERSIONS) == set(written_files()), "a version needs its file, and a file a version"
    for name, relative in written_files().items():
        defined = definitions((GSQL_DIR / relative).read_text())
        assert list(defined) == [name], relative
        assert version_rule(name, defined[name], VERSIONS[name]) == []
