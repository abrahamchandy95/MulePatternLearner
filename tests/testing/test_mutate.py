"""The planted-slip harness's mutants, on a query of the test's own. The harness itself runs
the tests in every query review, never inside pytest."""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner.testing import mutate
from mule_pattern_learner.testing.mutate import (
    MUTANTS_PER_QUERY,
    Mutant,
    applied,
    mutants,
    survivors,
)

QUERY = """USE GRAPH Mule_Pattern_Learner

/* Accounts that exist at t < with a visible owner. */
CREATE OR REPLACE QUERY list_items(UINT t_ms, INT page = 10000) FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {
    HeapAccum<Lot>(256, ts DESC, id DESC) @@lots;
    UINT lookback_ms = 7776000000;
    Page = SELECT a FROM Account:a -(Account_Owned_By_Party>:o)- Party:q
      WHERE a.opened_ms < t_ms AND NOT a.is_external AND a.account_type == "deposit"
        AND o.valid_from_ms < t_ms
        AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms)
      ORDER BY a.id ASC LIMIT page;
    PRINT "ok" AS status, "list_items-000000000000" AS version;
    PRINT Page[Page.id AS account_id] AS accounts;
}
"""


def _by_line() -> dict[int, list[Mutant]]:
    found: dict[int, list[Mutant]] = {}
    for mutant in mutants(QUERY, limit=1_000):
        found.setdefault(mutant.line, []).append(mutant)
    return found


def test_each_mutant_changes_one_line_of_code() -> None:
    lines = QUERY.split("\n")
    for mutant in mutants(QUERY, limit=1_000):
        changed = applied(QUERY, mutant).split("\n")
        assert len(changed) == len(lines)
        assert [i for i, (a, b) in enumerate(zip(lines, changed, strict=True)) if a != b] == [
            mutant.line - 1
        ]


def test_comments_the_header_and_the_version_are_never_mutated() -> None:
    by_line = _by_line()
    # Line 3 is a comment with a "<", line 1 the header and line 12 prints the version.
    assert not {1, 2, 3, 12} & set(by_line)


def test_the_mutants_hold_the_planted_slips_of_the_october_17_build() -> None:
    texts = {mutant.text.strip() for mutant in mutants(QUERY, limit=1_000)}
    # A "<=" at t, a link start without its end test and an uncapped heap.
    assert (
        'WHERE a.opened_ms <= t_ms AND NOT a.is_external AND a.account_type == "deposit"' in texts
    )
    assert "" in {mutant.text for mutant in mutants(QUERY, limit=1_000)}
    assert "HeapAccum<Lot>(ts DESC, id DESC) @@lots;" in texts
    # An unprojected PRINT, a label read and a link whose end moves to t itself.
    assert "PRINT Page AS accounts;" in texts
    assert 'WHERE a.opened_ms < t_ms AND NOT a.is_external AND a.pu_label == "deposit"' in texts
    assert "AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms)" in texts
    whats = {mutant.what for mutant in mutants(QUERY, limit=1_000)}
    assert {"drop NOT", "ASC becomes DESC", "256 becomes 257"} <= whats


def test_at_most_the_limit_are_drawn_the_same_way_each_time() -> None:
    every = mutants(QUERY, limit=1_000)
    assert len(every) > 20
    drawn = mutants(QUERY, limit=20)
    assert len(drawn) == 20 and drawn == mutants(QUERY, limit=20)
    assert set(drawn) <= set(every)
    assert [m.line for m in drawn] == sorted(m.line for m in drawn)
    assert MUTANTS_PER_QUERY == 60


def test_survivors_are_the_mutants_the_tests_pass_and_every_file_is_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "queries").mkdir()
    query = tmp_path / "queries" / "list_items.gsql"
    query.write_text(QUERY)
    registry = tmp_path / "__init__.py"
    registry.write_text('ITEMS = Query("list_items", "queries", "list_items-000000000000")\n')
    seen: list[str] = []

    def tests_pass() -> bool:
        # The tests catch a "<=" and nothing else; each mutant is restamped when they run.
        text = query.read_text()
        assert "list_items-000000000000" not in text and "list_items-" in registry.read_text()
        seen.append(text)
        return " <= " not in text

    monkeypatch.setattr(mutate, "GSQL_DIR", tmp_path)
    monkeypatch.setattr(mutate, "REGISTRY_FILE", registry)
    monkeypatch.setattr(mutate, "_tests_pass", tests_pass)
    count, survived = survivors("list_items", "queries/list_items.gsql")
    assert count == len(mutants(QUERY)) == len(seen)
    assert survived and all("<=" not in mutant.text for mutant in survived)
    assert len(survived) < count
    assert query.read_text() == QUERY
    assert (
        registry.read_text()
        == 'ITEMS = Query("list_items", "queries", "list_items-000000000000")\n'
    )
