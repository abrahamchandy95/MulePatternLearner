"""Query text comparison keeps string literals as written."""

from __future__ import annotations


def test_query_comparison_preserves_string_literal_case_and_spacing() -> None:
    from mule_pattern_learner.tigergraph.gsql_text import normalized

    assert normalized('PRINT "USD";') != normalized('PRINT "usd";')
    assert normalized('PRINT "a b";') != normalized('PRINT "ab";')
    assert normalized('PRINT  "USD";') == normalized('print "USD";')
