"""The console: the line of each retry, and how lines write counts and messages."""

from __future__ import annotations

import pytest

from mule_pattern_learner.runtime.console import brief, plural, retry_line, show


@pytest.mark.parametrize(
    "failure, operation, attempt, wait, reason, line",
    [
        (
            "availability",
            "connect",
            2,
            5.8,
            "starting workspace, HTTP 502",
            "TigerGraph is not answering yet (starting workspace, HTTP 502): attempt 2, "
            "retrying in 6 s",
        ),
        (
            "server_timeout",
            "fetch_subgraphs (64 keys)",
            1,
            3.2,
            "query timeout exceeded",
            "TigerGraph timed out on fetch_subgraphs, 64 keys (query timeout exceeded): "
            "attempt 1, retrying in 3 s",
        ),
        # A count of one reads in the singular.
        (
            "server_timeout",
            "fetch_subgraphs (1 key)",
            1,
            3.2,
            "query timeout exceeded",
            "TigerGraph timed out on fetch_subgraphs, 1 key (query timeout exceeded): "
            "attempt 1, retrying in 3 s",
        ),
        (
            "deterministic",
            "CREATE QUERY fetch_subgraphs",
            1,
            4.4,
            "runtime error: out of memory",
            "TigerGraph failed CREATE QUERY fetch_subgraphs (runtime error: out of memory): "
            "attempt 1, retrying in 4 s",
        ),
        (
            "availability",
            "connect",
            3,
            120.0,
            "",
            "TigerGraph is not answering yet (no answer): attempt 3, retrying in 2.0 min",
        ),
    ],
)
def test_each_retry_has_its_short_line(
    failure: str, operation: str, attempt: int, wait: float, reason: str, line: str
) -> None:
    assert retry_line(failure, operation, attempt, wait, reason) == line


def test_lines_write_counts_and_messages_for_a_person(capsys: pytest.CaptureFixture[str]) -> None:
    assert (plural(1, "query", "queries"), plural(1200, "query", "queries")) == (
        "1 query",
        "1,200 queries",
    )
    assert plural(2, "attempt") == "2 attempts"
    assert brief("a  short\n message") == "a short message"
    assert brief("word " * 30, 20) == "word word word wo..."
    show("one line")
    show("two\nlines")
    assert capsys.readouterr().out == "one line\ntwo\nlines\n"
