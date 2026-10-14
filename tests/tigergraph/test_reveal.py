"""The label reveal: its parameters, first-run and later behaviour, and query files."""

import re
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, ScopeConfig, SplitDates
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.contract.server import (
    REVEAL_QUERY,
    REVEAL_UNIFORMS_QUERY,
    TRAINING_QUERY_FILES,
    TRUTH_QUERY,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_blocks import flat
from mule_pattern_learner.tigergraph import gsql_text
from mule_pattern_learner.tigergraph import reveal as tigergraph_reveal
from mule_pattern_learner.tigergraph.gsql_text import repository_queries

REVEAL_FILE = GSQL_DIR / "queries/label_reveal.gsql"
# The built-in run's scope settings and split dates, which the reveal reads.
REVEAL_SETTINGS = (DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates)
# The label source every label of the built-in reveal records, before its outcome.
RECORD = "phantomledger_role;reveal_v1;salt=42;budget=1000"


def test_reveal_parameters_follow_the_run_dates_and_salt() -> None:
    scope, dates = DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates
    params = tigergraph_reveal.reveal_parameters(scope, dates, apply=True)
    # The splits are the partitions of the graph's one split, which no parameter names.
    assert params == {
        "train_cutoff_ms": timestamp("2024-07-01"),
        "validation_cutoff_ms": timestamp("2024-10-01"),
        "test_cutoff_ms": timestamp("2025-01-01"),
        "salt": 42,
        "apply": True,
    }
    several = SplitDates(train=("2024-05-01", "2024-07-01"))
    assert tigergraph_reveal.reveal_parameters(scope, several, apply=False)[
        "train_cutoff_ms"
    ] == timestamp("2024-07-01")
    other = ScopeConfig(reveal_salt=7)
    assert tigergraph_reveal.reveal_parameters(other, dates, apply=False) | {"apply": True} == (
        params | {"salt": 7}
    )


class RevealServer:
    """The reveal answering each of its runs in turn."""

    def __init__(self, reveal: dict[str, Any] | list[dict[str, Any]]):
        self.reveals = list(reveal) if isinstance(reveal, list) else [reveal]
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        assert name == REVEAL_QUERY
        return [self.reveals.pop(0), {"revealed_mules": []}]


def revealed_now(revealed: dict[str, int]) -> dict[str, Any]:
    return {"status": "ok", "revealed": revealed, "data_end_ts_ms": 1_735_689_599_999}


def test_first_run_reveals_every_discovered_mule(capsys: pytest.CaptureFixture[str]) -> None:
    found = {"1": 36, "2": 14, "3": 23}
    server = RevealServer(revealed_now(found))
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS)
    ((name, params, options),) = server.calls
    assert name == REVEAL_QUERY and params["apply"] is True and options["attempts"] == 1
    assert "force" not in params
    assert summary == {"labels": "revealed now", "revealed": found}
    assert capsys.readouterr().out == (
        "Revealed 73 known mules: 36 / 14 / 23 in train / validation / test\n"
    )


def test_labels_of_another_reveal_are_revealed_again(capsys: pytest.CaptureFixture[str]) -> None:
    other = {
        "status": "revealed_differently",
        "revealed_labels": 3,
        "other_reveal": 101,
        "reveal_record": RECORD,
    }
    found = {"1": 36, "2": 14, "3": 23}
    server = RevealServer([other, revealed_now(found)])
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS)
    (_, first, _), (_, second, options) = server.calls
    assert "force" not in first and second["force"] is True and options["attempts"] == 1
    assert summary["labels"] == "revealed again" and summary["replaced"]["other_reveal"] == 101
    assert capsys.readouterr().out == (
        "Revealed 73 known mules: 36 / 14 / 23 in train / validation / test, replacing the "
        "labels of another reveal\n"
    )


def test_this_reveals_labels_are_revealed_again_for_a_new_split(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The labels are this reveal's, drawn against the split the owner deleted.
    current = {"status": "already_revealed", "revealed_labels": 98, "other_reveal": 0, "mules": 386}
    found = {"1": 39, "2": 21, "3": 38}
    server = RevealServer([current, revealed_now(found)])
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS, new_split=True)
    (_, first, _), (_, second, options) = server.calls
    assert "force" not in first and second["force"] is True and second["apply"] is True
    assert options["attempts"] == 1
    # The event names the mules revealed per split, which an identical split repeats.
    assert summary == {"labels": tigergraph_reveal.NEW_SPLIT, "revealed": found}
    assert capsys.readouterr().out == (
        "Revealed 98 known mules: 39 / 21 / 38 in train / validation / test, again for the "
        "new split\n"
    )
    # A split found in place keeps them.
    kept = RevealServer(current)
    assert tigergraph_reveal.ensure_revealed_labels(kept, *REVEAL_SETTINGS)["labels"] == (
        "already revealed"
    )
    assert len(kept.calls) == 1


def test_labels_of_another_source_are_kept_for_a_new_split() -> None:
    # Known labels on a graph with no internal mule: the reveal did not write them, and
    # revealing again would overwrite them.
    other = {"status": "already_revealed", "revealed_labels": 40, "other_reveal": 0, "mules": 0}
    server = RevealServer(other)
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS, new_split=True)
    assert summary == {"labels": "already revealed", "revealed_labels": 40}
    ((_, params, _),) = server.calls
    assert "force" not in params


def test_a_run_refuses_labels_its_dataset_did_not_read() -> None:
    scope, dates = REVEAL_SETTINGS
    current = RevealServer({"status": "already_revealed", "other_reveal": 0})
    tigergraph_reveal.verify_labels(current, scope, dates)
    ((name, params, options),) = current.calls
    # A dry run: it writes nothing, and answers from the label sources alone.
    assert name == REVEAL_QUERY and params["apply"] is False and "force" not in params
    assert options["attempts"] == 1
    other = {"status": "revealed_differently", "other_reveal": 101, "reveal_record": "r"}
    with pytest.raises(ValueError, match="101 mules have another reveal's label"):
        tigergraph_reveal.verify_labels(RevealServer(other), scope, dates)
    with pytest.raises(ValueError, match="no revealed labels"):
        tigergraph_reveal.verify_labels(RevealServer({"status": "dry_run"}), scope, dates)


def test_existing_labels_are_kept_and_a_refused_reveal_fails() -> None:
    kept = RevealServer({"status": "already_revealed", "revealed_labels": 3})
    assert tigergraph_reveal.ensure_revealed_labels(kept, *REVEAL_SETTINGS) == {
        "labels": "already revealed",
        "revealed_labels": 3,
    }
    refused = RevealServer({"status": "scope_not_ready"})
    with pytest.raises(ValueError, match="scope_not_ready"):
        tigergraph_reveal.ensure_revealed_labels(refused, *REVEAL_SETTINGS)


def test_reveal_queries_are_installed_with_training_and_read_truth_only_there() -> None:
    queries = repository_queries(TRAINING_QUERY_FILES)
    assert {REVEAL_UNIFORMS_QUERY, REVEAL_QUERY, TRUTH_QUERY} <= set(queries)
    text = REVEAL_FILE.read_text()
    query = gsql_text.definitions(text)[REVEAL_QUERY]
    # The reveal simulates report delays; it never treats the instant oracle as a report.
    assert "z.label_available_ts_ms + (report_days + notify_days)" in text
    # Each mule's label source records the reveal, as the labels on the graph record it,
    # which tells labels of another reveal from this one's.
    assert (
        'STRING reveal_record = "phantomledger_role;reveal_v1;salt=" + to_string(salt) '
        '+ ";budget=1000";'
    ) in flat(query)
    assert "a.mule_label_source = reveal_record" in text
    assert 'status = "revealed_differently"' in text
    # Its splits are the partitions of the graph's one ready split, which nothing names.
    assert "Ready = SELECT r FROM Scopes:r WHERE r.ready;" in text
    assert 'IF Ready.size() != 1 THEN PRINT "scope_not_ready" AS status; RETURN; END;' in text
    # Every mule discovered before its split's cutoff is revealed.
    assert (
        "WHERE a.@part >= 1 AND a.@part <= 3 AND a.@t_known < a.@cutoff "
        "POST-ACCUM a.@revealed += TRUE"
    ) in flat(query)
    # It writes only the label attributes the schema keeps, and prints what a reader reads.
    written = set(re.findall(r"\ba\.(\w+) = ", query))
    assert written == {
        "mule_label_known",
        "pu_label",
        "mule_label_available_ts_ms",
        "mule_label_source",
    }
    # The two refusals before the label check print only their status.
    assert re.findall(r"\bAS (\w+)", query) == [
        "status",
        "status",
        "status",
        "revealed_labels",
        "other_reveal",
        "mules",
        "reveal_record",
        "status",
        "revealed",
        "data_end_ts_ms",
        "revealed_mules",
    ]
    sent = tigergraph_reveal.reveal_parameters(*REVEAL_SETTINGS, apply=False)
    assert set(sent) | {"force"} == gsql_text.parameter_names(queries[REVEAL_QUERY][1])
    # Feature and preparation queries never read ground truth.
    for relative in (
        "queries/training_context.gsql",
        "queries/hub_accounts.gsql",
        "queries/split_cutoffs.gsql",
    ):
        assert "is_mule" not in (GSQL_DIR / relative).read_text()
