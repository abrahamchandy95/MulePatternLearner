"""The label reveal: its parameters, first-run and later behaviour, and query files."""

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, ScopeConfig, SplitDates
from mule_pattern_learner.contract.bounds import REVEAL_PER_SPLIT
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.contract.server import (
    LABEL_CONTRACT_QUERY,
    REVEAL_QUERY,
    REVEAL_UNIFORMS_QUERY,
    TRAINING_QUERY_FILES,
    TRUTH_QUERY,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph import labels as tigergraph_labels
from mule_pattern_learner.tigergraph import reveal as tigergraph_reveal
from mule_pattern_learner.tigergraph.gsql_text import repository_queries

REVEAL_FILE = GSQL_DIR / "queries/label_reveal.gsql"
# The built-in run's scope and split dates, which the reveal reads.
REVEAL_SETTINGS = (DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates)
CLEAN = {
    "known_labels": 752623,
    "true_mules": 233,
    "revealed_positives": 54,
    **dict.fromkeys(tigergraph_labels.VIOLATIONS, 0),
}


def test_reveal_parameters_follow_the_run_dates_budget_and_salt() -> None:
    scope, dates = DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates
    params = tigergraph_reveal.reveal_parameters(scope, dates, apply=True)
    # The built-in run reveals every discovered mule: the most the reveal takes.
    assert scope.reveal_per_split is None
    assert params == {
        "scope_id": scope.id,
        "train_cutoff_ms": timestamp("2024-07-01"),
        "validation_cutoff_ms": timestamp("2024-10-01"),
        "test_cutoff_ms": timestamp("2025-01-01"),
        "budget": REVEAL_PER_SPLIT.high,
        "salt": 42,
        "apply": True,
    }
    several = SplitDates(train=("2024-05-01", "2024-07-01"))
    assert tigergraph_reveal.reveal_parameters(scope, several, apply=False)[
        "train_cutoff_ms"
    ] == timestamp("2024-07-01")
    other = ScopeConfig(reveal_salt=7, reveal_per_split=5)
    assert tigergraph_reveal.reveal_parameters(other, dates, apply=False) | {"apply": True} == (
        params | {"salt": 7, "budget": 5}
    )


class RevealServer:
    """The reveal answering each of its runs in turn, and the label-contract check."""

    def __init__(self, reveal: dict[str, Any] | list[dict[str, Any]], audit: dict[str, Any]):
        self.reveals = list(reveal) if isinstance(reveal, list) else [reveal]
        self.audit = audit
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        if name == REVEAL_QUERY:
            return [self.reveals.pop(0), {"revealed_mules": []}]
        assert name == LABEL_CONTRACT_QUERY
        return [self.audit]


def revealed_now(eligible: dict[str, int], revealed: dict[str, int], budget: int) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "reveal_v1",
        "budget": budget,
        "reveal_record": f"phantomledger_role;reveal_v1;salt=42;budget={budget}",
        "mules": {"1": 160, "2": 33, "3": 40},
        "eligible": eligible,
        "revealed": revealed,
        "eligible_by_channel": {"victim_report": 50, "monitoring": 15, "network_trace": 8},
        "revealed_by_channel": {"victim_report": 40, "monitoring": 9, "network_trace": 5},
    }


def test_first_run_reveals_every_discovered_mule(capsys: pytest.CaptureFixture[str]) -> None:
    found = {"1": 36, "2": 14, "3": 23}
    server = RevealServer(revealed_now(found, found, REVEAL_PER_SPLIT.high), CLEAN)
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS)
    (name, params, options), _ = server.calls
    assert name == REVEAL_QUERY and params["apply"] is True and options["attempts"] == 1
    assert "force" not in params
    assert summary["labels"] == "revealed now" and summary["revealed"] == found
    assert "shortfall_discovered_by_cutoff" not in summary
    assert capsys.readouterr().out == (
        "Revealed 73 known mules: 36 / 14 / 23 in train / validation / test\n"
    )
    # A split with more discovered mules than the reveal takes is refused, never capped.
    over = {"1": 1200, "2": 14, "3": 23}
    capped = {"1": 1000, "2": 14, "3": 23}
    server = RevealServer(revealed_now(over, capped, REVEAL_PER_SPLIT.high), CLEAN)
    with pytest.raises(ValueError, match="More mules were discovered.*'train': 1200"):
        tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS)


def test_a_capped_reveal_reports_the_shortfall(capsys: pytest.CaptureFixture[str]) -> None:
    eligible, revealed = {"1": 36, "2": 14, "3": 23}, {"1": 20, "2": 14, "3": 20}
    server = RevealServer(revealed_now(eligible, revealed, 20), CLEAN)
    scope = replace(DEFAULT_CONFIG.scope, reveal_per_split=20)
    summary = tigergraph_reveal.ensure_revealed_labels(server, scope, DEFAULT_CONFIG.dataset.dates)
    assert server.calls[0][1]["budget"] == 20
    assert summary["labels"] == "revealed now" and summary["revealed"] == revealed
    # Validation had only 14 mules a bank would have found by 1 October: never padded.
    assert summary["shortfall_discovered_by_cutoff"] == {"validation": 14}
    assert summary["contract"]["revealed_positives"] == 54
    assert capsys.readouterr().out == (
        "Revealed 54 known mules: 20 / 14 / 20 in train / validation / test; fewer than the "
        "budget were discovered by the cutoff of validation\n"
    )


def test_labels_of_another_reveal_are_revealed_again(capsys: pytest.CaptureFixture[str]) -> None:
    other = {
        "status": "revealed_differently",
        "known_labels": 9,
        "revealed_labels": 3,
        "other_reveal": 101,
        "reveal_record": "phantomledger_role;reveal_v1;salt=42;budget=1000",
    }
    found = {"1": 36, "2": 14, "3": 23}
    server = RevealServer([other, revealed_now(found, found, REVEAL_PER_SPLIT.high)], CLEAN)
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS)
    (_, first, _), (_, second, options), _ = server.calls
    assert "force" not in first and second["force"] is True and options["attempts"] == 1
    assert summary["labels"] == "revealed again" and summary["replaced"]["other_reveal"] == 101
    assert capsys.readouterr().out == (
        "Revealed 73 known mules: 36 / 14 / 23 in train / validation / test, replacing the "
        "labels of another reveal\n"
    )


def test_a_run_refuses_labels_its_dataset_did_not_read() -> None:
    scope, dates = REVEAL_SETTINGS
    current = RevealServer({"status": "already_revealed", "other_reveal": 0}, CLEAN)
    tigergraph_reveal.verify_labels(current, scope, dates)
    ((name, params, options),) = current.calls
    # A dry run: it writes nothing, and answers from the label sources alone.
    assert name == REVEAL_QUERY and params["apply"] is False and "force" not in params
    assert options["attempts"] == 1
    other = {"status": "revealed_differently", "other_reveal": 101, "reveal_record": "r"}
    with pytest.raises(ValueError, match="101 mules have another reveal's label"):
        tigergraph_reveal.verify_labels(RevealServer(other, CLEAN), scope, dates)
    with pytest.raises(ValueError, match="no revealed labels"):
        tigergraph_reveal.verify_labels(RevealServer({"status": "dry_run"}, CLEAN), scope, dates)


def test_existing_labels_are_kept_and_contract_violations_fail() -> None:
    kept = RevealServer(
        {"status": "already_revealed", "known_labels": 9, "revealed_labels": 3}, CLEAN
    )
    assert (
        tigergraph_reveal.ensure_revealed_labels(kept, *REVEAL_SETTINGS)["labels"]
        == "already revealed"
    )
    broken = RevealServer({"status": "already_revealed"}, {**CLEAN, "invalid_clocks": 2})
    with pytest.raises(ValueError, match="invalid_clocks"):
        tigergraph_reveal.ensure_revealed_labels(broken, *REVEAL_SETTINGS)
    refused = RevealServer({"status": "scope_not_ready"}, CLEAN)
    with pytest.raises(ValueError, match="scope_not_ready"):
        tigergraph_reveal.ensure_revealed_labels(refused, *REVEAL_SETTINGS)


def test_reveal_queries_are_installed_with_training_and_read_truth_only_there() -> None:
    queries = repository_queries(TRAINING_QUERY_FILES)
    assert {REVEAL_UNIFORMS_QUERY, REVEAL_QUERY} <= set(queries)
    assert {TRUTH_QUERY, LABEL_CONTRACT_QUERY} <= set(queries)
    text = REVEAL_FILE.read_text()
    # The reveal simulates report delays; it never treats the instant oracle as a report.
    assert "z.label_available_ts_ms + (report_days + notify_days)" in text
    # Each mule's label source records the reveal and its budget, which tells labels of
    # another reveal from the configured one's.
    assert '+ ";budget=" + to_string(budget)' in text
    assert "a.mule_label_source = reveal_record" in text
    assert 'status = "revealed_differently"' in text
    # Feature and preparation queries never read ground truth.
    for relative in (
        "queries/training_context.gsql",
        "queries/hub_accounts.gsql",
        "queries/split_cutoffs.gsql",
    ):
        assert "is_mule" not in (GSQL_DIR / relative).read_text()
