"""The one-time label reveal: its parameters, first-run behaviour and query files."""

from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, ScopeConfig, SplitDates
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
    assert params == {
        "scope_id": scope.id,
        "train_cutoff_ms": timestamp("2024-07-01"),
        "validation_cutoff_ms": timestamp("2024-10-01"),
        "test_cutoff_ms": timestamp("2025-01-01"),
        "budget": 20,
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
    def __init__(self, reveal: dict[str, Any], audit: dict[str, Any]) -> None:
        self.reveal, self.audit = reveal, audit
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        if name == REVEAL_QUERY:
            return [self.reveal, {"revealed_mules": []}]
        assert name == LABEL_CONTRACT_QUERY
        return [self.audit]


def test_first_run_reveals_once_and_reports_the_shortfall(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reveal = {
        "status": "ok",
        "version": "reveal_v1",
        "budget": 20,
        "mules": {"1": 160, "2": 33, "3": 40},
        "eligible": {"1": 36, "2": 14, "3": 23},
        "revealed": {"1": 20, "2": 14, "3": 20},
        "eligible_by_channel": {"victim_report": 50, "monitoring": 15, "network_trace": 8},
        "revealed_by_channel": {"victim_report": 40, "monitoring": 9, "network_trace": 5},
    }
    server = RevealServer(reveal, CLEAN)
    summary = tigergraph_reveal.ensure_revealed_labels(server, *REVEAL_SETTINGS)
    name, params, options = server.calls[0]
    assert name == REVEAL_QUERY and params["apply"] is True and options["attempts"] == 1
    assert summary["labels"] == "revealed now" and summary["revealed"] == reveal["revealed"]
    # Validation had only 14 mules a bank would have found by 1 October: never padded.
    assert summary["shortfall_discovered_by_cutoff"] == {"validation": 14}
    assert summary["contract"]["revealed_positives"] == 54
    assert "revealed now" in capsys.readouterr().out


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
    # Feature and preparation queries never read ground truth.
    for relative in (
        "queries/training_context.gsql",
        "queries/hub_accounts.gsql",
        "queries/split_cutoffs.gsql",
    ):
        assert "is_mule" not in (GSQL_DIR / relative).read_text()
