"""The one-time label reveal: its parameters, defaults and first-run behaviour."""

import re
from typing import Any

import numpy as np
import pytest

from mule_pattern_learner.config import DEFAULT_RUN, run_config, validate_config
from mule_pattern_learner.contract.clock import timestamp
from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.tigergraph import labels as tigergraph_labels
from mule_pattern_learner.tigergraph import reveal as tigergraph_reveal
from mule_pattern_learner.tigergraph.gsql_text import definitions, repository_queries
from mule_pattern_learner.tigergraph.installer import TRAINING_QUERY_FILES

REVEAL_FILE = REPOSITORY_ROOT / "gsql/queries/label_reveal.gsql"
CLEAN = {
    "known_labels": 752623,
    "true_mules": 233,
    "revealed_positives": 54,
    **dict.fromkeys(tigergraph_labels.VIOLATIONS, 0),
}


def test_hash_mirror_is_pinned_uniform_and_stream_independent() -> None:
    # Pinned: the GSQL temporal_reveal_uniforms must return exactly these values.
    assert tigergraph_reveal.reveal_uniforms(123456789, 42, 3) == [
        0.8522561180182994,
        0.339975114371616,
        0.2213301638706262,
    ]
    assert tigergraph_reveal.reveal_uniforms(1, 1042, 1) == [0.856376556379896]
    draws = np.array(
        [tigergraph_reveal.reveal_uniforms(k, 42, 2) for k in range(60_000_000, 60_020_000)]
    )
    assert ((draws > 0) & (draws < 1)).all()
    assert abs(draws.mean() - 0.5) < 0.01 and abs(draws.var() - 1 / 12) < 0.003
    assert abs(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]) < 0.03
    assert abs(np.corrcoef(draws[:-1, 0], draws[1:, 0])[0, 1]) < 0.03
    # Every intermediate product stays inside a signed 64-bit integer, as in GSQL.
    assert (2147483647 - 1) ** 2 + 1013904223 < 2**63


def test_reveal_parameters_follow_the_run_dates_budget_and_seed() -> None:
    config = run_config()
    params = tigergraph_reveal.reveal_parameters(config, apply=True)
    assert params == {
        "scope_id": DEFAULT_RUN["scope_id"],
        "train_cutoff_ms": timestamp("2024-07-01"),
        "validation_cutoff_ms": timestamp("2024-10-01"),
        "test_cutoff_ms": timestamp("2025-01-01"),
        "budget": 20,
        "salt": 42,
        "apply": True,
    }
    several = {**config, "dates": {**config["dates"], "train": ["2024-05-01", "2024-07-01"]}}
    assert tigergraph_reveal.reveal_parameters(several, apply=False)[
        "train_cutoff_ms"
    ] == timestamp("2024-07-01")
    assert (
        tigergraph_reveal.reveal_parameters({**config, "reveal_salt": 7}, apply=False)["salt"] == 7
    )
    # A JSON override may write null; it means "not set", like cohort_seed.
    unset = validate_config({**config, "reveal_salt": None, "reveal_per_split": None, "seed": 5})
    assert tigergraph_reveal.reveal_parameters(unset, apply=False)["salt"] == 5
    assert tigergraph_reveal.reveal_parameters(unset, apply=False)["budget"] == 20


def test_reveal_defaults_are_the_query_defaults() -> None:
    query = definitions(REVEAL_FILE.read_text())[tigergraph_reveal.REVEAL_QUERY]
    header = query.split("(", 1)[1].split(") FOR GRAPH", 1)[0]
    declared = re.findall(r"\b(?:INT|DOUBLE)\s+(\w+)\s*=\s*([-\d.]+)", header)
    assert {name: float(value) for name, value in declared} == tigergraph_reveal.REVEAL_DEFAULTS
    assert DEFAULT_RUN["reveal_per_split"] == tigergraph_reveal.REVEAL_DEFAULTS["budget"]


class RevealServer:
    def __init__(self, reveal: dict[str, Any], audit: dict[str, Any]) -> None:
        self.reveal, self.audit = reveal, audit
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        if name == tigergraph_reveal.REVEAL_QUERY:
            return [self.reveal, {"revealed_mules": []}]
        assert name == tigergraph_labels.VALIDATE_QUERY
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
    summary = tigergraph_reveal.ensure_revealed_labels(server, run_config())
    name, params, options = server.calls[0]
    assert (
        name == tigergraph_reveal.REVEAL_QUERY
        and params["apply"] is True
        and options["attempts"] == 1
    )
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
        tigergraph_reveal.ensure_revealed_labels(kept, run_config())["labels"] == "already revealed"
    )
    broken = RevealServer({"status": "already_revealed"}, {**CLEAN, "invalid_clocks": 2})
    with pytest.raises(ValueError, match="invalid_clocks"):
        tigergraph_reveal.ensure_revealed_labels(broken, run_config())
    refused = RevealServer({"status": "scope_not_ready"}, CLEAN)
    with pytest.raises(ValueError, match="scope_not_ready"):
        tigergraph_reveal.ensure_revealed_labels(refused, run_config())


def test_reveal_queries_are_installed_with_training_and_read_truth_only_there() -> None:
    queries = repository_queries(TRAINING_QUERY_FILES)
    assert {"temporal_reveal_uniforms", "temporal_reveal_mule_labels"} <= set(queries)
    assert {"temporal_get_account_supervision", "temporal_validate_account_supervision"} <= set(
        queries
    )
    text = REVEAL_FILE.read_text()
    # The reveal simulates report delays; it never treats the instant oracle as a report.
    assert "z.label_available_ts_ms + (report_days + notify_days)" in text
    # Feature and preparation queries never read ground truth.
    for relative in (
        "gsql/queries/training_context.gsql",
        "gsql/queries/hub_accounts.gsql",
        "gsql/queries/split_cutoffs.gsql",
    ):
        assert "is_mule" not in (REPOSITORY_ROOT / relative).read_text()
