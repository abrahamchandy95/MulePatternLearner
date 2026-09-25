"""The Account label contract and its one-time reveal: columns, parameters, mirror, wrapper."""

from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

from mule_pattern_learner.temporal.common import timestamp
from mule_pattern_learner.temporal.live import labels, reveal_model
from mule_pattern_learner.temporal.live.config_schema import (
    DEFAULT_RUN,
    run_config,
    validate_config,
)
from mule_pattern_learner.temporal.live.installation import (
    TRAINING_QUERY_FILES,
    definitions,
    repository_queries,
)
from temporal_fakes import reveal_inputs

ROOT = Path(__file__).resolve().parents[2]
REVEAL_FILE = ROOT / "gsql/temporal/label_reveal.gsql"


def test_hash_mirror_is_pinned_uniform_and_stream_independent() -> None:
    # Pinned: the GSQL temporal_reveal_uniforms must return exactly these values.
    assert labels.reveal_uniforms(123456789, 42, 3) == [
        0.8522561180182994,
        0.339975114371616,
        0.2213301638706262,
    ]
    assert labels.reveal_uniforms(1, 1042, 1) == [0.856376556379896]
    draws = np.array([labels.reveal_uniforms(k, 42, 2) for k in range(60_000_000, 60_020_000)])
    assert ((draws > 0) & (draws < 1)).all()
    assert abs(draws.mean() - 0.5) < 0.01 and abs(draws.var() - 1 / 12) < 0.003
    assert abs(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]) < 0.03
    assert abs(np.corrcoef(draws[:-1, 0], draws[1:, 0])[0, 1]) < 0.03
    # Every intermediate product stays inside a signed 64-bit integer, as in GSQL.
    assert (2147483647 - 1) ** 2 + 1013904223 < 2**63


def test_reveal_parameters_follow_the_run_dates_budget_and_seed() -> None:
    config = run_config()
    params = labels.reveal_parameters(config, apply=True)
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
    assert labels.reveal_parameters(several, apply=False)["train_cutoff_ms"] == timestamp(
        "2024-07-01"
    )
    assert labels.reveal_parameters({**config, "reveal_salt": 7}, apply=False)["salt"] == 7
    # A JSON override may write null; it means "not set", like cohort_seed.
    unset = validate_config({**config, "reveal_salt": None, "reveal_per_split": None, "seed": 5})
    assert labels.reveal_parameters(unset, apply=False)["salt"] == 5
    assert labels.reveal_parameters(unset, apply=False)["budget"] == 20


def test_reveal_defaults_are_the_query_defaults() -> None:
    query = definitions(REVEAL_FILE.read_text())[labels.REVEAL_QUERY]
    header = query.split("(", 1)[1].split(") FOR GRAPH", 1)[0]
    declared = re.findall(r"\b(?:INT|DOUBLE)\s+(\w+)\s*=\s*([-\d.]+)", header)
    assert {name: float(value) for name, value in declared} == labels.REVEAL_DEFAULTS
    assert DEFAULT_RUN["reveal_per_split"] == labels.REVEAL_DEFAULTS["budget"]


# Every mule is reported, acted on and traced; there is no proactive discovery.
CERTAIN = {
    "p_report": 1.0,
    "p_action_first": 1.0,
    "p_action_later": 1.0,
    "proactive_per_day": 0.0,
    "trace_probability": 1.0,
}


@pytest.mark.parametrize("salt", [1, 2])
def test_reveal_model_finds_reports_and_traces_and_reveals_within_budget(salt: int) -> None:
    params = {**labels.reveal_parameters(run_config(), apply=False), **CERTAIN, "salt": salt}
    result = reveal_model.plan(reveal_inputs(), {**params, "budget": 1})
    mules = result["mules"]
    assert {name: mules[name]["channel"] for name in "ABD"} == {
        "A": "victim_report",
        "B": "network_trace",
        "D": "victim_report",
    }
    assert mules["C"]["t"] == reveal_model.NEVER
    # E has no split, so it is never eligible, however early it was found.
    assert mules["E"]["t"] < reveal_model.NEVER and result["eligible"] == {"A", "B", "D"}
    assert reveal_model.counts_by_split(result, "eligible") == {1: 2, 2: 0, 3: 1}
    # One of the two train mules fits the budget; the test split's only mule is revealed.
    assert len(result["revealed"] & {"A", "B"}) == 1 and "D" in result["revealed"]
    everyone = reveal_model.plan(reveal_inputs(), {**params, "budget": 20})
    assert everyone["revealed"] == {"A", "B", "D"}
    # A zero budget (schema-valid) reveals nothing and keeps the eligible set.
    none = reveal_model.plan(reveal_inputs(), {**params, "budget": 0})
    assert none["revealed"] == set() and none["eligible"] == {"A", "B", "D"}


def test_reveal_model_uses_the_query_defaults_and_monitoring() -> None:
    params = labels.reveal_parameters(run_config(), apply=False)
    model = {key: labels.REVEAL_DEFAULTS[key] for key in CERTAIN}
    implicit = reveal_model.plan(reveal_inputs(), params)
    explicit = reveal_model.plan(reveal_inputs(), {**params, **model})
    assert implicit["mules"] == explicit["mules"] and implicit["revealed"] == explicit["revealed"]
    # A high proactive hazard finds every mule shortly after its first observation.
    watched = reveal_model.plan(reveal_inputs(), {**params, "proactive_per_day": 1000.0})
    assert {m["channel"] for m in watched["mules"].values()} == {"monitoring"}
    day = 86_400_000
    mule = {"t": 5.5 * day, "first": 0}
    assert reveal_model.available_ms(mule, 10 * day) == 6 * day - 1
    assert reveal_model.available_ms(mule, 5 * day + 7) == 5 * day + 7
    assert reveal_model.available_ms({**mule, "first": 8 * day}, 10 * day) == 8 * day


class RevealServer:
    def __init__(self, reveal: dict[str, Any], audit: dict[str, Any]) -> None:
        self.reveal, self.audit = reveal, audit
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def run(self, name: str, params: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((name, params, kwargs))
        if name == labels.REVEAL_QUERY:
            return [self.reveal, {"revealed_mules": []}]
        assert name == labels.VALIDATE_QUERY
        return [self.audit]


CLEAN = {
    "known_labels": 752623,
    "true_mules": 233,
    "revealed_positives": 54,
    **dict.fromkeys(labels.VIOLATIONS, 0),
}


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
    summary = labels.ensure_revealed_labels(server, run_config())
    name, params, options = server.calls[0]
    assert name == labels.REVEAL_QUERY and params["apply"] is True and options["attempts"] == 1
    assert summary["labels"] == "revealed now" and summary["revealed"] == reveal["revealed"]
    # Validation had only 14 mules a bank would have found by 1 October: never padded.
    assert summary["shortfall_discovered_by_cutoff"] == {"validation": 14}
    assert summary["contract"]["revealed_positives"] == 54
    assert "revealed now" in capsys.readouterr().out


def test_existing_labels_are_kept_and_contract_violations_fail() -> None:
    kept = RevealServer(
        {"status": "already_revealed", "known_labels": 9, "revealed_labels": 3}, CLEAN
    )
    assert labels.ensure_revealed_labels(kept, run_config())["labels"] == "already revealed"
    broken = RevealServer({"status": "already_revealed"}, {**CLEAN, "invalid_clocks": 2})
    with pytest.raises(ValueError, match="invalid_clocks"):
        labels.ensure_revealed_labels(broken, run_config())
    refused = RevealServer({"status": "scope_not_ready"}, CLEAN)
    with pytest.raises(ValueError, match="scope_not_ready"):
        labels.ensure_revealed_labels(refused, run_config())
    with pytest.raises(ValueError, match="strict_inductive"):
        labels.ensure_revealed_labels(
            refused, {**run_config(), "evaluation_protocol": "shared_history"}
        )


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
        "gsql/temporal/training_context.gsql",
        "gsql/temporal/hub_registry.gsql",
        "gsql/temporal/training_cutoffs.gsql",
    ):
        assert "is_mule" not in (ROOT / relative).read_text()


def test_account_schema_contract_matches_canonical_ddl() -> None:
    ddl = (ROOT / "gsql/schema/temporal_schema.gsql").read_text()
    block = ddl.split("ADD VERTEX Account (", 1)[1].split(") WITH", 1)[0]
    fields = re.findall(
        r"^\s*(?:PRIMARY_ID )?(\w+)\s+(?:STRING|BOOL|UINT|INT)", block, re.MULTILINE
    )
    assert fields == labels.ACCOUNT_STORAGE_COLUMNS
    assert re.search(r"is_mule INT DEFAULT 0", block)
    loader = (ROOT / "gsql/schema/temporal_account_loading.gsql").read_text()
    columns = re.findall(r'\$"(\w+)"', loader)
    assert columns == labels.ACCOUNT_STORAGE_COLUMNS
    header = loader.split("DEFINE HEADER account_header =", 1)[1].split(";", 1)[0]
    assert re.findall(r'"(\w+)"', header) == labels.ACCOUNT_LOAD_COLUMNS
