"""The CPU mirror of the label reveal job."""

from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.reference import label_reveal
from mule_pattern_learner.testing.builders import reveal_inputs
from mule_pattern_learner.tigergraph import reveal as tigergraph_reveal

# The built-in run's scope and split dates.
BUILT_IN = (DEFAULT_CONFIG.scope, DEFAULT_CONFIG.dataset.dates)
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
    params = {
        **tigergraph_reveal.reveal_parameters(*BUILT_IN, apply=False),
        **CERTAIN,
        "salt": salt,
    }
    result = label_reveal.plan(reveal_inputs(), {**params, "budget": 1})
    mules = result["mules"]
    assert {name: mules[name]["channel"] for name in "ABD"} == {
        "A": "victim_report",
        "B": "network_trace",
        "D": "victim_report",
    }
    assert mules["C"]["t"] == label_reveal.NEVER
    # E has no split, so it is never eligible, however early it was found.
    assert mules["E"]["t"] < label_reveal.NEVER and result["eligible"] == {"A", "B", "D"}
    assert label_reveal.counts_by_split(result, "eligible") == {1: 2, 2: 0, 3: 1}
    # One of the two train mules fits the budget; the test split's only mule is revealed.
    assert len(result["revealed"] & {"A", "B"}) == 1 and "D" in result["revealed"]
    everyone = label_reveal.plan(reveal_inputs(), {**params, "budget": 20})
    assert everyone["revealed"] == {"A", "B", "D"}
    # A zero budget (schema-valid) reveals nothing and keeps the eligible set.
    none = label_reveal.plan(reveal_inputs(), {**params, "budget": 0})
    assert none["revealed"] == set() and none["eligible"] == {"A", "B", "D"}


def test_reveal_model_uses_the_query_defaults_and_monitoring() -> None:
    params = tigergraph_reveal.reveal_parameters(*BUILT_IN, apply=False)
    model = {key: tigergraph_reveal.REVEAL_DEFAULTS[key] for key in CERTAIN}
    implicit = label_reveal.plan(reveal_inputs(), params)
    explicit = label_reveal.plan(reveal_inputs(), {**params, **model})
    assert implicit["mules"] == explicit["mules"] and implicit["revealed"] == explicit["revealed"]
    # A high proactive hazard finds every mule shortly after its first observation.
    watched = label_reveal.plan(reveal_inputs(), {**params, "proactive_per_day": 1000.0})
    assert {m["channel"] for m in watched["mules"].values()} == {"monitoring"}
    day = 86_400_000
    mule = {"t": 5.5 * day, "first": 0}
    assert label_reveal.available_ms(mule, 10 * day) == 6 * day - 1
    assert label_reveal.available_ms(mule, 5 * day + 7) == 5 * day + 7
    assert label_reveal.available_ms({**mule, "first": 8 * day}, 10 * day) == 8 * day


def dry_run_of(result: dict[str, Any], data_end: int = 10**13) -> dict[str, Any]:
    """What the installed job prints with apply = FALSE when it agrees with the mirror."""
    mules = result["mules"]
    eligible = label_reveal.counts_by_split(result, "eligible")
    return {
        "status": "dry_run",
        "data_end_ts_ms": data_end,
        "eligible": {str(part): n for part, n in eligible.items() if n},
        "revealed_mules": [
            {
                "account_id": k,
                "channel": mules[k]["channel"],
                "known_ts_ms": label_reveal.available_ms(mules[k], data_end),
            }
            for k in sorted(result["revealed"])
        ],
    }


def test_a_dry_run_is_compared_with_the_mirror_mule_by_mule() -> None:
    params = tigergraph_reveal.reveal_parameters(*BUILT_IN, apply=False)
    expected = label_reveal.plan(reveal_inputs(), params)
    agreeing = dry_run_of(expected)
    assert expected["revealed"] and label_reveal.dry_run_differences(expected, agreeing) == []
    first, *rest = agreeing["revealed_mules"]
    missing = {**agreeing, "revealed_mules": rest}
    assert label_reveal.dry_run_differences(expected, missing)[0].startswith("revealed sets")
    moved = {**agreeing, "revealed_mules": [{**first, "known_ts_ms": 1}, *rest]}
    assert label_reveal.dry_run_differences(expected, moved) == [
        f"{first['account_id']}: GSQL {first['channel']} 1, Python {first['channel']} "
        f"{first['known_ts_ms']}"
    ]
    counted = {**agreeing, "eligible": {}}
    assert label_reveal.dry_run_differences(expected, counted)[0].startswith("eligible counts")
