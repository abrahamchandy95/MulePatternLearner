"""The CPU mirror of the label reveal job."""

import pytest

from mule_pattern_learner.config import run_config
from mule_pattern_learner.reference import label_reveal as reveal_model
from mule_pattern_learner.testing.builders import reveal_inputs
from mule_pattern_learner.tigergraph import reveal as tigergraph_reveal

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
        **tigergraph_reveal.reveal_parameters(run_config(), apply=False),
        **CERTAIN,
        "salt": salt,
    }
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
    params = tigergraph_reveal.reveal_parameters(run_config(), apply=False)
    model = {key: tigergraph_reveal.REVEAL_DEFAULTS[key] for key in CERTAIN}
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
