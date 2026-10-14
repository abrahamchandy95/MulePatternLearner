"""Request parameters and response parsing of the queries, on responses written by hand.

The responses follow docs/design.md, "Response format": the reference graph must produce
exactly them, and a live fetch_subgraphs is checked against it.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.tigergraph.queries import (
    SUBGRAPH_VERSION,
    AccountNode,
    LinkEdge,
    PaymentEdge,
    Phase,
    PopulationAccount,
    SentEdge,
    SplitWritten,
    SubgraphRequest,
    Supervision,
    parse_population,
    parse_split,
    parse_subgraphs,
    parse_supervision,
    population_params,
    split_params,
    subgraph_params,
    supervision_params,
)

DAY = 86_400_000
T = 1_719_792_000_000  # 2024-07-01 00:00 UTC

# Seed A1 paid B2 by Zelle and was paid back; B2, a counterparty, paid the external C3.
# A1's owner P1 is enrolled on D1, and so is E5's owner; A1 sent from I1. A2 is new.
SUBGRAPHS: list[dict[str, Any]] = [
    {"status": "ok", "version": SUBGRAPH_VERSION, "as_of_ms": T, "held_out_visible": False},
    {
        "seed": "A1",
        "accounts": [
            {
                "id": "A1",
                "account_type": "deposit",
                "is_external": False,
                "age_ms": 480 * DAY,
                "token_present": True,
                "token_age_ms": 30 * DAY,
                "in_7d": 1,
                "in_30d": 2,
                "in_365d": 14,
                "out_7d": 1,
                "out_30d": 2,
                "out_365d": 9,
                "amount_in_30d": 1800.0,
                "amount_out_30d": 1750.5,
                "counterparties_in_30d": 1,
                "counterparties_out_30d": 1,
                "first_inflows_7d": 0,
                "first_inflows_30d": 1,
                "passed_on_30d": 900,
                "last_in_present": True,
                "since_last_in_ms": 2 * DAY - 3_600_000,
                "last_out_present": True,
                "since_last_out_ms": DAY,
                "closed": False,
            },
            {
                "id": "B2",
                "account_type": "deposit",
                "is_external": False,
                "age_ms": 200 * DAY,
                "token_present": False,
                "token_age_ms": 0,
                "in_7d": 1,
                "in_30d": 1,
                "in_365d": 1,
                "out_7d": 2,
                "out_30d": 2,
                "out_365d": 2,
                "amount_in_30d": 875.25,
                "amount_out_30d": 1720.0,
                "counterparties_in_30d": 1,
                "counterparties_out_30d": 2,
                "first_inflows_7d": 1,
                "first_inflows_30d": 1,
                "passed_on_30d": 875.25,
                "last_in_present": True,
                "since_last_in_ms": 2 * DAY,
                "last_out_present": True,
                "since_last_out_ms": 2 * DAY - 3_600_000,
                "closed": False,
            },
            {
                "id": "C3",
                "account_type": "unknown",
                "is_external": True,
                "age_ms": 3 * DAY,
                "token_present": False,
                "token_age_ms": 0,
                "in_7d": 1,
                "in_30d": 1,
                "in_365d": 1,
                "out_7d": 0,
                "out_30d": 0,
                "out_365d": 0,
                "amount_in_30d": 820.0,
                "amount_out_30d": 0,
                "counterparties_in_30d": 1,
                "counterparties_out_30d": 0,
                "first_inflows_7d": 1,
                "first_inflows_30d": 1,
                "passed_on_30d": 0,
                "last_in_present": True,
                "since_last_in_ms": 3 * DAY,
                "last_out_present": False,
                "since_last_out_ms": 0,
                "closed": False,
            },
            {
                "id": "E5",
                "account_type": "deposit",
                "is_external": False,
                "age_ms": 90 * DAY,
                "token_present": False,
                "token_age_ms": 0,
                "in_7d": 0,
                "in_30d": 0,
                "in_365d": 0,
                "out_7d": 0,
                "out_30d": 0,
                "out_365d": 0,
                "amount_in_30d": 0,
                "amount_out_30d": 0,
                "counterparties_in_30d": 0,
                "counterparties_out_30d": 0,
                "first_inflows_7d": 0,
                "first_inflows_30d": 0,
                "passed_on_30d": 0,
                "last_in_present": False,
                "since_last_in_ms": 0,
                "last_out_present": False,
                "since_last_out_ms": 0,
                "closed": False,
            },
        ],
        "parties": [{"id": "P1", "age_ms": 480 * DAY, "open_accounts": 1, "closed_accounts": 0}],
        "devices": [
            {
                "id": "D1",
                "age_ms": 400 * DAY,
                "parties_now": 1,
                "parties_ever": 2,
                "accounts": 2,
                "new_parties_30d": 0,
                "observed_only": False,
                "hub": False,
            }
        ],
        "ips": [
            {
                "id": "I1",
                "age_ms": 20 * DAY,
                "parties_now": 0,
                "parties_ever": 0,
                "accounts": 1,
                "new_parties_30d": 0,
                "observed_only": True,
                "hub": False,
            }
        ],
        "addresses": [],
        "pays": [
            {
                "id": "T1",
                "from_id": "A1",
                "to_id": "B2",
                "amount": 875.25,
                "rail": "zelle",
                "age_ms": 2 * DAY,
                "gap_present": False,
                "gap_ms": 0,
                "pair_age_ms": 0,
                "prior_count": 0,
                "first_in_pair": True,
                "next_outflow_present": True,
                "next_outflow_ms": 3_600_000,
            },
            {
                "id": "T2",
                "from_id": "B2",
                "to_id": "A1",
                "amount": 900.0,
                "rail": "ach",
                "age_ms": 2 * DAY - 3_600_000,
                "gap_present": False,
                "gap_ms": 0,
                "pair_age_ms": 0,
                "prior_count": 0,
                "first_in_pair": True,
                "next_outflow_present": True,
                "next_outflow_ms": DAY - 3_600_000,
            },
            {
                "id": "T3",
                "from_id": "B2",
                "to_id": "C3",
                "amount": 820.0,
                "rail": "card",
                "age_ms": 3 * DAY,
                "gap_present": False,
                "gap_ms": 0,
                "pair_age_ms": 0,
                "prior_count": 0,
                "first_in_pair": True,
                "next_outflow_present": False,
                "next_outflow_ms": 0,
            },
        ],
        "owned_by": [
            {
                "from_id": "A1",
                "to_id": "P1",
                "since_start_ms": 480 * DAY,
                "active": True,
                "since_end_ms": 0,
                "held_ms": 480 * DAY,
            }
        ],
        "enrolled_on_device": [
            {
                "from_id": "A1",
                "to_id": "D1",
                "since_start_ms": 400 * DAY,
                "active": True,
                "since_end_ms": 0,
                "held_ms": 400 * DAY,
            },
            {
                "from_id": "E5",
                "to_id": "D1",
                "since_start_ms": 60 * DAY,
                "active": False,
                "since_end_ms": 10 * DAY,
                "held_ms": 50 * DAY,
            },
        ],
        "enrolled_on_ip": [],
        "at_address": [],
        "sent_from_device": [],
        "sent_from_ip": [
            {
                "from_id": "A1",
                "to_id": "I1",
                "since_first_ms": 20 * DAY,
                "since_last_ms": DAY,
                "uses": 4,
            }
        ],
    },
    {
        "seed": "A2",
        "accounts": [
            {
                "id": "A2",
                "account_type": "deposit",
                "is_external": False,
                "age_ms": 0,
                "token_present": False,
                "token_age_ms": 0,
                "in_7d": 0,
                "in_30d": 0,
                "in_365d": 0,
                "out_7d": 0,
                "out_30d": 0,
                "out_365d": 0,
                "amount_in_30d": 0,
                "amount_out_30d": 0,
                "counterparties_in_30d": 0,
                "counterparties_out_30d": 0,
                "first_inflows_7d": 0,
                "first_inflows_30d": 0,
                "passed_on_30d": 0,
                "last_in_present": False,
                "since_last_in_ms": 0,
                "last_out_present": False,
                "since_last_out_ms": 0,
                "closed": False,
            }
        ],
        "parties": [{"id": "P2", "age_ms": 0, "open_accounts": 1, "closed_accounts": 0}],
        "devices": [],
        "ips": [],
        "addresses": [],
        "pays": [],
        "owned_by": [
            {
                "from_id": "A2",
                "to_id": "P2",
                "since_start_ms": 0,
                "active": True,
                "since_end_ms": 0,
                "held_ms": 0,
            }
        ],
        "enrolled_on_device": [],
        "enrolled_on_ip": [],
        "at_address": [],
        "sent_from_device": [],
        "sent_from_ip": [],
    },
]
REQUEST = SubgraphRequest(("A1", "A2"), T, Phase.TRAINING)


def _response() -> list[dict[str, Any]]:
    return copy.deepcopy(SUBGRAPHS)


def test_a_subgraph_response_parses_into_one_subgraph_per_seed() -> None:
    batch = parse_subgraphs(_response(), REQUEST)
    assert (batch.version, batch.as_of_ms, batch.phase) == (SUBGRAPH_VERSION, T, Phase.TRAINING)
    first, second = batch.subgraphs
    assert (first.seed, second.seed) == ("A1", "A2")
    assert [account.id for account in first.accounts] == ["A1", "B2", "C3", "E5"]
    seed = first.accounts[0]
    assert isinstance(seed, AccountNode) and seed.passed_on_30d == 900.0
    assert isinstance(seed.passed_on_30d, float) and seed.token_age_ms == 30 * DAY
    assert first.pays[0] == PaymentEdge(
        "T1", "A1", "B2", 875.25, "zelle", 2 * DAY, False, 0, 0, 0, True, True, 3_600_000
    )
    assert first.enrolled_on_device[1] == LinkEdge("E5", "D1", 60 * DAY, False, 10 * DAY, 50 * DAY)
    assert first.sent_from_ip == (SentEdge("A1", "I1", 20 * DAY, DAY, 4),)
    assert first.ips[0].observed_only and not first.devices[0].hub
    assert second.pays == () and second.parties[0].id == "P2"


def test_a_response_of_another_version_is_refused() -> None:
    rows = _response()
    rows[0]["version"] = "subgraphs-000000000000"
    with pytest.raises(ValueError, match="mule install"):
        parse_subgraphs(rows, REQUEST)


def test_a_response_to_another_request_is_refused() -> None:
    with pytest.raises(ValueError, match="one subgraph per seed"):
        parse_subgraphs(_response(), replace(REQUEST, seed_ids=("A2", "A1")))
    with pytest.raises(ValueError, match="as-of time or phase"):
        parse_subgraphs(_response(), replace(REQUEST, phase=Phase.TEST))
    with pytest.raises(ValueError, match="as-of time or phase"):
        parse_subgraphs(_response(), replace(REQUEST, as_of_ms=T + DAY))


def test_a_refused_request_raises_with_its_status() -> None:
    rows = [{"status": "unknown_seed", "seeds": ["A9"]}]
    with pytest.raises(ValueError, match="unknown_seed"):
        parse_subgraphs(rows, REQUEST)


# Changes to a seed's row of the response, and what the refusal says.
MALFORMED: list[tuple[Callable[[dict[str, Any]], object], str]] = [
    (lambda row: row["pays"][0].update(age_ms="5"), "age_ms"),
    (lambda row: row["pays"][0].update(gap_ms=-1), "gap_ms"),
    (lambda row: row["pays"][0].update(first_in_pair=1), "first_in_pair"),
    (lambda row: row["devices"][0].update(colour="red"), "fields"),
    (lambda row: row["owned_by"][0].pop("held_ms"), "fields"),
    (lambda row: row.pop("sent_from_ip"), "fields"),
    (lambda row: row.update(ips=None), "not a list"),
    (lambda row: row["pays"][2].update(to_id="Z9"), "without its nodes"),
    (lambda row: row["sent_from_ip"][0].update(to_id="D1"), "without its nodes"),
    (lambda row: row["accounts"].append(row["accounts"][1]), "repeats"),
    (lambda row: row["accounts"].pop(0), "does not hold its seed"),
]


@pytest.mark.parametrize(("change", "message"), MALFORMED)
def test_a_malformed_subgraph_is_refused(
    change: Callable[[dict[str, Any]], object], message: str
) -> None:
    rows = _response()
    change(rows[1])
    with pytest.raises(ValueError, match=message):
        parse_subgraphs(rows, REQUEST)


def test_subgraph_parameters_are_the_request_and_every_setting() -> None:
    params = subgraph_params(REQUEST, DEFAULT_CONFIG.subgraph)
    assert params == {
        "seed_ids": ["A1", "A2"],
        "as_of_ms": T,
        "held_out_visible": False,
        "seed_payments": 32,
        "other_payments": 16,
        "owners": 1,
        "siblings": 8,
        "enrolled_devices": 8,
        "enrolled_ips": 8,
        "addresses": 4,
        "sent_devices": 8,
        "sent_ips": 8,
        "reverse_accounts": 16,
        "identity_accounts": 48,
        "hub_threshold": 200,
        "lookback_days": 365,
    }
    test = replace(REQUEST, phase=Phase.TEST)
    assert subgraph_params(test, DEFAULT_CONFIG.subgraph)["held_out_visible"] is True


@pytest.mark.parametrize(
    ("seeds", "as_of_ms"), [((), T), (("A1", "A1"), T), (("A1",), 0), (("A1",), -T)]
)
def test_a_request_needs_distinct_seeds_and_a_time(seeds: tuple[str, ...], as_of_ms: int) -> None:
    with pytest.raises(ValueError, match="subgraph request"):
        SubgraphRequest(seeds, as_of_ms, Phase.TRAINING)


def test_population_pages_parse_into_customer_accounts() -> None:
    assert population_params(T, Phase.TRAINING) == {"as_of_ms": T, "held_out_visible": False}
    page = [
        {"account_id": "A1", "account_type": "deposit", "opened_ms": T - 480 * DAY, "split": 0},
        {"account_id": "A3", "account_type": "credit", "opened_ms": T - DAY, "split": 1},
    ]
    assert parse_population(page) == [
        PopulationAccount("A1", "deposit", T - 480 * DAY, 0),
        PopulationAccount("A3", "credit", T - DAY, 1),
    ]
    for split in (-1, 2):
        with pytest.raises(ValueError, match="split"):
            parse_population([{**page[0], "split": split}])
    with pytest.raises(ValueError, match="fields"):
        parse_population([{**page[0], "is_mule": 1}])


def test_supervision_parses_and_refuses_a_mule_discovered_after_the_cutoff() -> None:
    assert supervision_params(T, Phase.TEST) == {"cutoff_ms": T, "held_out_visible": True}
    found = {
        "account_id": "A7",
        "discovered_ms": T - 5 * DAY,
        "discovery_channel": "victim_report",
        "mule_onset_ms": T - 40 * DAY,
    }
    rows = [{"status": "ok"}, {"accounts": [{"v_id": "A7", "attributes": found}]}]
    assert parse_supervision(rows, T) == [
        Supervision("A7", T - 5 * DAY, "victim_report", T - 40 * DAY)
    ]
    with pytest.raises(ValueError, match="not discovered in time"):
        parse_supervision(rows, T - 5 * DAY)
    unknown = [{"status": "ok"}, {"accounts": [{**found, "discovery_channel": "luck"}]}]
    with pytest.raises(ValueError, match="luck"):
        parse_supervision(unknown, T)
    with pytest.raises(ValueError, match="invalid_parameters"):
        parse_supervision([{"status": "invalid_parameters"}], T)


def test_split_parameters_and_its_answer() -> None:
    assert split_params({"P1": 0, "P2": 1}) == {"party_ids": ["P1", "P2"], "splits": [0, 1]}
    for splits in ({}, {"P1": 2}):
        with pytest.raises(ValueError, match="split 0 or 1"):
            split_params(splits)
    rows = [{"status": "ok", "parties": 2, "accounts": 5}]
    assert parse_split(rows) == SplitWritten(2, 5)
    with pytest.raises(ValueError, match="unknown_party"):
        parse_split([{"status": "unknown_party"}])


def test_only_the_test_phase_sees_held_out_customers() -> None:
    assert not Phase.TRAINING.held_out_visible and Phase.TEST.held_out_visible
