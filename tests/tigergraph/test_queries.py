"""Request parameters and response parsing of the queries, on responses written by hand.

The responses follow docs/design.md, "Response format", "Date tables" and "Other queries";
the reference graph must produce exactly them, and a live graph is checked against it.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import fields, replace
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    PROFILE_HUBS,
    READ_LEAVES,
    READ_SHARING,
    READ_SUPERVISION,
    READ_WORLD,
    SUBGRAPH_VERSION,
    VERSIONS,
    AccountNode,
    CommunityRow,
    HubProfile,
    IdentityNode,
    Leaves,
    LinkEdge,
    OwnedByEdge,
    PartyNode,
    PaymentEdge,
    PopulationAccount,
    SentEdge,
    SharingPage,
    SharingRow,
    SubgraphRequest,
    Supervision,
    Work,
    World,
    leaves_params,
    parse_hub_profile,
    parse_leaves,
    parse_population,
    parse_sharing,
    parse_subgraphs,
    parse_supervision,
    parse_world,
    population_params,
    profile_params,
    sharing_params,
    sharing_rows,
    subgraph_params,
    supervision_params,
)

DAY = 86_400_000
HOUR = 3_600_000
T = 1_719_792_000_000  # 2024-07-01 00:00 UTC
CONFIG = DEFAULT_CONFIG.subgraph
LEAVES = Leaves(("H1",), ("D9",), ("D8",), (), ("I9",))


def _row(kind: type, **values: Any) -> dict[str, Any]:
    """A row of kind with every field 0, False or "" but values."""
    blank = {"int": 0, "float": 0.0, "bool": False, "str": ""}
    return {field.name: blank[str(field.type)] for field in fields(kind)} | values


# Seed A1 (owner P1, enrolled on D1 since 40 days, observed on I1) received 900.00 from B2
# by ACH two days before t and passed 875.25 on to B2 by Zelle 30 minutes later; B2 has no
# owner. E5 is another customer's deposit account on D1, whose owner held it from 60 to 10
# days before t. H1 is a hub account that paid A1 its salary. A2 opened yesterday.
SUBGRAPHS: list[dict[str, Any]] = [
    {
        "status": "ok",
        "version": SUBGRAPH_VERSION,
        "as_of_ms": T,
        "work": {"payments_read": 40, "identity_nodes_read": 3, "community_candidates": 1},
    },
    {
        "seed": "A1",
        "accounts": [
            _row(
                AccountNode,
                id="A1",
                owner_id="P1",
                account_type="deposit",
                in_7d=1,
                in_30d=2,
                in_365d=2,
                out_7d=1,
                out_30d=1,
                out_365d=1,
                amount_in_7d=90_000,
                amount_in_30d=390_000,
                amount_out_7d=87_525,
                amount_out_30d=87_525,
                counterparties_in_30d=2,
                counterparties_in_365d=2,
                counterparties_out_30d=1,
                counterparties_out_365d=1,
                new_payers_7d=1,
                new_payers_30d=2,
                new_payees_7d=1,
                new_payees_30d=1,
                burst_in_1h=1,
                burst_in_24h=1,
                burst_out_1h=1,
                burst_out_24h=1,
                burst_new_payers_24h=1,
                passed_on_1h=90_000,
                passed_on_24h=90_000,
                shortest_in_gap_present=True,
                shortest_in_gap_ms=18 * DAY,
                last_in_present=True,
                since_last_in_ms=2 * DAY,
                last_out_present=True,
                since_last_out_ms=2 * DAY - HOUR // 2,
            ),
            _row(AccountNode, id="B2", account_type="deposit", in_30d=1, out_30d=1),
            _row(AccountNode, id="H1", account_type="gl", hub=True),
            _row(AccountNode, id="E5", owner_id="P5", account_type="deposit"),
        ],
        "parties": [_row(PartyNode, id="P1", accounts=1, devices_now=1)],
        "devices": [
            _row(IdentityNode, id="D1", parties_now=1, parties_lookback=2, accounts=2),
            _row(IdentityNode, id="D9", leaf=True),
        ],
        "ips": [
            _row(
                IdentityNode,
                id="I1",
                observed_only=True,
                accounts=1,
                payments_7d=1,
                payments_30d=1,
                senders_30d=1,
            )
        ],
        "community": _row(CommunityRow, parties=1, accounts=1, external_payments_30d=3),
        "pays": [
            _row(
                PaymentEdge,
                id="T1",
                from_id="B2",
                to_id="A1",
                amount=90_000,
                rail="ach",
                age_ms=2 * DAY,
                first_in_pair=True,
                next_outflow_present=True,
                next_outflow_ms=HOUR // 2,
                receiver_gap_present=True,
                receiver_gap_ms=18 * DAY,
            ),
            _row(
                PaymentEdge,
                id="Z2",
                from_id="A1",
                to_id="B2",
                amount=87_525,
                rail="zelle",
                age_ms=2 * DAY - HOUR // 2,
                first_in_pair=True,
            ),
            _row(
                PaymentEdge,
                id="T3",
                from_id="H1",
                to_id="A1",
                amount=300_000,
                rail="ach",
                age_ms=20 * DAY,
                first_in_pair=True,
            ),
        ],
        "owned_by": [{"from_id": "A1", "to_id": "P1"}],
        "enrolled_on_device": [
            _row(
                LinkEdge,
                from_id="A1",
                to_id="D1",
                start_known=True,
                since_start_ms=40 * DAY,
                held_ms=40 * DAY,
                active=True,
            ),
            _row(
                LinkEdge,
                from_id="E5",
                to_id="D1",
                start_known=True,
                since_start_ms=60 * DAY,
                held_ms=50 * DAY,
                since_end_ms=10 * DAY,
            ),
            _row(LinkEdge, from_id="A1", to_id="D9", active=True),
        ],
        "enrolled_on_ip": [],
        "sent_from_device": [],
        "sent_from_ip": [
            _row(
                SentEdge,
                from_id="A1",
                to_id="I1",
                since_first_ms=2 * DAY - HOUR // 2,
                since_last_ms=2 * DAY - HOUR // 2,
                uses=1,
            )
        ],
    },
    {
        "seed": "A2",
        "accounts": [_row(AccountNode, id="A2", owner_id="P2", account_type="credit")],
        "parties": [_row(PartyNode, id="P2", accounts=1)],
        "devices": [],
        "ips": [],
        "community": _row(CommunityRow, parties=1, accounts=1),
        "pays": [],
        "owned_by": [{"from_id": "A2", "to_id": "P2"}],
        "enrolled_on_device": [],
        "enrolled_on_ip": [],
        "sent_from_device": [],
        "sent_from_ip": [],
    },
]
REQUEST = SubgraphRequest(("A1", "A2"), T)


def _response() -> list[dict[str, Any]]:
    return copy.deepcopy(SUBGRAPHS)


def test_a_subgraph_response_parses_into_one_subgraph_per_seed() -> None:
    batch = parse_subgraphs(_response(), REQUEST)
    assert (batch.version, batch.as_of_ms, batch.work) == (SUBGRAPH_VERSION, T, Work(40, 3, 1))
    first, second = batch.subgraphs
    assert (first.seed, second.seed) == ("A1", "A2")
    assert [account.id for account in first.accounts] == ["A1", "B2", "H1", "E5"]
    seed = first.accounts[0]
    assert (seed.owner_id, seed.passed_on_1h, seed.shortest_in_gap_ms) == ("P1", 90_000, 18 * DAY)
    assert first.accounts[2].hub and first.devices[1].leaf and first.ips[0].observed_only
    no_gaps = (False, 0, 0, False, 0, False, 0, False, 0)
    assert first.pays[1] == PaymentEdge(
        "Z2", "A1", "B2", 87_525, "zelle", 2 * DAY - HOUR // 2, 0, True, *no_gaps
    )
    assert first.owned_by == (OwnedByEdge("A1", "P1"),)
    assert first.enrolled_on_device[1] == LinkEdge(
        "E5", "D1", True, 60 * DAY, 50 * DAY, False, 10 * DAY
    )
    assert first.sent_from_ip == (
        SentEdge("A1", "I1", 2 * DAY - HOUR // 2, 2 * DAY - HOUR // 2, 1),
    )
    assert first.community.external_payments_30d == 3
    assert isinstance(first.community.density, float)
    assert second.pays == () and second.parties[0].id == "P2"


def test_subgraph_parameters_are_the_request_its_settings_and_the_leaves() -> None:
    assert subgraph_params(REQUEST, CONFIG, LEAVES) == {
        "seed_ids": ["A1", "A2"],
        "as_of_ms": T,
        "seed_payments": 32,
        "other_payments": 16,
        "owners": 1,
        "siblings": 8,
        "enrolled_devices": 8,
        "enrolled_ips": 8,
        "sent_devices": 8,
        "sent_ips": 8,
        "reverse_accounts": 16,
        "identity_accounts": 48,
        "community_parties": 64,
        "recent_payments": 256,
        "lookback_days": 365,
        "hub_accounts": [{"id": "H1"}],
        "hub_devices": [{"id": "D9"}],
        "crowd_devices": [{"id": "D8"}],
        "hub_ips": [],
        "crowd_ips": [{"id": "I9"}],
    }


@pytest.mark.parametrize(
    ("seeds", "as_of_ms"), [((), T), (("A1", "A1"), T), (("A1",), 0), (("A1",), -T)]
)
def test_a_request_needs_distinct_seeds_and_a_time(seeds: tuple[str, ...], as_of_ms: int) -> None:
    with pytest.raises(ValueError, match="subgraph request"):
        SubgraphRequest(seeds, as_of_ms)


def test_a_response_to_another_request_is_refused() -> None:
    with pytest.raises(ValueError, match="one subgraph per seed"):
        parse_subgraphs(_response(), replace(REQUEST, seed_ids=("A2", "A1")))
    with pytest.raises(ValueError, match="as-of time"):
        parse_subgraphs(_response(), replace(REQUEST, as_of_ms=T + DAY))


def test_a_refused_request_raises_with_its_status() -> None:
    with pytest.raises(ValueError, match="unknown_seed"):
        parse_subgraphs([{"status": "unknown_seed", "seeds": ["A9"]}], REQUEST)
    with pytest.raises(ValueError, match="invisible_seed"):
        parse_subgraphs([{"status": "invisible_seed", "seeds": ["A2"]}], REQUEST)


# Changes to a seed's row of the response, and what the refusal says.
MALFORMED: list[tuple[Callable[[dict[str, Any]], object], str]] = [
    (lambda row: row["pays"][0].update(age_ms="5"), "age_ms"),
    (lambda row: row["pays"][0].update(gap_ms=-1), "gap_ms"),
    (lambda row: row["pays"][0].update(amount=900.0), "amount"),
    (lambda row: row["pays"][0].update(first_in_pair=1), "first_in_pair"),
    (lambda row: row["pays"][0].update(rail="wire"), "rail"),
    (lambda row: row["accounts"][1].update(account_type="loan"), "account type"),
    (lambda row: row["devices"][0].update(colour="red"), "fields"),
    (lambda row: row["enrolled_on_device"][0].pop("held_ms"), "fields"),
    (lambda row: row.pop("sent_from_ip"), "fields"),
    (lambda row: row.pop("community"), "fields"),
    (lambda row: row.update(at_address=[]), "fields"),
    (lambda row: row.update(ips=None), "not a list"),
    (lambda row: row["community"].update(density=1.5), "share"),
    (lambda row: row["community"].update(capped=0), "capped"),
    (lambda row: row["pays"][2].update(to_id="Z9"), "without its nodes"),
    (lambda row: row["owned_by"][0].update(to_id="P9"), "without its nodes"),
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


def test_a_malformed_work_count_is_refused() -> None:
    rows = _response()
    rows[0]["work"] = {"payments_read": 40}
    with pytest.raises(ValueError, match="Work"):
        parse_subgraphs(rows, REQUEST)


LEAVES_ROWS: list[dict[str, Any]] = [
    {
        "status": "ok",
        "version": VERSIONS[READ_LEAVES],
        "hub_accounts": ["H2", "H1"],
        "hub_devices": ["D9"],
        "crowd_devices": ["D8"],
        "hub_ips": [],
        "crowd_ips": ["I9"],
    }
]


def test_the_leaves_parse_sorted() -> None:
    assert leaves_params(T, CONFIG) == {
        "as_of_ms": T,
        "lookback_days": 365,
        "hub_threshold": 200,
        "observed_threshold": 8,
    }
    assert parse_leaves(copy.deepcopy(LEAVES_ROWS)) == Leaves(
        ("H1", "H2"), ("D9",), ("D8",), (), ("I9",)
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"crowd_devices": ["D9"]}, "both a hub and a crowd"),
        ({"hub_ips": ["I1", "I1"]}, "repeats"),
        ({"hub_ips": [7]}, "list of ids"),
        ({"hub_ips": None}, "list of ids"),
        ({"more": []}, "fields"),
    ],
)
def test_malformed_leaves_are_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_leaves([{**LEAVES_ROWS[0], **change}])


def _sharing_page(last_id: str, *rows: SharingRow) -> list[dict[str, Any]]:
    items = [{field.name: getattr(row, field.name) for field in fields(row)} for row in rows]
    return [{"status": "ok", "version": VERSIONS[READ_SHARING], "last_id": last_id, "rows": items}]


class PagedExecutor:
    """Answers read_sharing with scripted pages, recording each request's page."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.requests: list[dict[str, Any]] = []

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        assert name == READ_SHARING
        self.requests.append(params)
        return self.pages.pop(0)


def test_sharing_pages_are_read_until_a_page_holds_no_customer() -> None:
    """The second page's customers have no co-user, so it has no rows but goes on."""
    first = SharingRow("P1", 2, 1, 1, 0)
    third = SharingRow("P7", 0, 3, 0, 3)
    executor = PagedExecutor(
        [
            _sharing_page("P3", first),
            _sharing_page("P5"),
            _sharing_page("P8", third),
            _sharing_page(""),
        ]
    )
    params = sharing_params(T, CONFIG, LEAVES)
    assert params == {
        "as_of_ms": T,
        "lookback_days": 365,
        "hub_devices": [{"id": "D9"}],
        "crowd_devices": [{"id": "D8"}],
        "hub_ips": [],
        "crowd_ips": [{"id": "I9"}],
    }
    assert sharing_rows(executor, params, page_size=3) == [first, third]
    assert [(r["after_id"], r["batch_size"]) for r in executor.requests] == [
        ("", 3),
        ("P3", 3),
        ("P5", 3),
        ("P8", 3),
    ]
    assert parse_sharing(_sharing_page("P3", first), "") == SharingPage("P3", (first,))


@pytest.mark.parametrize(
    ("page", "after_id", "message"),
    [
        (_sharing_page("P3"), "P3", "strictly increasing"),
        (_sharing_page("P3", SharingRow("P4", 1, 0, 0, 0)), "", "outside its page"),
        (_sharing_page("P3", SharingRow("P1", 1, 0, 0, 0)), "P1", "outside its page"),
        (_sharing_page("P3", SharingRow("P2", 0, 0, 0, 0)), "", "invalid co-users"),
        (_sharing_page("P3", SharingRow("P2", 1, 0, 2, 0)), "", "invalid co-users"),
    ],
)
def test_a_malformed_sharing_page_is_refused(
    page: list[dict[str, Any]], after_id: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_sharing(page, after_id)


def test_population_pages_parse_into_owned_accounts() -> None:
    assert population_params(T) == {"as_of_ms": T}
    page = [
        {"account_id": "A1", "owner_id": "P1", "account_type": "deposit", "opened_ms": T - DAY},
        {"account_id": "A3", "owner_id": "P3", "account_type": "credit", "opened_ms": T - 9},
    ]
    assert parse_population(page) == [
        PopulationAccount("A1", "P1", "deposit", T - DAY),
        PopulationAccount("A3", "P3", "credit", T - 9),
    ]
    for change in ({"owner_id": ""}, {"account_type": "loan"}):
        with pytest.raises(ValueError, match="not an owned account"):
            parse_population([{**page[0], **change}])
    with pytest.raises(ValueError, match="fields"):
        parse_population([{**page[0], "is_mule": 1}])
    with pytest.raises(ValueError, match="positive"):
        population_params(0)


def test_supervision_parses_and_refuses_a_mule_discovered_after_its_date() -> None:
    assert supervision_params(T) == {"known_by_ms": T}
    found = {
        "account_id": "A7",
        "discovered_ms": T,
        "discovery_channel": "victim_report",
        "mule_onset_ms": T - 40 * DAY,
        "mule_last_ms": T - 3 * DAY,
    }
    head = {"status": "ok", "version": VERSIONS[READ_SUPERVISION]}
    rows = [head, {"accounts": [{"v_id": "A7", "attributes": found}]}]
    assert parse_supervision(rows, T) == [
        Supervision("A7", T, "victim_report", T - 40 * DAY, T - 3 * DAY)
    ]
    with pytest.raises(ValueError, match="not discovered in time"):
        parse_supervision(rows, T - 1)
    unknown = [head, {"accounts": [{**found, "discovery_channel": "luck"}]}]
    with pytest.raises(ValueError, match="luck"):
        parse_supervision(unknown, T)
    with pytest.raises(ValueError, match="invalid_parameters"):
        parse_supervision([{"status": "invalid_parameters"}], T)
    with pytest.raises(ValueError, match="positive"):
        supervision_params(0)


WORLD_ROWS: list[dict[str, Any]] = [
    {
        "status": "ok",
        "version": VERSIONS[READ_WORLD],
        "vertices": {"Party": 2, "Account": 3},
        "edges": {"Party_Owns_Account": 2, "Account_Owned_By_Party": 2},
        "account_types": {"deposit": 2, "unknown": 1},
        "payment_rails": {"ach": 4},
        "external_accounts": 1,
        "seconds": {"Party.joined_ms": 3_400_000_000},
        "cents": {"Payment_Transaction": 123_456},
        "squared_degrees": {"Party_Owns_Account": 2, "Account_Owned_By_Party": 2},
        "first_joined_ms": 1_672_531_200_000,
        "first_event_ms": 1_672_531_260_000,
        "last_event_ms": 1_735_603_200_000,
    }
]


def test_the_world_parses_its_integers_by_name() -> None:
    world = parse_world(copy.deepcopy(WORLD_ROWS))
    assert isinstance(world, World)
    assert world.vertices == {"Account": 3, "Party": 2}
    assert (world.external_accounts, world.first_joined_ms) == (1, 1_672_531_200_000)
    for change in ({"cents": {"Zelle_Transfer": 1.5}}, {"edges": [1]}, {"last_event_ms": -1}):
        with pytest.raises(ValueError, match="read_world printed"):
            parse_world([{**WORLD_ROWS[0], **change}])


def test_the_hub_profile_has_one_count_per_bucket() -> None:
    buckets = [0, 3, 9, 201]
    assert profile_params(T, CONFIG, buckets) == {
        "as_of_ms": T,
        "lookback_days": 365,
        "buckets": buckets,
    }
    counts = {name: [5, 4, 3, 0] for name in ("accounts", "enrolled_devices")} | {
        name: [1, 1, 0, 0] for name in ("observed_devices", "enrolled_ips", "observed_ips")
    }
    head = {"status": "ok", "version": VERSIONS[PROFILE_HUBS], "buckets": buckets, **counts}
    profile = parse_hub_profile([head], buckets)
    assert profile == HubProfile(
        (0, 3, 9, 201), (5, 4, 3, 0), (5, 4, 3, 0), (1, 1, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0)
    )
    with pytest.raises(ValueError, match="accounts"):
        parse_hub_profile([{**head, "accounts": [5, 4]}], buckets)
    with pytest.raises(ValueError, match="other buckets"):
        parse_hub_profile([head], [0, 3, 9, 200])
    for bad in ([], [-1, 2], [0, 5, 5]):
        with pytest.raises(ValueError, match="ascending"):
            profile_params(T, CONFIG, bad)


# Each query whose parser reads its head, and the parser.
PARSERS: list[tuple[str, Callable[[list[dict[str, Any]]], object]]] = [
    (FETCH_SUBGRAPHS, lambda rows: parse_subgraphs(rows, REQUEST)),
    (READ_LEAVES, parse_leaves),
    (READ_SHARING, lambda rows: parse_sharing(rows, "")),
    (READ_SUPERVISION, lambda rows: parse_supervision(rows, T)),
    (READ_WORLD, parse_world),
    (PROFILE_HUBS, lambda rows: parse_hub_profile(rows, [0])),
]


@pytest.mark.parametrize(("query", "parse"), PARSERS)
def test_a_response_of_another_version_is_refused(
    query: str, parse: Callable[[list[dict[str, Any]]], object]
) -> None:
    other = VERSIONS[query][:-12] + "000000000000"
    with pytest.raises(ValueError, match="mule install"):
        parse([{"status": "ok", "version": other}])
