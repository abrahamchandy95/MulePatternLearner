"""read_counterparties' parameter builder and row parser, on answers written by hand in
TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set as a list
of vertices with their projected fields under "attributes".

The parser takes a row for every account asked for and no other, and refuses the values the
query cannot give.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries import (
    READ_COUNTERPARTIES,
    VERSIONS,
    read_counterparties,
)
from mule_pattern_learner.tigergraph.queries.read_counterparties import Counterparties
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS, QueryRefused

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday


def test_a_request_takes_a_monday_its_accounts_and_the_hubs_each_once() -> None:
    """The accounts go as a list of ids, which the query checks itself; the hubs as a set of
    vertices, in TigerGraph's JSON for SET<VERTEX<Account>>."""
    assert read_counterparties.params(T_MS, ["a_02", "a_01"], ["h_01", "h_02"]) == {
        "t_ms": T_MS,
        "account_ids": ["a_02", "a_01"],
        "hub_accounts": [{"id": "h_01"}, {"id": "h_02"}],
    }
    # A Monday may have no account hub.
    assert read_counterparties.params(T_MS, ["a_01"], [])["hub_accounts"] == []
    with pytest.raises(ValueError, match="Monday"):
        read_counterparties.params(T_MS + DAY_MS, ["a_01"], [])
    with pytest.raises(ValueError, match="repeats an account"):
        read_counterparties.params(T_MS, ["a_01", "a_01"], [])
    with pytest.raises(ValueError, match="at least one account"):
        read_counterparties.params(T_MS, [], [])
    with pytest.raises(ValueError, match="repeats an id"):
        read_counterparties.params(T_MS, ["a_01"], ["h_01", "h_01"])
    with pytest.raises(ValueError, match="holds ids"):
        read_counterparties.params(T_MS, ["a_01"], [""])


# A mule account paid by three new payers in one evening and forwarding to one onward account,
# and a saver with no payment in the last 30 days.
# - a_01's payers: two of this bank and one of another, each first paying in the last 7 days,
#   and a fourth, its landlord's deposit return, paying since long before: 3 new of 4 payers.
#   Of the 10,000 cents received in 30 days, 7,500 came from the three new payers: 0.75. The
#   three first payments fell within 3 hours: max_new_payers_24h 3.
# - Its payees: the onward account (new, 66,200 cents) and a biller it paid before (a hub,
#   3,800 cents): top 66,200 / 70,000, hubs 3,800 / 70,000.
MULE = Counterparties(
    account_id="a_01",
    payers_30d=4,
    payees_30d=2,
    new_internal_payers_7d=2,
    new_internal_payers_30d=2,
    new_external_payers_7d=1,
    new_external_payers_30d=1,
    new_payees_7d=1,
    new_payees_30d=1,
    new_payer_share_30d=0.75,
    max_new_payers_24h=3,
    top_payee_share_30d=66_200 / 70_000,
    hub_payee_share_30d=3_800 / 70_000,
)
SAVER = Counterparties(
    account_id="a_02",
    payers_30d=0,
    payees_30d=0,
    new_internal_payers_7d=0,
    new_internal_payers_30d=0,
    new_external_payers_7d=0,
    new_external_payers_30d=0,
    new_payees_7d=0,
    new_payees_30d=0,
    new_payer_share_30d=0.0,
    max_new_payers_24h=0,
    top_payee_share_30d=0.0,
    hub_payee_share_30d=0.0,
)


def _row(row: Counterparties) -> dict[str, Any]:
    return {"v_id": row.account_id, "v_type": "Account", "attributes": vars(row)}


def _answer(rows: list[Counterparties]) -> list[dict[str, Any]]:
    """An answer as TigerGraph prints it."""
    return [
        {"status": "ok", "version": VERSIONS[READ_COUNTERPARTIES]},
        {"accounts": [_row(row) for row in rows]},
    ]


def test_an_answer_parses_each_account_sorted_by_id() -> None:
    assert read_counterparties.parse(_answer([SAVER, MULE]), ["a_01", "a_02"]) == (MULE, SAVER)
    # A share printed as an integer, as TigerGraph prints 0 and 1, is a float.
    answer = _answer([MULE, SAVER])
    answer[1]["accounts"][1]["attributes"]["top_payee_share_30d"] = 0
    parsed = read_counterparties.parse(answer, {"a_01", "a_02"})
    assert parsed[1].top_payee_share_30d == 0.0
    assert isinstance(parsed[1].top_payee_share_30d, float)


def test_a_status_row_or_another_version_is_refused() -> None:
    for status in (
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
        "invisible_hub",
    ):
        with pytest.raises(QueryRefused, match=status) as refused:
            read_counterparties.parse([{"status": status}], ["a_01"])
        assert refused.value.status == status
    with pytest.raises(ValueError, match="mule install"):
        read_counterparties.parse(
            [{"status": "ok", "version": "read_counterparties-0123456789ab"}], []
        )


@pytest.mark.parametrize(
    ("rows", "asked"),
    [
        ([MULE], ["a_01", "a_02"]),
        ([MULE, SAVER], ["a_01"]),
        ([MULE, MULE, SAVER], ["a_01", "a_02"]),
    ],
    ids=["missing", "not asked", "twice"],
)
def test_a_row_for_every_account_asked_and_no_other(
    rows: list[Counterparties], asked: list[str]
) -> None:
    with pytest.raises(ValueError, match="answered the accounts"):
        read_counterparties.parse(_answer(rows), asked)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(MULE, new_internal_payers_7d=3), "more new internal payers over 7 days"),
        (replace(MULE, new_external_payers_7d=2), "more new external payers over 7 days"),
        (replace(MULE, new_payees_7d=2), "more new payees over 7 days"),
        (replace(MULE, payers_30d=2), "more new payers than payers"),
        (
            replace(MULE, payees_30d=0, top_payee_share_30d=0.0, hub_payee_share_30d=0.0),
            "more new payees than payees",
        ),
        (replace(MULE, new_payer_share_30d=1.5), "outside"),
        (replace(MULE, hub_payee_share_30d=-0.1), "outside"),
        (replace(SAVER, new_payer_share_30d=0.5), "from new payers without one"),
        (replace(SAVER, top_payee_share_30d=1.0), "without a payee"),
        (replace(MULE, max_new_payers_24h=4), "more new payers in 24 hours"),
        (replace(MULE, max_new_payers_24h=0), "no new payer in 24 hours"),
        (replace(SAVER, max_new_payers_24h=1), "more new payers in 24 hours"),
    ],
    ids=[
        "internal 7 over 30",
        "external 7 over 30",
        "payees 7 over 30",
        "new payers over payers",
        "new payees over payees",
        "a share above 1",
        "a negative share",
        "a new-payer share with no new payer",
        "a payee share with no payee",
        "a burst of more payers than new",
        "no burst of kept payers",
        "a burst with no payer",
    ],
)
def test_a_row_the_query_cannot_give_is_refused(row: Counterparties, message: str) -> None:
    others = [SAVER] if row.account_id == "a_01" else [MULE]
    with pytest.raises(ValueError, match=message):
        read_counterparties.parse(_answer([row, *others]), ["a_01", "a_02"])


def test_more_new_payers_than_the_cap_may_keep_none_in_24_hours() -> None:
    """With more than 256 new payers the whole-second rule may drop every first payment (257
    in one second leave none after it), so only the cap bounds the burst; under it, a new payer
    is always kept and counts at least itself."""
    many = replace(
        MULE,
        payers_30d=400,
        new_internal_payers_7d=300,
        new_internal_payers_30d=300,
        max_new_payers_24h=0,
    )
    assert read_counterparties.parse(_answer([many, SAVER]), ["a_01", "a_02"])[0] == many
    with pytest.raises(ValueError, match="more new payers in 24 hours"):
        read_counterparties.parse(
            _answer([replace(many, max_new_payers_24h=257), SAVER]), ["a_01", "a_02"]
        )
