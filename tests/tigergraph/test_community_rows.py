"""read_community's parameter builder and row parser, on answers written by hand in
TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set as a list
of vertices with their projected fields under "attributes".

The parser takes a block for every account asked for and no other, and refuses the values the
query cannot give.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.tigergraph.queries import READ_COMMUNITY, VERSIONS, read_community
from mule_pattern_learner.tigergraph.queries.read_community import Community
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS, QueryRefused

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday
CAP = DEFAULT_CONFIG.subgraph.community  # 64


def test_a_request_takes_a_time_its_accounts_and_the_hubs_each_once() -> None:
    """Like B2, B4 and B5: the accounts as a list of ids, which the query checks itself; the
    hubs as a set of vertices, in TigerGraph's JSON for SET<VERTEX<Account>>."""
    assert read_community.params(T_MS, ["a_02", "a_01"], ["h_01", "h_02"]) == {
        "t_ms": T_MS,
        "account_ids": ["a_02", "a_01"],
        "hub_accounts": [{"id": "h_01"}, {"id": "h_02"}],
    }
    assert read_community.params(T_MS, ["a_01"], [])["hub_accounts"] == []
    # Any whole second is a time: a Tuesday afternoon is one, a millisecond is not.
    assert read_community.params(T_MS + DAY_MS + 3_661_000, ["a_01"], [])["t_ms"] == (
        T_MS + DAY_MS + 3_661_000
    )
    with pytest.raises(ValueError, match="whole second"):
        read_community.params(T_MS + 1, ["a_01"], [])
    with pytest.raises(ValueError, match="repeats an account"):
        read_community.params(T_MS, ["a_01", "a_01"], [])
    with pytest.raises(ValueError, match="at least one account"):
        read_community.params(T_MS, [], [])
    with pytest.raises(ValueError, match="repeats an id"):
        read_community.params(T_MS, ["a_01"], ["h_01", "h_01"])
    with pytest.raises(ValueError, match="holds ids"):
        read_community.params(T_MS, ["a_01"], [""])


# Four blocks, worked by hand.
# - RING, a_01: its owner c_01 (round 0) shares the ring phone d_01 with c_02, c_03 and c_04
#   (round 1), and the ring IP i_01 with c_02 and c_03; c_02 shares its home IP i_02 with its
#   partner c_05 (round 2); a_01 paid an account of c_06 (the payment round). 6 members.
#   Shared nodes: d_01 (4 members), i_01 (3), i_02 (2): 3 nodes, 9 links, density
#   9 / (6 x 3) = 0.5. The ring links all began in the last 30 days, 4 on d_01 and 3 on i_01,
#   and the household's on i_02 long ago: 7 / 9. Of 16 payments of the members' accounts in
#   30 days, 4 of 66,200 cents went between members: 264,800 cents, isolation 4 / 16 = 0.25.
# - HOUSEHOLD, a_02: c_10 and c_11 share their home IP from long ago: 2 members, 1 shared
#   node, 2 links, density 2 / (2 x 1) = 1.0, no new link; 1 transfer of 50,000 cents between
#   them and 39 payments with the world: isolation 1 / 40 = 0.025.
# - LONER, a_03: its owner shares nothing: 1 member, 25 payments with the world.
# - ORPHAN, a_04: no visible owner at t, so no member and an empty block.
RING = Community(
    account_id="a_01",
    members=6,
    capped=False,
    shared_nodes=3,
    density=0.5,
    new_link_share_30d=7 / 9,
    internal_amount_30d=264_800,
    isolation=0.25,
    member_payments_30d=16,
)
HOUSEHOLD = Community(
    account_id="a_02",
    members=2,
    capped=False,
    shared_nodes=1,
    density=1.0,
    new_link_share_30d=0.0,
    internal_amount_30d=50_000,
    isolation=0.025,
    member_payments_30d=40,
)
LONER = Community(
    account_id="a_03",
    members=1,
    capped=False,
    shared_nodes=0,
    density=0.0,
    new_link_share_30d=0.0,
    internal_amount_30d=0,
    isolation=0.0,
    member_payments_30d=25,
)
ORPHAN = Community(
    account_id="a_04",
    members=0,
    capped=False,
    shared_nodes=0,
    density=0.0,
    new_link_share_30d=0.0,
    internal_amount_30d=0,
    isolation=0.0,
    member_payments_30d=0,
)
ALL = [RING, HOUSEHOLD, LONER, ORPHAN]
IDS = ["a_01", "a_02", "a_03", "a_04"]


def _row(row: Community) -> dict[str, Any]:
    return {"v_id": row.account_id, "v_type": "Account", "attributes": dict(vars(row))}


def _answer(rows: list[Community]) -> list[dict[str, Any]]:
    """An answer as TigerGraph prints it."""
    return [
        {"status": "ok", "version": VERSIONS[READ_COMMUNITY]},
        {"accounts": [_row(row) for row in rows]},
    ]


def test_an_answer_parses_each_account_sorted_by_id() -> None:
    assert read_community.parse(_answer(ALL[::-1]), IDS) == tuple(ALL)
    # A share printed as an integer, as TigerGraph prints 0 and 1, is a float.
    answer = _answer([HOUSEHOLD])
    answer[1]["accounts"][0]["attributes"]["density"] = 1
    parsed = read_community.parse(answer, {"a_02"})
    assert parsed[0].density == 1.0 and isinstance(parsed[0].density, float)


def test_a_status_row_is_refused() -> None:
    for status in (
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
        "invisible_hub",
    ):
        with pytest.raises(QueryRefused, match=status) as refused:
            read_community.parse([{"status": status}], ["a_01"])
        assert refused.value.status == status


@pytest.mark.parametrize(
    ("rows", "asked"),
    [
        ([RING], ["a_01", "a_02"]),
        ([RING, HOUSEHOLD], ["a_01"]),
        ([RING, RING, HOUSEHOLD], ["a_01", "a_02"]),
    ],
    ids=["missing", "not asked", "twice"],
)
def test_a_block_for_every_account_asked_and_no_other(
    rows: list[Community], asked: list[str]
) -> None:
    with pytest.raises(ValueError, match="answered the accounts"):
        read_community.parse(_answer(rows), asked)


def test_a_field_of_another_type_is_refused() -> None:
    for field, value in (("capped", 0), ("members", -1), ("internal_amount_30d", 1.5)):
        answer = _answer([RING])
        answer[1]["accounts"][0]["attributes"][field] = value
        with pytest.raises(ValueError, match=f"Community.{field}"):
            read_community.parse(answer, ["a_01"])


# A community cut at the cap holds 64, and so may one that was not cut.
FULL = replace(RING, members=CAP, capped=True, density=9 / (CAP * 3))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(FULL, members=CAP + 1), "more than 64 members"),
        (replace(FULL, members=CAP - 1), "a cut community that is not full"),
        (replace(RING, density=1.5), "outside"),
        (replace(RING, new_link_share_30d=-0.1), "outside"),
        (replace(HOUSEHOLD, isolation=1.25), "outside"),
        (replace(ORPHAN, member_payments_30d=3), "without an owner that is not empty"),
        (replace(ORPHAN, capped=True, members=0), "a cut community that is not full"),
        (replace(LONER, shared_nodes=1, density=1.0), "fewer"),
        (replace(LONER, density=0.5), "a density unlike its shared nodes"),
        (replace(HOUSEHOLD, density=0.0), "a density unlike its shared nodes"),
        (replace(LONER, new_link_share_30d=1.0), "new links without a shared node"),
        (replace(LONER, member_payments_30d=0, isolation=0.5), "an isolation without a payment"),
        (replace(LONER, isolation=0.04), "payments between members of a community of one"),
        (replace(LONER, internal_amount_30d=100), "payments between members of a community"),
        (replace(HOUSEHOLD, isolation=0.0), "cents between members without an internal"),
    ],
    ids=[
        "more than the cap",
        "cut below the cap",
        "a density above 1",
        "a negative new-link share",
        "an isolation above 1",
        "payments without an owner",
        "cut without an owner",
        "a shared node in a community of one",
        "a density without a shared node",
        "a shared node without a density",
        "new links without a shared node",
        "an isolation without a payment",
        "an internal payment in a community of one",
        "internal cents in a community of one",
        "internal cents without an internal payment",
    ],
)
def test_a_block_the_query_cannot_give_is_refused(row: Community, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_community.parse(_answer([row]), [row.account_id])


@pytest.mark.parametrize(
    "row",
    [
        FULL,
        replace(FULL, capped=False),
        # Every member pays only the others, and every pair shares each node.
        replace(HOUSEHOLD, isolation=1.0, member_payments_30d=1),
        # An internal payment of under half a cent rounds to 0 cents.
        replace(HOUSEHOLD, internal_amount_30d=0),
    ],
    ids=["cut at the cap", "full but not cut", "isolated", "internal cents rounding to 0"],
)
def test_a_block_at_an_edge_is_accepted(row: Community) -> None:
    assert read_community.parse(_answer([row]), [row.account_id]) == (row,)
