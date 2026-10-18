"""read_shared_identity's parameter builder and row parser, on answers written by hand in
TigerGraph's shape: a status row first, then one row per PRINT, a printed vertex set as a list
of vertices with their projected fields under "attributes", and a printed map as an object.

The parser builds each scored account's device and IP graph from the rows printed once and the
ids and edges printed per scored account, and refuses what the query cannot give.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries import (
    READ_SHARED_IDENTITY,
    VERSIONS,
    read_shared_identity,
)
from mule_pattern_learner.tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityAccount,
    IdentityGraph,
    NodeRow,
    OwnerEnrolled,
    Seen,
)
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS, LOOKBACK_MS, QueryRefused

T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, a Monday
HOUR_MS = 3_600_000


def test_a_request_takes_a_monday_its_accounts_and_the_hubs_each_once() -> None:
    assert read_shared_identity.params(T_MS, ["a_02", "a_01"], ["h_01"]) == {
        "t_ms": T_MS,
        "account_ids": ["a_02", "a_01"],
        "hub_accounts": [{"id": "h_01"}],
    }
    with pytest.raises(ValueError, match="Monday"):
        read_shared_identity.params(T_MS + DAY_MS, ["a_01"], [])
    with pytest.raises(ValueError, match="repeats an account"):
        read_shared_identity.params(T_MS, ["a_01", "a_01"], [])
    with pytest.raises(ValueError, match="repeats an id"):
        read_shared_identity.params(T_MS, ["a_01"], ["h_01", "h_01"])


# A scored mule a_01 of customer q_01, and a scored a_02 whose graph is empty, on 2024-04-01.
# - q_01 has had phone d_01 since before the lookback (an unknown start) and last paid from it
#   12 hours before t; it is enrolled on the carrier IP i_09, a leaf. a_01's payments were
#   seen on d_01 three times, first 20 days before t, a known start, and on IP i_01, which
#   nobody registered, twice. q_01 owns a_01 and its savings account.
# - q_07, a co-user, enrolled d_01 10 days before t, a known start, and last paid from it 2
#   days before t with its deposit account b_07, which it brings; the pair began sharing d_01
#   10 days before t, the later of the two starts, so it is new over 30 days and not over 7.
# - v_01, a hop-1 account of a_01 that is no hub, is expanded: its customer q_05 is enrolled
#   on d_05, which only it uses, and v_01 was seen there once, 5 days before t.
Q_01 = CustomerRow("q_01", 2, 1, 1, 0, 0, 1, 1, 0)
Q_05 = CustomerRow("q_05", 1, 1, 0, 0, 0, 0, 0, 0)
Q_07 = CustomerRow("q_07", 1, 1, 0, 1, 0, 1, 1, 0)
D_01 = NodeRow("d_01", False, 2, 2, 0, 1, 2, 5, False)
D_05 = NodeRow("d_05", False, 1, 1, 0, 0, 1, 1, False)
I_01 = NodeRow("i_01", False, 1, 1, 0, 0, 1, 2, True)
I_09 = NodeRow("i_09", True, 0, 0, 0, 0, 0, 0, False)
B_07 = IdentityAccount("b_07", False)
ENROLLED = [
    Enrolled("q_01", "d_01", False, 0, True, 12 * HOUR_MS),
    Enrolled("q_01", "i_09", False, 0, False, 0),
    Enrolled("q_07", "d_01", True, 10 * DAY_MS, True, 2 * DAY_MS),
]
OWNER_ENROLLED = [
    OwnerEnrolled("a_01", "d_01", False, 0, True, 12 * HOUR_MS),
    OwnerEnrolled("a_01", "i_09", False, 0, False, 0),
    OwnerEnrolled("b_07", "d_01", True, 10 * DAY_MS, True, 2 * DAY_MS),
    OwnerEnrolled("v_01", "d_05", False, 0, True, 5 * DAY_MS),
]
SEEN = [
    Seen("a_01", "d_01", 3, True, 20 * DAY_MS, 12 * HOUR_MS),
    Seen("a_01", "i_01", 2, False, 0, DAY_MS),
    Seen("b_07", "d_01", 1, True, 2 * DAY_MS, 2 * DAY_MS),
    Seen("v_01", "d_05", 1, False, 0, 5 * DAY_MS),
]


def _vertices(kind: str, rows: list[Any], key: str) -> list[dict[str, Any]]:
    return [{"v_id": getattr(row, key), "v_type": kind, "attributes": vars(row)} for row in rows]


def _answer(
    *,
    customers: list[CustomerRow] | None = None,
    devices: list[NodeRow] | None = None,
    ips: list[NodeRow] | None = None,
    accounts: list[IdentityAccount] | None = None,
    members: dict[str, dict[str, list[str]]] | None = None,
    edges: dict[str, dict[str, list[Any]]] | None = None,
) -> list[dict[str, Any]]:
    """An answer as TigerGraph prints it, a_01's graph unless told otherwise."""
    found_members = {
        "subgraph_customers": {"a_01": ["q_01", "q_07", "q_05"]},
        "subgraph_nodes": {"a_01": ["d_01", "i_09", "i_01", "d_05"]},
        "co_user_accounts": {"a_01": ["b_07"]},
        **(members or {}),
    }
    found_edges = {
        "enrolled": {"a_01": ENROLLED},
        "owner_enrolled": {"a_01": OWNER_ENROLLED},
        "seen": {"a_01": SEEN},
        **(edges or {}),
    }
    return [
        {"status": "ok", "version": VERSIONS[READ_SHARED_IDENTITY]},
        {"customers": _vertices("Party", customers or [Q_01, Q_05, Q_07], "customer_id")},
        {"devices": _vertices("Device", devices or [D_01, D_05], "node_id")},
        {"ips": _vertices("IP", ips or [I_01, I_09], "node_id")},
        {"accounts": _vertices("Account", accounts or [B_07], "account_id")},
        *({key: value} for key, value in found_members.items()),
        *(
            {key: {scored: [vars(edge) for edge in listed] for scored, listed in value.items()}}
            for key, value in found_edges.items()
        ),
    ]


def test_an_answer_parses_each_scored_accounts_graph_sorted() -> None:
    mule, empty = read_shared_identity.parse(_answer(), ["a_02", "a_01"])
    assert mule == IdentityGraph(
        "a_01",
        (Q_01, Q_05, Q_07),
        (D_01, D_05),
        (I_01, I_09),
        (B_07,),
        tuple(ENROLLED),
        tuple(OWNER_ENROLLED),
        tuple(SEEN),
    )
    assert empty == IdentityGraph("a_02", (), (), (), (), (), (), ())


def test_a_status_row_or_another_version_is_refused() -> None:
    for status in (
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
        "invisible_hub",
    ):
        with pytest.raises(QueryRefused, match=status):
            read_shared_identity.parse([{"status": status}], ["a_01"])
    with pytest.raises(ValueError, match="mule install"):
        read_shared_identity.parse(
            [{"status": "ok", "version": "read_shared_identity-0123456789ab"}], []
        )


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (_answer(members={"subgraph_nodes": {"a_03": []}}), "printed subgraph_nodes of"),
        (_answer(customers=[Q_01, Q_05]), "a customer or account it lacks"),
        (_answer(accounts=[IdentityAccount("b_08", False)]), "a customer or account it lacks"),
        (_answer(devices=[D_01]), "a node it lacks"),
        (_answer(ips=[I_01, I_09, replace(D_05, node_id="d_01")]), "as a device and an IP"),
        (_answer(customers=[Q_01, Q_05, Q_07, Q_07]), "a row of customers twice"),
        (
            _answer(customers=[Q_01, Q_05, Q_07, replace(Q_05, customer_id="q_09")]),
            "of no subgraph",
        ),
        (
            _answer(
                edges={
                    "enrolled": {"a_01": [*ENROLLED, Enrolled("q_09", "d_01", False, 0, False, 0)]}
                }
            ),
            "an enrolment edge of a_01 with an end outside it",
        ),
        (
            _answer(edges={"seen": {"a_01": [*SEEN, Seen("a_01", "d_09", 1, False, 0, DAY_MS)]}}),
            "a seen edge of a_01 to a node outside it",
        ),
        (
            _answer(
                edges={
                    "owner_enrolled": {
                        "a_01": [*OWNER_ENROLLED, OwnerEnrolled("a_01", "d_09", False, 0, False, 0)]
                    }
                }
            ),
            "an owner-enrolment edge of a_01 to a node outside it",
        ),
        (
            _answer(edges={"seen": {"a_01": [*SEEN, replace(SEEN[0], uses_90d=4)]}}),
            "an edge twice",
        ),
    ],
    ids=[
        "a graph not asked for",
        "a customer without its row",
        "a co-user's account without its row",
        "a node without its row",
        "a node of both kinds",
        "a customer twice",
        "a customer of no graph",
        "an enrolment edge from outside",
        "a seen edge to outside",
        "an owner-enrolment edge to outside",
        "one edge twice",
    ],
)
def test_an_answer_that_is_no_set_of_graphs_is_refused(
    answer: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_shared_identity.parse(answer, ["a_01", "a_02"])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(Q_07, new_devices_30d=2), "more new links than links"),
        (replace(Q_01, new_ips_30d=2), "more new links than links"),
        (replace(Q_01, new_co_users_30d=2), "more new co-users"),
        (replace(Q_01, co_users_both=2), "more new co-users"),
    ],
)
def test_a_customer_row_the_query_cannot_give_is_refused(row: CustomerRow, message: str) -> None:
    customers = [
        row if other.customer_id == row.customer_id else other for other in (Q_01, Q_05, Q_07)
    ]
    with pytest.raises(ValueError, match=message):
        read_shared_identity.parse(_answer(customers=customers), ["a_01"])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(I_09, customers_90d=9), "a leaf with more than its flag"),
        (replace(I_09, observed_only=True), "a leaf with more than its flag"),
        (replace(D_01, customers_90d=9, customers_30d=2), "a crowd that is no leaf"),
        (replace(D_01, new_customers_7d=2), "more new customers than customers sharing it"),
        (replace(D_01, new_customers_30d=3), "more new customers than customers sharing it"),
        (replace(D_01, customers_30d=3), "more customers sharing it than customers"),
        (replace(D_01, accounts_seen_30d=6), "more accounts seen than payments"),
    ],
)
def test_a_node_row_the_query_cannot_give_is_refused(row: NodeRow, message: str) -> None:
    devices = [row if row.node_id == "d_01" else D_01, D_05]
    ips = [I_01, row if row.node_id == "i_09" else I_09]
    with pytest.raises(ValueError, match=message):
        read_shared_identity.parse(_answer(devices=devices, ips=ips), ["a_01"])


def _ends(edge: Enrolled | OwnerEnrolled | Seen) -> tuple[str, str]:
    start = edge.customer_id if isinstance(edge, Enrolled) else edge.account_id
    return start, edge.node_id


@pytest.mark.parametrize(
    ("key", "edge", "message"),
    [
        ("enrolled", replace(ENROLLED[0], since_start_ms=5), "a since_start it cannot give"),
        (
            "enrolled",
            replace(ENROLLED[2], since_start_ms=LOOKBACK_MS),
            "a since_start it cannot give",
        ),
        ("enrolled", replace(ENROLLED[2], since_start_ms=0), "a since_start it cannot give"),
        (
            "owner_enrolled",
            replace(OWNER_ENROLLED[1], since_last_seen_ms=5),
            "a since_last_seen it cannot give",
        ),
        (
            "owner_enrolled",
            replace(OWNER_ENROLLED[0], since_last_seen_ms=LOOKBACK_MS + 1),
            "a since_last_seen it cannot give",
        ),
        ("seen", replace(SEEN[0], uses_90d=0), "a seen edge of no payment"),
        ("seen", replace(SEEN[1], since_first_use_ms=5), "a since_first_use it cannot give"),
        ("seen", replace(SEEN[0], since_last_use_ms=0), "a since_last_use it cannot give"),
        ("seen", replace(SEEN[0], since_first_use_ms=HOUR_MS), "a first use after its last"),
    ],
    ids=[
        "an absent start not 0",
        "a start before the lookback",
        "a present start of 0",
        "an absent last seen not 0",
        "a last seen before the lookback",
        "a seen edge of no use",
        "an absent first use not 0",
        "a last use at t",
        "a first use after the last",
    ],
)
def test_an_edge_the_query_cannot_give_is_refused(key: str, edge: Any, message: str) -> None:
    listed: list[Any] = {"enrolled": ENROLLED, "owner_enrolled": OWNER_ENROLLED, "seen": SEEN}[key]
    changed = [edge if _ends(other) == _ends(edge) else other for other in listed]
    with pytest.raises(ValueError, match=message):
        read_shared_identity.parse(_answer(edges={key: {"a_01": changed}}), ["a_01"])
