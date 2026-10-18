"""The test world builder: the 22 contract tables with their columns and types, each account's
labels by the convention, ownership over the account's life, payments with their roles, and the
helpers for a household, a ring, a joiner and another bank's account."""

from __future__ import annotations

import pandas as pd
import pytest

from mule_pattern_learner.contract import TABLES
from mule_pattern_learner.testing.reference.graph import DTYPES, frame
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, NO_RING, Mule, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries.rows import MONDAY_MS, WEEK_MS


def _rows(world: WorldBuilder, table: str) -> list[tuple[object, ...]]:
    return list(world.tables()[table].itertuples(index=False, name=None))


def test_t_is_a_scored_monday_at_midnight() -> None:
    # 2024-07-01 is 182 days, 26 weeks, after the export start 2024-01-01 (1,704,067,200,000).
    assert T == 1_704_067_200_000 + 182 * DAY_MS
    assert T % WEEK_MS == MONDAY_MS and DAY_MS == 24 * HOUR_MS


def test_every_table_has_the_contracts_columns_and_types_even_when_empty() -> None:
    tables = WorldBuilder().tables()
    assert list(tables) == [table.name for table in TABLES]
    for table in TABLES:
        found = tables[table.name]
        assert tuple(found.columns) == table.column_names and found.empty
        assert [str(dtype) for dtype in found.dtypes] == [
            str(pd.Series([], dtype=DTYPES[column.graph_type]).dtype) for column in table.columns
        ]


def test_an_accounts_labels_follow_the_convention() -> None:
    w = WorldBuilder()
    w.party("p_1", T - 400 * DAY_MS)
    w.account("a_plain", "p_1", T - 300 * DAY_MS)
    w.account("a_hidden", "p_1", T - 200 * DAY_MS, mule=Mule(T - 50 * DAY_MS, T - 40 * DAY_MS, 3))
    caught = Mule(T - 30 * DAY_MS, T - 20 * DAY_MS, identified=True)
    w.account("a_caught", "p_1", T - 100 * DAY_MS, closed_ms=T + DAY_MS, mule=caught)
    w.external("x_other", T - 10 * DAY_MS)
    assert _rows(w, "Account") == [
        ("a_plain", "deposit", False, T - 300 * DAY_MS, 0, 0, 0, NO_RING, 0, 0),
        (
            "a_hidden",
            "deposit",
            False,
            T - 200 * DAY_MS,
            0,
            1,
            0,
            3,
            T - 50 * DAY_MS,
            T - 40 * DAY_MS,
        ),
        (
            "a_caught",
            "deposit",
            False,
            T - 100 * DAY_MS,
            T + DAY_MS,
            1,
            1,
            NO_RING,
            T - 30 * DAY_MS,
            T - 20 * DAY_MS,
        ),
        ("x_other", "unknown", True, T - 10 * DAY_MS, 0, -1, -1, NO_RING, 0, 0),
    ]
    # Each owned account is owned over its life, opening to closing; another bank's is not.
    assert _rows(w, "Party_Owns_Account") == [
        ("p_1", "a_plain", T - 300 * DAY_MS, 0),
        ("p_1", "a_hidden", T - 200 * DAY_MS, 0),
        ("p_1", "a_caught", T - 100 * DAY_MS, T + DAY_MS),
    ]


def test_another_banks_account_has_no_owner_or_truth() -> None:
    w = WorldBuilder()
    with pytest.raises(ValueError, match="no owner or truth"):
        w.account("x_1", "p_1", T, external=True)
    with pytest.raises(ValueError, match="no owner or truth"):
        w.account("x_1", None, T, external=True, mule=Mule(T, T))


def test_a_payment_has_its_roles_and_adds_each_bare_vertex_once() -> None:
    w = WorldBuilder()
    w.customer("ann", T - 400 * DAY_MS)
    w.customer("ben", T - 400 * DAY_MS)
    w.pay("x_1", "a_ann", "a_ben", T - DAY_MS, 12.5, "ach", device="d_1", ip="i_1")
    w.pay("z_1", "a_ben", "a_ann", T - HOUR_MS, 7.25, "zelle", device="d_1", tokens=("t_b", "t_a"))
    w.device("p_ann", "d_1", T - 300 * DAY_MS)
    w.token("t_a", "p_ann", "a_ann", T - 200 * DAY_MS)
    assert _rows(w, "Payment_Transaction") == [("x_1", 12.5, "ach", T - DAY_MS)]
    assert _rows(w, "Zelle_Transfer") == [("z_1", 7.25, T - HOUR_MS)]
    assert _rows(w, "Transaction_From_Account") == [("x_1", "a_ann", T - DAY_MS)]
    assert _rows(w, "Transaction_To_Account") == [("x_1", "a_ben", T - DAY_MS)]
    assert _rows(w, "Transaction_Used_Device") == [("x_1", "d_1", T - DAY_MS)]
    assert _rows(w, "Transaction_Used_IP") == [("x_1", "i_1", T - DAY_MS)]
    assert _rows(w, "Transfer_From_Account") == [("z_1", "a_ben", T - HOUR_MS)]
    assert _rows(w, "Transfer_To_Account") == [("z_1", "a_ann", T - HOUR_MS)]
    assert _rows(w, "Transfer_From_Token") == [("z_1", "t_b", T - HOUR_MS)]
    assert _rows(w, "Transfer_To_Token") == [("z_1", "t_a", T - HOUR_MS)]
    assert _rows(w, "Transfer_Used_Device") == [("z_1", "d_1", T - HOUR_MS)]
    assert _rows(w, "Transfer_Used_IP") == []
    assert _rows(w, "Device") == [("d_1",)] and _rows(w, "IP") == [("i_1",)]
    assert _rows(w, "Token") == [("t_b",), ("t_a",)]
    assert _rows(w, "Party_Uses_Token") == [("p_ann", "t_a", T - 200 * DAY_MS, 0)]
    assert _rows(w, "Token_Bound_To_Account") == [("t_a", "a_ann", T - 200 * DAY_MS, 0)]
    with pytest.raises(ValueError, match="no tokens"):
        w.pay("x_2", "a_ann", "a_ben", T - DAY_MS, 1.0, "ach", tokens=("t_a", "t_b"))


def test_the_helpers_build_a_household_a_ring_a_joiner_and_another_banks_account() -> None:
    w = WorldBuilder()
    home = w.household(["ann", "ben"], T - 400 * DAY_MS, "d_home", "i_home")
    ring = w.ring(
        7,
        ["mia", "max"],
        T - 300 * DAY_MS,
        "d_ring",
        T - 61 * DAY_MS,
        T - 60 * DAY_MS,
        T - 10 * DAY_MS,
        identified={"mia"},
    )
    joiner = w.joiner("new", T - 20 * DAY_MS, "d_new", "i_new", T - 20 * DAY_MS + 5 * HOUR_MS)
    other = w.external("x_bank", T - 500 * DAY_MS, account_type="deposit")
    assert (home, ring, joiner, other) == (
        ["a_ann", "a_ben"],
        ["a_mia", "a_max"],
        "a_new",
        "x_bank",
    )
    assert _rows(w, "Party") == [
        ("p_ann", T - 400 * DAY_MS),
        ("p_ben", T - 400 * DAY_MS),
        ("p_mia", T - 300 * DAY_MS),
        ("p_max", T - 300 * DAY_MS),
        ("p_new", T - 20 * DAY_MS),
    ]
    # The home's members since joining; the ring's from its tenure's start; the joiner's from
    # its first use, five hours after joining.
    assert _rows(w, "Party_Uses_Device") == [
        ("p_ann", "d_home", T - 400 * DAY_MS, 0),
        ("p_ben", "d_home", T - 400 * DAY_MS, 0),
        ("p_mia", "d_ring", T - 61 * DAY_MS, 0),
        ("p_max", "d_ring", T - 61 * DAY_MS, 0),
        ("p_new", "d_new", T - 20 * DAY_MS + 5 * HOUR_MS, 0),
    ]
    assert _rows(w, "Party_Uses_IP") == [
        ("p_ann", "i_home", T - 400 * DAY_MS, 0),
        ("p_ben", "i_home", T - 400 * DAY_MS, 0),
        ("p_new", "i_new", T - 20 * DAY_MS + 5 * HOUR_MS, 0),
    ]
    labels = {row[0]: row[5:] for row in _rows(w, "Account")}
    assert labels["a_mia"] == (1, 1, 7, T - 60 * DAY_MS, T - 10 * DAY_MS)
    assert labels["a_max"] == (1, 0, 7, T - 60 * DAY_MS, T - 10 * DAY_MS)
    assert labels["a_ann"] == (0, 0, NO_RING, 0, 0)
    assert labels["x_bank"] == (-1, -1, NO_RING, 0, 0)


def test_a_contract_frame_refuses_a_value_of_another_type() -> None:
    assert frame("Party", [("p_1", 5)]).dtypes.to_dict()["joined_ms"] == "int64"
    for row, message in (
        (("p_1", -1), "not a UINT"),
        (("p_1", True), "not a UINT"),
        (("p_1", 1.0), "not a UINT"),
        ((1, 5), "not a STRING"),
        (("p_1",), "1 values, not 2"),
    ):
        with pytest.raises(ValueError, match=message):
            frame("Party", [row])
    with pytest.raises(ValueError, match="not a BOOL"):
        frame("Account", [("a_1", "deposit", 0, 1, 0, 0, 0, -1, 0, 0)])
    # A DOUBLE takes an integer as a float.
    assert frame("Zelle_Transfer", [("z_1", 3, 1)])["amount"].tolist() == [3.0]
