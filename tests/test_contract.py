"""The export contract as docs/design.md ("The export contract") states it: 22 tables, their
columns in order with their graph types, the closed value lists and the label convention."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mule_pattern_learner.contract import (
    ACCOUNT_TYPES,
    CONVENTION_COLUMNS,
    EXTERNAL_LABEL,
    LABEL_COLUMNS,
    LABEL_VALUES,
    PAYMENT_RAILS,
    TABLES,
    Kind,
)

BY_NAME = {table.name: table for table in TABLES}
VERTICES = {
    "Party": ("id", "joined_ms"),
    "Account": (
        "id",
        "account_type",
        "is_external",
        "opened_ms",
        "closed_ms",
        "is_mule",
        "pu_label",
        "mule_ring_id",
        "mule_onset_ms",
        "mule_last_ms",
    ),
    "Token": ("id",),
    "Device": ("id",),
    "IP": ("id",),
    "Payment_Transaction": ("id", "amount", "payment_rail", "event_ts_ms"),
    "Zelle_Transfer": ("id", "amount", "event_ts_ms"),
}
# Each edge table's FROM and TO vertex types and its reverse edge type.
LINKS = {
    "Party_Owns_Account": ("Party", "Account", "Account_Owned_By_Party"),
    "Party_Uses_Token": ("Party", "Token", "Token_Used_By_Party"),
    "Token_Bound_To_Account": ("Token", "Account", "Account_Bound_From_Token"),
    "Party_Uses_Device": ("Party", "Device", "Device_Used_By_Party"),
    "Party_Uses_IP": ("Party", "IP", "IP_Used_By_Party"),
}
ROLES = {
    "Transaction_From_Account": (
        "Payment_Transaction",
        "Account",
        "Account_Initiated_Transaction",
    ),
    "Transaction_To_Account": ("Payment_Transaction", "Account", "Account_Received_Transaction"),
    "Transaction_Used_Device": ("Payment_Transaction", "Device", "Device_Used_By_Transaction"),
    "Transaction_Used_IP": ("Payment_Transaction", "IP", "IP_Observed_In_Transaction"),
    "Transfer_From_Account": ("Zelle_Transfer", "Account", "Account_Sent_Zelle_Transfer"),
    "Transfer_To_Account": ("Zelle_Transfer", "Account", "Account_Received_Zelle_Transfer"),
    "Transfer_From_Token": ("Zelle_Transfer", "Token", "Token_Sent_Transfer"),
    "Transfer_To_Token": ("Zelle_Transfer", "Token", "Token_Received_Transfer"),
    "Transfer_Used_Device": ("Zelle_Transfer", "Device", "Device_Used_By_Transfer"),
    "Transfer_Used_IP": ("Zelle_Transfer", "IP", "IP_Used_By_Transfer"),
}


def test_the_contract_has_22_tables_7_vertex_5_link_and_10_role() -> None:
    assert len(TABLES) == len(BY_NAME) == 22
    kinds = {kind: [table.name for table in TABLES if table.kind is kind] for kind in Kind}
    assert kinds == {
        Kind.VERTEX: list(VERTICES),
        Kind.LINK: list(LINKS),
        Kind.ROLE: list(ROLES),
    }


def test_each_vertex_table_has_the_designs_columns_in_order() -> None:
    for name, columns in VERTICES.items():
        table = BY_NAME[name]
        assert table.column_names == columns, name
        assert (table.source, table.target, table.reverse) == (None, None, None), name


def test_every_link_and_role_table_has_its_ends_its_times_and_its_reverse() -> None:
    for name, (source, target, reverse) in LINKS.items():
        table = BY_NAME[name]
        assert table.column_names == ("from_id", "to_id", "valid_from_ms", "valid_to_ms"), name
        assert (table.source, table.target, table.reverse) == (source, target, reverse), name
    for name, (source, target, reverse) in ROLES.items():
        table = BY_NAME[name]
        assert table.column_names == ("from_id", "to_id", "event_ts_ms"), name
        assert (table.source, table.target, table.reverse) == (source, target, reverse), name
    reverses = [table.reverse for table in TABLES if table.kind is not Kind.VERTEX]
    assert len(set(reverses)) == 15 and not set(reverses) & set(BY_NAME)


def test_every_column_has_the_designs_graph_type() -> None:
    expected = {
        "id": "STRING",
        "from_id": "STRING",
        "to_id": "STRING",
        "account_type": "STRING",
        "payment_rail": "STRING",
        "is_external": "BOOL",
        "is_mule": "INT",
        "pu_label": "INT",
        "mule_ring_id": "INT",
        "amount": "DOUBLE",
    }
    for table in TABLES:
        for column in table.columns:
            # Every time is UINT milliseconds (the owner: "unsigned integer is fine").
            graph_type = "UINT" if column.name.endswith("_ms") else expected[column.name]
            assert column.graph_type == graph_type, (table.name, column.name)


def test_nothing_left_out_of_the_export_is_in_the_contract() -> None:
    left_out = {
        "split",
        "discovered_ms",
        "discovery_channel",
        "mule_layer",
        "mule_flow",
        "device_type",
        "country_code",
        "fraud_label",
        "is_fraud",
        "Address",
        "Party_Has_Address",
        "Account_Uses_Device",
        "Transaction_From_Token",
        "Transaction_To_Token",
    }
    names = {table.name for table in TABLES} | {
        column.name for table in TABLES for column in table.columns
    }
    assert not names & left_out


def test_the_value_lists_are_the_designs() -> None:
    assert ACCOUNT_TYPES == ("deposit", "credit", "brokerage", "gl", "unknown")
    # A Zelle payment is a Zelle_Transfer, never a rail of Payment_Transaction.
    assert PAYMENT_RAILS == ("ach", "card", "cash", "check", "internal", "unknown")


def test_the_label_columns_and_their_convention_are_the_designs() -> None:
    assert LABEL_COLUMNS == ("is_mule", "pu_label", "mule_ring_id", "mule_onset_ms", "mule_last_ms")
    assert set(LABEL_COLUMNS) <= set(BY_NAME["Account"].column_names)
    # is_mule and pu_label are -1 exactly on external accounts, and 0 or 1 on every internal one.
    assert CONVENTION_COLUMNS == ("is_mule", "pu_label")
    assert (LABEL_VALUES, EXTERNAL_LABEL) == ((-1, 0, 1), -1)


def test_the_contract_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        BY_NAME["Account"].name = "Customer"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(FrozenInstanceError):
        BY_NAME["Party"].columns[0].graph_type = "INT"  # pyright: ignore[reportAttributeAccessIssue]
