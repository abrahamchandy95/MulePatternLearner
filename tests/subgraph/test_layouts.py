"""The layouts of a batch: every column in the order of docs/features.md's tables, raw and
encoded widths as its "Widths" table gives them, and the sixteen relations with the degree
columns each node type counts."""

from __future__ import annotations

import pytest

from mule_pattern_learner.model.encode import Kind, Layout
from mule_pattern_learner.subgraph.layouts import (
    ACCOUNT_LAYOUT,
    COMMUNITY_LAYOUT,
    CUSTOMER_LAYOUT,
    DEGREES,
    DEVICE_LAYOUT,
    EDGE_LAYOUTS,
    ENROLMENT_LAYOUT,
    FORWARD,
    IP_LAYOUT,
    LAYOUT,
    NODE_LAYOUTS,
    PAYMENT_LAYOUT,
    RELATIONS,
    SEEN_LAYOUT,
)


def _duration(name: str) -> list[str]:
    return [f"{name}_present", f"{name}_ms"]


# docs/features.md, "Account node": columns 1 to 46, each duration as presence and
# milliseconds, then the 7 degree columns of "Degree columns".
ACCOUNT_RAW = [
    "is_external",
    "hub",
    "hub_customers_90d",
    "in_amount_7d",
    "in_amount_30d",
    "in_amount_90d",
    "out_amount_7d",
    "out_amount_30d",
    "out_amount_90d",
    "flow_present",
    "left_15m",
    "left_1h",
    "left_6h",
    "left_1d",
    "left_3d",
    "left_7d",
    "held_share",
    *_duration("median_dwell"),
    "left_by_transfer",
    "left_by_card",
    "left_by_cash",
    "payers_30d",
    "payees_30d",
    "new_internal_payers_7d",
    "new_internal_payers_30d",
    "new_external_payers_7d",
    "new_external_payers_30d",
    "new_payees_7d",
    "new_payees_30d",
    "new_payer_share_30d",
    "max_new_payers_24h",
    "top_payee_share_30d",
    "hub_payee_share_30d",
    "in_7d",
    "in_30d",
    "in_90d",
    "out_7d",
    "out_30d",
    "out_90d",
    "burst_in_1h",
    "burst_in_24h",
    "burst_out_1h",
    "burst_out_24h",
    *_duration("shortest_in_gap"),
    *_duration("since_last_in"),
    *_duration("since_last_out"),
    "degree_pays",
    "degree_rev_pays",
    "degree_rev_owned_by",
    "degree_rev_owner_enrolled_on_device",
    "degree_rev_owner_enrolled_on_ip",
    "degree_rev_seen_on_device",
    "degree_rev_seen_on_ip",
]
CUSTOMER_RAW = [
    "accounts_owned",
    "devices_30d",
    "ips_30d",
    "new_devices_30d",
    "new_ips_30d",
    "co_users",
    "new_co_users_30d",
    "co_users_both",
    "degree_owned_by",
    "degree_rev_enrolled_on_device",
    "degree_rev_enrolled_on_ip",
]
NODE_RAW = [
    "customers_30d",
    "customers_90d",
    "new_customers_7d",
    "new_customers_30d",
    "accounts_seen_30d",
    "payments_seen_30d",
    "observed_only",
    "crowd",
    "hub",
]
PAYMENT_RAW = [
    "amount",
    *(f"rail={rail}" for rail in ("ach", "card", "cash", "check", "internal", "unknown", "zelle")),
    *_duration("age"),
    "pair_prior_count",
    *_duration("pair_gap"),
    *_duration("pair_age"),
    *_duration("sender_gap"),
    *_duration("receiver_gap"),
    "dwell_tracked",
    "left_24h",
    *_duration("dwell"),
]
COMMUNITY_RAW = [
    "members",
    "capped",
    "shared_nodes",
    "density",
    "new_link_share_30d",
    "internal_amount_30d",
    "isolation",
    "member_payments_30d",
]


def test_every_column_is_in_the_order_of_the_features_tables() -> None:
    assert list(ACCOUNT_LAYOUT.raw_names) == ACCOUNT_RAW
    assert list(CUSTOMER_LAYOUT.raw_names) == CUSTOMER_RAW
    assert list(DEVICE_LAYOUT.raw_names) == [
        *NODE_RAW,
        "degree_enrolled_on_device",
        "degree_owner_enrolled_on_device",
        "degree_seen_on_device",
    ]
    assert list(IP_LAYOUT.raw_names) == [
        *NODE_RAW,
        "degree_enrolled_on_ip",
        "degree_owner_enrolled_on_ip",
        "degree_seen_on_ip",
    ]
    assert list(PAYMENT_LAYOUT.raw_names) == PAYMENT_RAW
    assert list(ENROLMENT_LAYOUT.raw_names) == [
        *_duration("since_start"),
        *_duration("since_last_seen"),
    ]
    assert list(SEEN_LAYOUT.raw_names) == [
        "uses_90d",
        *_duration("since_first_use"),
        *_duration("since_last_use"),
    ]
    assert list(COMMUNITY_LAYOUT.raw_names) == COMMUNITY_RAW


@pytest.mark.parametrize(
    ("layout", "raw", "encoded"),
    [
        # Stored before encoding 50 plus 7 degrees, encoded 309: 3 + 82 (B1) + 12 (B2) + 205
        # (B3) + 7.
        (ACCOUNT_LAYOUT, 57, 309),
        (CUSTOMER_LAYOUT, 11, 11),
        (DEVICE_LAYOUT, 12, 12),
        (IP_LAYOUT, 12, 12),
        # 1 + 7 + 65 + 1 + 4 x 65 + 1 + 1 + 65.
        (PAYMENT_LAYOUT, 23, 401),
        (ENROLMENT_LAYOUT, 4, 130),
        (SEEN_LAYOUT, 5, 131),
        (COMMUNITY_LAYOUT, 8, 8),
    ],
)
def test_the_widths_are_those_of_the_widths_table(layout: Layout, raw: int, encoded: int) -> None:
    assert (layout.raw_width, layout.width) == (raw, encoded)


def test_each_kind_of_column_is_the_features_encoding() -> None:
    kinds = {column.name: column.kind for column in ACCOUNT_LAYOUT.columns}
    assert kinds["is_external"] is kinds["flow_present"] is Kind.FLAG
    assert kinds["in_amount_7d"] is kinds["out_amount_90d"] is Kind.AMOUNT
    assert kinds["left_15m"] is kinds["held_share"] is kinds["hub_payee_share_30d"] is Kind.SHARE
    assert kinds["median_dwell"] is kinds["since_last_out"] is Kind.DURATION
    assert kinds["max_new_payers_24h"] is kinds["degree_pays"] is Kind.COUNT
    payment = {column.name: column.kind for column in PAYMENT_LAYOUT.columns}
    assert payment["amount"] is Kind.AMOUNT and payment["rail=zelle"] is Kind.FLAG
    assert payment["pair_prior_count"] is Kind.COUNT and payment["left_24h"] is Kind.SHARE
    community = {column.name: column.kind for column in COMMUNITY_LAYOUT.columns}
    assert community["internal_amount_30d"] is Kind.AMOUNT and community["capped"] is Kind.FLAG
    for layout in (CUSTOMER_LAYOUT, DEVICE_LAYOUT):
        assert {column.kind for column in layout.columns[:6]} == {Kind.COUNT}


def test_sixteen_relations_each_with_its_reverse_and_ownership_without_attributes() -> None:
    assert len(FORWARD) == 8 and len(RELATIONS) == 16 == len(set(RELATIONS))
    names = [name for _, name, _ in FORWARD]
    assert names == [
        "pays",
        "owned_by",
        "enrolled_on_device",
        "enrolled_on_ip",
        "owner_enrolled_on_device",
        "owner_enrolled_on_ip",
        "seen_on_device",
        "seen_on_ip",
    ]
    for source, name, target in FORWARD:
        assert (target, f"rev_{name}", source) in RELATIONS
    assert len(EDGE_LAYOUTS) == 14
    assert not {relation for relation in RELATIONS if "owned_by" in relation[1]} & set(EDGE_LAYOUTS)
    assert set(LAYOUT.nodes) == set(NODE_LAYOUTS) == {"account", "customer", "device", "ip"}


def test_each_node_type_counts_every_relation_arriving_at_it() -> None:
    for node_type, counted in DEGREES.items():
        assert set(counted) == {relation for relation in RELATIONS if relation[2] == node_type}
        assert all(relation[2] == node_type for relation in counted)
