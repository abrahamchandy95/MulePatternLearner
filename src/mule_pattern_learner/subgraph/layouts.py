"""The node types, the sixteen relations and the column order of every row of a batch, exactly
as docs/features.md's tables give them, raw before encoding (docs/features.md, "Widths").

Raw, each duration is two columns, its presence bit and its milliseconds, and each payment's
rail seven one-hot flags. Raw widths: account 50 plus 7 degree columns, customer 8 plus 3,
device and IP 9 plus 3 each, payment edge 23, enrolment edge 4, seen edge 5, community 8.
Encoded: account 309, customer 11, device and IP 12, payment edge 401, enrolment edge 130,
seen edge 131, community 8.

The degree columns come last in each node row, one per relation arriving at the node type, in
the order of docs/features.md's "Degree columns"; they hold the raw number of edges, which the
encoding turns into log(1 + n).
"""

from __future__ import annotations

from ..model.encode import BatchLayout, Column, EdgeType, Kind, Layout
from ..tigergraph.queries.read_payment_events import RAILS

# The node types.
ACCOUNT = "account"
CUSTOMER = "customer"
DEVICE = "device"
IP = "ip"
NODE_TYPES = (ACCOUNT, CUSTOMER, DEVICE, IP)

# The eight forward relations; each has a reverse, rev_ and its name, from its target to its
# source, with the same attributes.
PAYS: EdgeType = (ACCOUNT, "pays", ACCOUNT)
OWNED_BY: EdgeType = (ACCOUNT, "owned_by", CUSTOMER)
ENROLLED_ON_DEVICE: EdgeType = (CUSTOMER, "enrolled_on_device", DEVICE)
ENROLLED_ON_IP: EdgeType = (CUSTOMER, "enrolled_on_ip", IP)
OWNER_ENROLLED_ON_DEVICE: EdgeType = (ACCOUNT, "owner_enrolled_on_device", DEVICE)
OWNER_ENROLLED_ON_IP: EdgeType = (ACCOUNT, "owner_enrolled_on_ip", IP)
SEEN_ON_DEVICE: EdgeType = (ACCOUNT, "seen_on_device", DEVICE)
SEEN_ON_IP: EdgeType = (ACCOUNT, "seen_on_ip", IP)
FORWARD: tuple[EdgeType, ...] = (
    PAYS,
    OWNED_BY,
    ENROLLED_ON_DEVICE,
    ENROLLED_ON_IP,
    OWNER_ENROLLED_ON_DEVICE,
    OWNER_ENROLLED_ON_IP,
    SEEN_ON_DEVICE,
    SEEN_ON_IP,
)
# The relations from a node to a device or an IP, with their kind of node.
IDENTITY_RELATIONS: tuple[EdgeType, ...] = FORWARD[2:]


def reverse(relation: EdgeType) -> EdgeType:
    """The reverse of a forward relation."""
    source, name, target = relation
    return (target, f"rev_{name}", source)


RELATIONS: tuple[EdgeType, ...] = tuple(
    relation for forward in FORWARD for relation in (forward, reverse(forward))
)

# The relations counted in each node type's degree columns, every relation arriving at it, in
# the order of docs/features.md's "Degree columns".
DEGREES: dict[str, tuple[EdgeType, ...]] = {
    ACCOUNT: (
        PAYS,
        reverse(PAYS),
        reverse(OWNED_BY),
        reverse(OWNER_ENROLLED_ON_DEVICE),
        reverse(OWNER_ENROLLED_ON_IP),
        reverse(SEEN_ON_DEVICE),
        reverse(SEEN_ON_IP),
    ),
    CUSTOMER: (OWNED_BY, reverse(ENROLLED_ON_DEVICE), reverse(ENROLLED_ON_IP)),
    DEVICE: (ENROLLED_ON_DEVICE, OWNER_ENROLLED_ON_DEVICE, SEEN_ON_DEVICE),
    IP: (ENROLLED_ON_IP, OWNER_ENROLLED_ON_IP, SEEN_ON_IP),
}


def degree_name(relation: EdgeType) -> str:
    """The name of the degree column that counts a relation's edges."""
    return f"degree_{relation[1]}"


def _columns(kind: Kind, *names: str) -> tuple[Column, ...]:
    return tuple(Column(name, kind) for name in names)


def _degrees(node_type: str) -> tuple[Column, ...]:
    return _columns(Kind.COUNT, *(degree_name(relation) for relation in DEGREES[node_type]))


# The account node, columns 1 to 46 and its degree columns (docs/features.md, "Account node").
# B1's columns 4 to 21, B2's 22 to 33 and B3's 34 to 46.
B1_COLUMNS: tuple[Column, ...] = (
    *_columns(
        Kind.AMOUNT,
        "in_amount_7d",
        "in_amount_30d",
        "in_amount_90d",
        "out_amount_7d",
        "out_amount_30d",
        "out_amount_90d",
    ),
    Column("flow_present", Kind.FLAG),
    *_columns(
        Kind.SHARE,
        "left_15m",
        "left_1h",
        "left_6h",
        "left_1d",
        "left_3d",
        "left_7d",
        "held_share",
    ),
    Column("median_dwell", Kind.DURATION),
    *_columns(Kind.SHARE, "left_by_transfer", "left_by_card", "left_by_cash"),
)
B2_COLUMNS: tuple[Column, ...] = (
    *_columns(
        Kind.COUNT,
        "payers_30d",
        "payees_30d",
        "new_internal_payers_7d",
        "new_internal_payers_30d",
        "new_external_payers_7d",
        "new_external_payers_30d",
        "new_payees_7d",
        "new_payees_30d",
    ),
    Column("new_payer_share_30d", Kind.SHARE),
    Column("max_new_payers_24h", Kind.COUNT),
    *_columns(Kind.SHARE, "top_payee_share_30d", "hub_payee_share_30d"),
)
B3_COLUMNS: tuple[Column, ...] = (
    *_columns(
        Kind.COUNT,
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
    ),
    *_columns(Kind.DURATION, "shortest_in_gap", "since_last_in", "since_last_out"),
)
ACCOUNT_LAYOUT = Layout(
    (
        Column("is_external", Kind.FLAG),
        Column("hub", Kind.FLAG),
        Column("hub_customers_90d", Kind.COUNT),
        *B1_COLUMNS,
        *B2_COLUMNS,
        *B3_COLUMNS,
        *_degrees(ACCOUNT),
    )
)
# The customer node, columns 1 to 8 and its degree columns.
CUSTOMER_COLUMNS: tuple[Column, ...] = _columns(
    Kind.COUNT,
    "accounts_owned",
    "devices_30d",
    "ips_30d",
    "new_devices_30d",
    "new_ips_30d",
    "co_users",
    "new_co_users_30d",
    "co_users_both",
)
CUSTOMER_LAYOUT = Layout((*CUSTOMER_COLUMNS, *_degrees(CUSTOMER)))
# The device and IP nodes, columns 1 to 9 and their degree columns.
NODE_COLUMNS: tuple[Column, ...] = (
    *_columns(
        Kind.COUNT,
        "customers_30d",
        "customers_90d",
        "new_customers_7d",
        "new_customers_30d",
        "accounts_seen_30d",
        "payments_seen_30d",
    ),
    *_columns(Kind.FLAG, "observed_only", "crowd", "hub"),
)
DEVICE_LAYOUT = Layout((*NODE_COLUMNS, *_degrees(DEVICE)))
IP_LAYOUT = Layout((*NODE_COLUMNS, *_degrees(IP)))
# The payment edge, attributes 1 to 17 (docs/features.md, "Payment edges").
PAYMENT_LAYOUT = Layout(
    (
        Column("amount", Kind.AMOUNT),
        *_columns(Kind.FLAG, *(f"rail={rail}" for rail in RAILS)),
        Column("age", Kind.DURATION),
        Column("pair_prior_count", Kind.COUNT),
        *_columns(Kind.DURATION, "pair_gap", "pair_age", "sender_gap", "receiver_gap"),
        Column("dwell_tracked", Kind.FLAG),
        Column("left_24h", Kind.SHARE),
        Column("dwell", Kind.DURATION),
    )
)
# The enrolment edge, of a customer's link and of its account-level copy, and the seen edge.
ENROLMENT_LAYOUT = Layout(_columns(Kind.DURATION, "since_start", "since_last_seen"))
SEEN_LAYOUT = Layout(
    (
        Column("uses_90d", Kind.COUNT),
        *_columns(Kind.DURATION, "since_first_use", "since_last_use"),
    )
)
# The community block (docs/features.md, "The community block").
COMMUNITY_LAYOUT = Layout(
    (
        Column("members", Kind.COUNT),
        Column("capped", Kind.FLAG),
        Column("shared_nodes", Kind.COUNT),
        *_columns(Kind.SHARE, "density", "new_link_share_30d"),
        Column("internal_amount_30d", Kind.AMOUNT),
        Column("isolation", Kind.SHARE),
        Column("member_payments_30d", Kind.COUNT),
    )
)

NODE_LAYOUTS: dict[str, Layout] = {
    ACCOUNT: ACCOUNT_LAYOUT,
    CUSTOMER: CUSTOMER_LAYOUT,
    DEVICE: DEVICE_LAYOUT,
    IP: IP_LAYOUT,
}
# The attributes of each forward relation; ownership carries none.
FORWARD_EDGE_LAYOUTS: dict[EdgeType, Layout] = {
    PAYS: PAYMENT_LAYOUT,
    ENROLLED_ON_DEVICE: ENROLMENT_LAYOUT,
    ENROLLED_ON_IP: ENROLMENT_LAYOUT,
    OWNER_ENROLLED_ON_DEVICE: ENROLMENT_LAYOUT,
    OWNER_ENROLLED_ON_IP: ENROLMENT_LAYOUT,
    SEEN_ON_DEVICE: SEEN_LAYOUT,
    SEEN_ON_IP: SEEN_LAYOUT,
}
EDGE_LAYOUTS: dict[EdgeType, Layout] = {
    relation: layout
    for forward, layout in FORWARD_EDGE_LAYOUTS.items()
    for relation in (forward, reverse(forward))
}
# Every layout of a batch, which the model is built from.
LAYOUT = BatchLayout(NODE_LAYOUTS, RELATIONS, EDGE_LAYOUTS, COMMUNITY_LAYOUT, ACCOUNT)
