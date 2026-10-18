"""The batch builder: the hand-built two-example batch of conftest.py with every column of every
row checked, nodes reached by two paths, flag-only hubs and leaf rows from read_hubs, B1's
dwell on its payment edges, its refusals, and a batch built from a fetched cache (docs/features.md,
"What one example's subgraph holds" and "The other tensors of a batch")."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pytest
import torch
from torch import Tensor

from mule_pattern_learner.subgraph import build as build_module
from mule_pattern_learner.subgraph.build import (
    Batch,
    Example,
    ExampleRows,
    Leaves,
    build,
    read_rows,
)
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.subgraph.layouts import (
    ACCOUNT_LAYOUT,
    ENROLLED_ON_DEVICE,
    ENROLLED_ON_IP,
    FORWARD,
    OWNED_BY,
    OWNER_ENROLLED_ON_DEVICE,
    OWNER_ENROLLED_ON_IP,
    PAYMENT_LAYOUT,
    PAYS,
    SEEN_ON_DEVICE,
    SEEN_ON_IP,
    reverse,
)
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T
from mule_pattern_learner.tigergraph.queries.read_hubs import Hubs, Leaf
from mule_pattern_learner.tigergraph.queries.read_money_flow import TrackedInflow
from mule_pattern_learner.tigergraph.queries.read_payment_events import OwnedBy
from mule_pattern_learner.tigergraph.queries.read_shared_identity import OwnerEnrolled

MIN = 60_000
T1 = T + 7 * DAY_MS


def flow(k: int) -> list[float]:
    """B1's columns 4 to 21 of the account numbered k in conftest.hand_flow: the amounts in
    and out over 7, 30 and 90 days, flow_present, the six dwell shares, held_share, the
    median dwell as presence and milliseconds (k minutes), and the three rail shares."""
    return [
        1000 * k + 1,
        1000 * k + 2,
        1000 * k + 3,
        1000 * k + 4,
        1000 * k + 5,
        1000 * k + 6,
        1,
        0.1 + k / 1000,
        0.2 + k / 1000,
        0.3 + k / 1000,
        0.4 + k / 1000,
        0.5 + k / 1000,
        0.6 + k / 1000,
        0.35 + k / 1000,
        1,
        k * MIN,
        0.11 + k / 1000,
        0.12 + k / 1000,
        0.13 + k / 1000,
    ]


def counterparties(k: int) -> list[float]:
    """B2's columns 22 to 33: payers and payees, new internal and external payers and new
    payees over 7 and 30 days, the new-payer share, the 24-hour burst of new payers, and the
    top and hub payee shares."""
    return [10 * k + n for n in range(1, 9)] + [
        0.7 + k / 1000,
        10 * k + 9,
        0.8 + k / 1000,
        0.9 + k / 1000,
    ]


def activity(k: int) -> list[float]:
    """B3's columns 34 to 46: the counts and bursts, the shortest gap (k seconds), the time
    since the last inflow (k hours), and no outflow in the lookback."""
    counts: list[float] = [100 * k + n for n in range(11, 21)]
    return counts + [1, k * 1000, 1, k * HOUR_MS, 0, 0]


def full(is_external: int, hub: int, measure: int, k: int, degrees: Sequence[float]) -> list[float]:
    return [is_external, hub, measure, *flow(k), *counterparties(k), *activity(k), *degrees]


def flag_only(measure: int, degrees: Sequence[float]) -> list[float]:
    return [0, 1, measure, *[0] * 47, *degrees]


# The account nodes in the batch's order, each example's sorted by id, with their degree
# columns: payments received, payments sent, ownership, owner-enrolment on a device and on an
# IP, seen on a device and on an IP.
ACCOUNTS: list[tuple[int, str]] = [
    (0, "a_co"),
    (0, "a_hub"),
    (0, "a_own"),
    (0, "a_s1"),
    (0, "a_two"),
    (0, "x_payer"),
    (1, "a_s2"),
    (1, "a_three"),
    (1, "x_payer"),
]
EXPECTED_ACCOUNTS = [
    full(0, 0, 0, 5, [0, 0, 0, 1, 0, 0, 0]),
    # The hub a_hub is not scored: is_external, hub, its measure and its degrees alone.
    flag_only(250, [1, 0, 0, 0, 0, 0, 0]),
    full(0, 0, 0, 4, [1, 0, 1, 1, 0, 0, 0]),
    full(0, 0, 0, 1, [1, 2, 1, 1, 1, 2, 0]),
    full(0, 0, 0, 3, [0, 1, 0, 0, 0, 0, 1]),
    full(1, 0, 0, 2, [1, 1, 0, 0, 0, 0, 0]),
    # a_s2 is a hub and the scored account: its full row, with its hub flag and measure.
    full(0, 1, 300, 6, [1, 1, 1, 0, 0, 0, 1]),
    full(0, 0, 0, 7, [1, 0, 0, 0, 0, 0, 0]),
    full(1, 0, 0, 8, [0, 1, 0, 0, 0, 0, 0]),
]
# The customers: B5's row, then ownership edges, enrolment on a device and on an IP.
EXPECTED_CUSTOMERS = [
    [2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1],
    [1, 1, 2, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
]
# Devices and IPs: customers_30d, customers_90d, new over 7 and 30 days, accounts and payments
# seen, observed_only, crowd, hub, then enrolment, owner-enrolment and seen edges. A leaf's
# measure, observed_only and hub come from read_hubs, B5's row giving its flag alone.
EXPECTED_DEVICES = [
    [0, 12, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [2, 3, 1, 2, 2, 5, 0, 0, 0, 2, 3, 1],
]
EXPECTED_IPS = [
    [1, 2, 0, 1, 1, 4, 0, 0, 0, 1, 1, 1],
    [0, 220, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
]


def rail(name: str) -> list[float]:
    rails = ("ach", "card", "cash", "check", "internal", "unknown", "zelle")
    return [float(name == other) for other in rails]


# The payment edges by payment id: amount in cents, the rail, age (always present), the
# earlier pair payments, the pair gap and age, the sender and receiver gaps, then B1's
# dwell_tracked, left_24h and dwell.
EXPECTED_PAYMENTS = [
    # p1, x_payer to a_s1, tracked by a_s1: 95% left within a day, half within 10 minutes.
    [50_000, *rail("zelle"), 1, 3 * DAY_MS, 0, 0, 0, 0, 0, 1, 2 * HOUR_MS, 1, 5 * DAY_MS]
    + [1, 0.95, 1, 10 * MIN],
    # p2, a_s1 to the hub a_hub: no dwell at a hub end.
    [1_999, *rail("card"), 1, 2 * DAY_MS, 2, 1, 30 * DAY_MS, 1, 60 * DAY_MS, 1, DAY_MS, 0, 0]
    + [0, 0, 0, 0],
    # p3, a_two to x_payer, tracked by x_payer: nothing left, so no dwell.
    [70_000, *rail("ach"), 1, 4 * DAY_MS, 1, 1, 7 * DAY_MS, 1, 7 * DAY_MS, 0, 0, 0, 0]
    + [1, 0, 0, 0],
    # p4, a_s1 to a_own, which tracked no lot of it.
    [47_500, *rail("internal"), 1, 3 * DAY_MS - 10 * MIN, 0, 0, 0, 0, 0, 1, 10 * MIN, 1]
    + [6 * DAY_MS, 0, 0, 0, 0],
    # p5, x_payer to a_s2, the scored hub: B1 tracked it, but a hub end carries no dwell.
    [1_250, *rail("check"), 1, DAY_MS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # p6, a_s2 to a_three, tracked by a_three: half left within 2 hours, so the dwell is 2 hours.
    [900, *rail("unknown"), 1, 6 * HOUR_MS, 0, 0, 0, 0, 0, 0, 0, 1, 2 * DAY_MS, 1, 0.5, 1]
    + [2 * HOUR_MS],
]
# Edges by relation, as (source, target) positions in the batch.
EXPECTED_EDGES = {
    PAYS: [(5, 3), (3, 1), (4, 5), (3, 2), (8, 6), (6, 7)],
    OWNED_BY: [(2, 0), (3, 0), (6, 2)],
    ENROLLED_ON_DEVICE: [(0, 1), (1, 1)],
    ENROLLED_ON_IP: [(0, 0)],
    OWNER_ENROLLED_ON_DEVICE: [(0, 1), (2, 1), (3, 1)],
    OWNER_ENROLLED_ON_IP: [(3, 0)],
    SEEN_ON_DEVICE: [(3, 0), (3, 1)],
    SEEN_ON_IP: [(4, 0), (6, 1)],
}
# The enrolment and owner-enrolment edges: since_start and since_last_seen, each as presence
# and milliseconds; the seen edges: uses, since_first_use and since_last_use.
EXPECTED_ATTRS = {
    ENROLLED_ON_DEVICE: [[1, 20 * DAY_MS, 1, 3 * DAY_MS], [1, 5 * DAY_MS, 0, 0]],
    ENROLLED_ON_IP: [[0, 0, 1, 3 * DAY_MS]],
    OWNER_ENROLLED_ON_DEVICE: [
        [1, 5 * DAY_MS, 0, 0],
        [1, 20 * DAY_MS, 1, 3 * DAY_MS],
        [1, 20 * DAY_MS, 1, 3 * DAY_MS],
    ],
    OWNER_ENROLLED_ON_IP: [[0, 0, 1, 3 * DAY_MS]],
    SEEN_ON_DEVICE: [[2, 0, 0, 1, 3 * DAY_MS], [1, 1, 3 * DAY_MS, 1, 3 * DAY_MS]],
    SEEN_ON_IP: [[4, 1, 40 * DAY_MS, 1, 4 * DAY_MS], [1, 0, 0, 1, DAY_MS]],
}
EXPECTED_COMMUNITY = [
    [3, 0, 1, 0.75, 0.5, 47_500, 0.25, 4],
    [1, 0, 0, 0, 0, 0, 0, 7],
]


def _equal(found: Tensor, expected: Sequence[Sequence[float]]) -> None:
    assert found.dtype == torch.float64
    np.testing.assert_array_equal(found.numpy(), np.array(expected, dtype=np.float64))


def _pairs(edge_index: Tensor) -> list[tuple[int, int]]:
    return [(int(source), int(target)) for source, target in edge_index.T.tolist()]


def test_every_column_of_the_hand_built_batch(hand_batch: Batch) -> None:
    data = hand_batch.data
    _equal(data["account"].x, EXPECTED_ACCOUNTS)
    _equal(data["customer"].x, EXPECTED_CUSTOMERS)
    _equal(data["device"].x, EXPECTED_DEVICES)
    _equal(data["ip"].x, EXPECTED_IPS)
    for relation, pairs in EXPECTED_EDGES.items():
        assert _pairs(data[relation].edge_index) == pairs, relation
    _equal(data[PAYS].edge_attr, EXPECTED_PAYMENTS)
    for relation, attrs in EXPECTED_ATTRS.items():
        _equal(data[relation].edge_attr, attrs)
    assert "edge_attr" not in data[OWNED_BY] and "edge_attr" not in data[reverse(OWNED_BY)]
    assert data.scored_index.tolist() == [3, 6]
    _equal(data.community, EXPECTED_COMMUNITY)
    assert data.label.tolist() == [1, 0] and data.label.dtype == torch.long
    assert data.weight.tolist() == [2.5, 0.75]
    assert hand_batch.keys == (Example("w", T, "a_s1"), Example("w", T1, "a_s2"))


def test_every_relation_has_its_reverse_with_the_same_attributes(hand_batch: Batch) -> None:
    data = hand_batch.data
    assert len(data.edge_types) == 16
    for relation in FORWARD:
        backward = data[reverse(relation)]
        assert _pairs(backward.edge_index) == [
            (target, source) for source, target in _pairs(data[relation].edge_index)
        ]
        if "edge_attr" in data[relation]:
            assert torch.equal(backward.edge_attr, data[relation].edge_attr)


def test_a_node_reached_by_two_paths_appears_once_in_its_example(hand_batch: Batch) -> None:
    """a_two is a hop-2 account of B4 and a co-user's account of B5, and d_home is both a
    device of p_c1 and one seen on a_s1's payments: one node each. x_payer is in both examples,
    so it has a copy in each, with its own Monday's rows."""
    rows = hand_batch.data["account"].x
    assert rows.shape[0] == 9 and hand_batch.data["device"].num_nodes == 2
    # B1's in_amount_7d is 1000 k + 1 for the account numbered k: a_two is 3, x_payer 2 at T
    # and 8 at T + 1 week.
    numbered = rows[:, ACCOUNT_LAYOUT.index("in_amount_7d")].tolist()
    assert numbered.count(3001) == 1
    assert numbered.index(2001) == ACCOUNTS.index((0, "x_payer"))
    assert numbered.index(8001) == ACCOUNTS.index((1, "x_payer"))


def test_a_scoring_batch_is_the_training_batch_without_labels_and_weights(
    hand: list[ExampleRows], hand_batch: Batch
) -> None:
    scoring = build(hand, "w")
    assert "label" not in scoring.data and "weight" not in scoring.data
    assert scoring.keys == hand_batch.keys
    training = hand_batch.data
    assert set(scoring.data.node_types) == set(training.node_types)
    assert set(scoring.data.edge_types) == set(training.edge_types)
    for kind in training.node_types:
        assert torch.equal(scoring.data[kind].x, training[kind].x)
    for relation in training.edge_types:
        for key in ("edge_index", "edge_attr"):
            if key in training[relation]:
                assert torch.equal(scoring.data[relation][key], training[relation][key])
    assert torch.equal(scoring.data.scored_index, training.scored_index)
    assert torch.equal(scoring.data.community, training.community)


def test_a_leaf_flag_unlike_read_hubs_is_refused(hand: list[ExampleRows]) -> None:
    first = hand[0]
    without = Leaves.of(Hubs((), (), ()))
    with pytest.raises(ValueError, match="d_crowd as a leaf, unlike read_hubs"):
        build([replace(first, leaves=replace(without, accounts={"a_hub": 250}))], "w")
    marked = replace(
        first.leaves, devices={**first.leaves.devices, "d_home": Leaf("d_home", 9, False, False)}
    )
    with pytest.raises(ValueError, match="d_home as no leaf, unlike read_hubs"):
        build([replace(first, leaves=marked)], "w")
    # read_hubs names d_crowd among its IPs, not its devices.
    moved = replace(first.leaves, devices={}, ips={"d_crowd": first.leaves.devices["d_crowd"]})
    with pytest.raises(ValueError, match="d_crowd as a leaf"):
        build([replace(first, leaves=moved)], "w")


def test_rows_the_builder_cannot_join_are_refused(hand: list[ExampleRows]) -> None:
    first = hand[0]
    flows = {account: row for account, row in first.flows.items() if account != "a_two"}
    with pytest.raises(ValueError, match="lacks B1 to B3 rows of a_two"):
        build([replace(first, flows=flows)], "w")
    owned = replace(first.payments, owned_by=(*first.payments.owned_by, OwnedBy("a_two", "p_x")))
    with pytest.raises(ValueError, match="p_x, which has no customer row"):
        build([replace(first, payments=owned)], "w")
    inflows = {**first.inflows, "p4": TrackedInflow("a_s1", "p4", 0.5, False, 0)}
    with pytest.raises(ValueError, match="tracked p4 for a_s1, not its receiver"):
        build([replace(first, inflows=inflows)], "w")
    hub_edge = OwnerEnrolled("a_hub", "d_home", False, 0, False, 0)
    identity = replace(first.identity, owner_enrolled=(*first.identity.owner_enrolled, hub_edge))
    with pytest.raises(ValueError, match="account hub a_hub an edge"):
        build([replace(first, identity=identity)], "w")
    accounts = tuple(
        replace(account, is_external=True) if account.account_id == "a_two" else account
        for account in first.identity.accounts
    )
    with pytest.raises(ValueError, match="two values of is_external"):
        build([replace(first, identity=replace(first.identity, accounts=accounts))], "w")
    other = replace(first, community=replace(first.community, account_id="a_two"))
    with pytest.raises(ValueError, match="belong to other accounts"):
        build([other], "w")


def test_labels_and_weights_go_together_and_are_checked(hand: list[ExampleRows]) -> None:
    with pytest.raises(ValueError, match="labels and weights, a scoring batch neither"):
        build(hand, "w", labels=[1, 0])
    with pytest.raises(ValueError, match="needs 2 labels and weights"):
        build(hand, "w", labels=[1], weights=[1.0])
    with pytest.raises(ValueError, match="A label is one of"):
        build(hand, "w", labels=[1, 2], weights=[1.0, 1.0])
    for weight in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite number above 0"):
            build(hand, "w", labels=[1, 0], weights=[1.0, weight])
    with pytest.raises(ValueError, match="at least one example"):
        build([], "w")


def test_a_value_outside_its_column_is_refused(hand: list[ExampleRows]) -> None:
    first = hand[0]
    flows = {**first.flows, "a_s1": replace(first.flows["a_s1"], left_15m=1.5)}
    with pytest.raises(ValueError, match="account.left_15m holds a share outside"):
        build([replace(first, flows=flows)], "w")
    flows = {**first.flows, "a_s1": replace(first.flows["a_s1"], median_dwell_present=False)}
    with pytest.raises(ValueError, match="account.median_dwell is absent but not 0"):
        build([replace(first, flows=flows)], "w")
    # A count, cents or milliseconds is a whole number below 2**53, which float64 holds
    # exactly: a fraction, a negative or 2**53 itself is refused.
    for count in (1.5, -1, 2**53):
        rows = {
            **first.counterparties,
            "a_s1": replace(first.counterparties["a_s1"], payers_30d=count),
        }
        with pytest.raises(
            ValueError, match="account.payers_30d holds a value that is not a whole"
        ):
            build([replace(first, counterparties=rows)], "w")
    for cents in (0.5, 2**53):
        flows = {**first.flows, "a_s1": replace(first.flows["a_s1"], in_amount_7d=cents)}
        with pytest.raises(
            ValueError, match="account.in_amount_7d holds a value that is not a whole"
        ):
            build([replace(first, flows=flows)], "w")
    for milliseconds in (1.5, -1, 2**53):
        flows = {**first.flows, "a_s1": replace(first.flows["a_s1"], median_dwell_ms=milliseconds)}
        with pytest.raises(ValueError, match="account.median_dwell_ms holds a value that is not"):
            build([replace(first, flows=flows)], "w")


# The modules that build a batch, and the fields of every row they read.
BUILDER_MODULES = (
    "subgraph/build.py",
    "subgraph/layouts.py",
    "subgraph/degrees.py",
    "subgraph/switches.py",
    "model/encode.py",
)
DATES_OF_RECORD = ("opened_ms", "joined_ms", "closed_ms")
LABEL_FIELDS = ("is_mule", "pu_label", "mule_ring_id", "mule_onset_ms", "mule_last_ms")


def test_the_builder_never_reads_a_date_of_record_or_a_label() -> None:
    """No dates of record: opened_ms, joined_ms and closed_ms never reach a row, and no
    label either. The builder's modules name none, and no row it reads has such a field."""
    package = Path(build_module.__file__).parent.parent
    for module in BUILDER_MODULES:
        names: set[str] = set()
        for node in ast.walk(ast.parse((package / module).read_text())):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
        assert not names & {*DATES_OF_RECORD, *LABEL_FIELDS}, module
    read: list[type] = [ExampleRows]
    kinds: set[type] = set()
    while read:
        kind = read.pop()
        if kind in kinds:
            continue
        kinds.add(kind)
        hints = list(get_type_hints(kind).values())
        while hints:
            hint = hints.pop()
            hints += get_args(hint)
            if isinstance(hint, type) and is_dataclass(hint):
                read.append(hint)
    assert {kind.__name__ for kind in kinds} >= {"MoneyFlow", "PaymentEvent", "Seen", "Leaf"}
    names = {field.name for kind in kinds for field in fields(kind)}
    assert not names & {*DATES_OF_RECORD, *LABEL_FIELDS}


# A fetched cache: the world of conftest.py at T.
SCORED = ["a_c000", "a_shop", "a_h1", "a_r1", "a_z", "a_mule"]


@pytest.fixture
def fetched(graph: ReferenceGraph, tmp_path: Path, parity: Path) -> Cache:
    cache = Cache(tmp_path)
    fetch(Spy(graph), cache, {T: SCORED}, parity=parity)
    return cache


def _positions(rows: list[ExampleRows], example: int) -> dict[str, dict[str, int]]:
    """The position in a batch built from rows of each account and device node of one
    example, by id."""

    def accounts(found: ExampleRows) -> list[str]:
        return sorted(build_module.account_nodes(found.payments, found.identity))

    def devices(found: ExampleRows) -> list[str]:
        return [row.node_id for row in found.identity.devices]

    positions: dict[str, dict[str, int]] = {}
    for kind, ids in (("account", accounts), ("device", devices)):
        first = sum(len(ids(earlier)) for earlier in rows[:example])
        positions[kind] = {node: first + at for at, node in enumerate(ids(rows[example]))}
    return positions


def test_a_batch_built_from_the_cache(fetched: Cache) -> None:
    keys = [(T, account) for account in SCORED]
    rows = read_rows(fetched, keys)
    assert [(found.t_ms, found.scored) for found in rows] == keys
    batch = build(rows, "w")
    data = batch.data
    x = data["account"].x
    assert [example.account_id for example in batch.keys] == SCORED
    # a_shop scored: its full row with its hub flag and its measure, 201 customers.
    shop = x[_positions(rows, 1)["account"]["a_shop"]]
    assert shop[ACCOUNT_LAYOUT.index("hub")] == 1
    assert shop[ACCOUNT_LAYOUT.index("hub_customers_90d")] == 201
    assert shop[ACCOUNT_LAYOUT.index("in_30d")] > 0
    # a_c000 paid a_shop: in its example the shop is flag-only, and the edge has no dwell.
    leaf = _positions(rows, 0)["account"]["a_shop"]
    assert x[leaf, : ACCOUNT_LAYOUT.index("degree_pays")].tolist() == [0, 1, 201] + [0] * 47
    into_shop = data[PAYS].edge_index[1] == leaf
    assert bool(into_shop.any())
    assert not data[PAYS].edge_attr[into_shop][:, PAYMENT_LAYOUT.index("dwell_tracked")].any()
    # a_mule's victims' payments carry its walk's dwell.
    assert data[PAYS].edge_attr[:, PAYMENT_LAYOUT.index("dwell_tracked")].sum() > 0
    # a_z has no payment, and its example joins a_r1 to the ring's device d_ring.
    assert not rows[4].payments.payments
    at = _positions(rows, 4)
    edges = {
        relation: _pairs(data[relation].edge_index)
        for relation in (OWNER_ENROLLED_ON_DEVICE, SEEN_ON_DEVICE)
    }
    ring = (at["account"]["a_r1"], at["device"]["d_ring"])
    assert any(ring in pairs for pairs in edges.values())
    assert (at["account"]["a_z"], at["device"]["d_ring"]) in edges[OWNER_ENROLLED_ON_DEVICE]


def test_rows_read_from_the_cache_keep_the_order_asked(fetched: Cache) -> None:
    keys = [(T, "a_r1"), (T, "a_c000"), (T, "a_r1")]
    rows = read_rows(fetched, keys)
    assert [found.scored for found in rows] == ["a_r1", "a_c000", "a_r1"]
    # Each holds the B1 to B3 rows of its own account nodes but the hubs, and its own
    # payments' tracked inflows only.
    c000 = rows[1]
    nodes = build_module.account_nodes(c000.payments, c000.identity)
    assert set(c000.flows) == set(nodes) - {"a_shop"}
    assert set(c000.inflows) <= {payment.payment_id for payment in c000.payments.payments}
    with pytest.raises(KeyError, match="no hubs"):
        read_rows(fetched, [(T + 7 * DAY_MS, "a_r1")])
