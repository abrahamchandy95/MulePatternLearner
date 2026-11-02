"""The XGBoost vectors (docs/features.md, "Ablation and comparison switches", XGBoost inputs):
their columns in the order of docs/features.md, every value of a hand-built batch worked out by
hand, and the vectors of the batch world's batches read from the rows the GNN reads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from mule_pattern_learner.comparisons.features import (
    GRAPH_COLUMNS,
    OWN_COLUMNS,
    Vectors,
    vectors,
    vectors_of,
)
from mule_pattern_learner.subgraph.build import Batch, Example
from mule_pattern_learner.subgraph.layouts import (
    ACCOUNT,
    ACCOUNT_LAYOUT,
    CUSTOMER,
    DEVICE,
    ENROLLED_ON_DEVICE,
    ENROLLED_ON_IP,
    IP,
    OWNED_BY,
)


def test_the_own_vector_is_the_scored_accounts_columns_1_to_46() -> None:
    """50 numbers: is_external, hub, hub_customers_90d, B1 (the median dwell as presence and
    seconds), B2 and B3 (three durations as presence and seconds)."""
    assert len(OWN_COLUMNS) == 50
    assert OWN_COLUMNS[:4] == ("is_external", "hub", "hub_customers_90d", "in_amount_7d")
    assert OWN_COLUMNS[17:19] == ("median_dwell_present", "median_dwell_s")
    assert OWN_COLUMNS[19:24] == (
        "left_by_transfer",
        "left_by_card",
        "left_by_cash",
        "payers_30d",
        "payees_30d",
    )
    assert OWN_COLUMNS[34:38] == ("in_7d", "in_30d", "in_90d", "out_7d")
    assert OWN_COLUMNS[-6:] == (
        "shortest_in_gap_present",
        "shortest_in_gap_s",
        "since_last_in_present",
        "since_last_in_s",
        "since_last_out_present",
        "since_last_out_s",
    )
    assert not any(name.startswith("degree_") for name in GRAPH_COLUMNS)


def test_the_graph_vector_adds_customer_devices_ips_and_community() -> None:
    """50 + 8 + 18 + 18 + 8 = 102 numbers."""
    assert len(GRAPH_COLUMNS) == 102 and GRAPH_COLUMNS[:50] == OWN_COLUMNS
    assert GRAPH_COLUMNS[50:58] == tuple(
        f"customer.{name}"
        for name in (
            "accounts_owned",
            "devices_30d",
            "ips_30d",
            "new_devices_30d",
            "new_ips_30d",
            "co_users",
            "new_co_users_30d",
            "co_users_both",
        )
    )
    node = (
        "customers_30d",
        "customers_90d",
        "new_customers_7d",
        "new_customers_30d",
        "accounts_seen_30d",
        "payments_seen_30d",
        "observed_only",
        "crowd",
        "hub",
    )
    for start, prefix in ((58, "device_mean"), (67, "device_max"), (76, "ip_mean"), (85, "ip_max")):
        assert GRAPH_COLUMNS[start : start + 9] == tuple(f"{prefix}.{name}" for name in node)
    assert GRAPH_COLUMNS[94:] == tuple(
        f"community.{name}"
        for name in (
            "members",
            "capped",
            "shared_nodes",
            "density",
            "new_link_share_30d",
            "internal_amount_30d",
            "isolation",
            "member_payments_30d",
        )
    )


def _hand_batch() -> Batch:
    """Two examples. Example 0 scores account 0, owned by customer 0, which is enrolled on the
    devices 0 and 1 and on no IP; account 1, its neighbour, is owned by customer 1, enrolled on
    device 2. Example 1 scores account 2, owned by customer 2, enrolled on device 3 and IP 0.
    Each account row is its node number times 1,000 plus its raw column's position, so a
    duration of i milliseconds reads i / 1,000 seconds; each other row's values are given."""
    data = HeteroData()
    data[ACCOUNT].x = torch.tensor(
        [[1000.0 * node + column for column in range(57)] for node in range(3)],
        dtype=torch.float64,
    )
    data[CUSTOMER].x = torch.tensor(
        [[10.0 * node + column for column in range(11)] for node in range(3)],
        dtype=torch.float64,
    )
    data[DEVICE].x = torch.tensor(
        [
            [1, 2, 0, 0, 1, 4, 0, 0, 0, 9, 9, 9],
            [3, 2, 1, 1, 0, 0, 1, 0, 0, 9, 9, 9],
            [50, 50, 50, 50, 50, 50, 1, 1, 1, 9, 9, 9],
            [2, 2, 2, 2, 2, 2, 0, 0, 0, 9, 9, 9],
        ],
        dtype=torch.float64,
    )
    data[IP].x = torch.tensor([[5, 6, 1, 1, 2, 3, 1, 0, 0, 9, 9, 9]], dtype=torch.float64)
    data[OWNED_BY].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data[ENROLLED_ON_DEVICE].edge_index = torch.tensor([[0, 1, 0, 2], [0, 2, 1, 3]])
    data[ENROLLED_ON_IP].edge_index = torch.tensor([[2], [0]])
    data.scored_index = torch.tensor([0, 2])
    data.community = torch.tensor(
        [[3, 0, 1, 0.5, 0.25, 4000, 0.75, 6], [1, 0, 0, 0, 0, 0, 0, 2]], dtype=torch.float64
    )
    return Batch((Example("w", 10, "a0"), Example("w", 10, "a2")), data)


def test_every_value_of_a_hand_built_batch() -> None:
    found = vectors(_hand_batch())
    assert found.keys == ((10, "a0"), (10, "a2"))
    durations = [
        ACCOUNT_LAYOUT.index(f"{name}_ms")
        for name in (
            "median_dwell",
            "shortest_in_gap",
            "since_last_in",
            "since_last_out",
        )
    ]
    assert durations == [18, 45, 47, 49]
    assert found.own.dtype == found.graph.dtype == np.float32
    for row, node in enumerate((0, 2)):
        # The seconds are divided in float64 and then held as float32.
        expected = np.array([1000.0 * node + column for column in range(50)])
        expected[durations] /= 1000
        np.testing.assert_array_equal(found.own[row], expected.astype(np.float32))
        np.testing.assert_array_equal(found.graph[row, :50], expected.astype(np.float32))
        # The customer's columns 1 to 8, never its degree columns.
        np.testing.assert_array_equal(
            found.graph[row, 50:58], [10.0 * node + column for column in range(8)]
        )
    # Example 0: the mean and maximum of devices 0 and 1, never device 2 of the neighbour's
    # customer; no IP, so zeros.
    np.testing.assert_array_equal(found.graph[0, 58:67], [2, 2, 0.5, 0.5, 0.5, 2, 0.5, 0, 0])
    np.testing.assert_array_equal(found.graph[0, 67:76], [3, 2, 1, 1, 1, 4, 1, 0, 0])
    np.testing.assert_array_equal(found.graph[0, 76:94], np.zeros(18))
    # Example 1: device 3 and IP 0 alone, each its own mean and maximum.
    np.testing.assert_array_equal(found.graph[1, 58:67], [2, 2, 2, 2, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(found.graph[1, 67:76], [2, 2, 2, 2, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(found.graph[1, 76:85], [5, 6, 1, 1, 2, 3, 1, 0, 0])
    np.testing.assert_array_equal(found.graph[1, 85:94], [5, 6, 1, 1, 2, 3, 1, 0, 0])
    np.testing.assert_array_equal(
        found.graph[:, 94:], _hand_batch().data.community.numpy().astype(np.float32)
    )
    assert found.own.shape == (2, 50) and found.graph.shape == (2, 102)


def test_every_enrolment_edge_of_the_customer_is_read_however_many() -> None:
    """E_s is every device the customer has an enrolment edge to in the subgraph, exactly what
    the GNN sees (docs/features.md, XGBoost inputs), not 8 kept links: customer 0 is enrolled on
    its devices 0 and 1 and on 7 more, 9 edges, and all 9 rows are aggregated. The 7 new devices
    read 10, 20, ..., 70 customers in the last 30 days and nothing else, so the column of
    customers_30d has the mean (1 + 3 + 280) / 9 and the maximum 70, and customers_90d, 2 for
    devices 0 and 1 and 0 for the others, the mean 4 / 9 and the maximum 2."""
    batch = _hand_batch()
    data = batch.data
    more = torch.zeros(7, 12, dtype=torch.float64)
    more[:, 0] = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    data[DEVICE].x = torch.cat([data[DEVICE].x, more])
    extra = torch.tensor([[0] * 7, list(range(4, 11))])
    data[ENROLLED_ON_DEVICE].edge_index = torch.cat([data[ENROLLED_ON_DEVICE].edge_index, extra], 1)
    found = vectors(batch)
    mean_30d = GRAPH_COLUMNS.index("device_mean.customers_30d")
    max_30d = GRAPH_COLUMNS.index("device_max.customers_30d")
    mean_90d = GRAPH_COLUMNS.index("device_mean.customers_90d")
    max_90d = GRAPH_COLUMNS.index("device_max.customers_90d")
    assert found.graph[0, mean_30d] == pytest.approx(284 / 9) and found.graph[0, max_30d] == 70
    assert found.graph[0, mean_90d] == pytest.approx(4 / 9) and found.graph[0, max_90d] == 2
    # Example 1's customer is enrolled on one device, whatever customer 0 holds.
    assert found.graph[1, mean_30d] == found.graph[1, max_30d] == 2


def test_a_scored_account_without_its_customer_is_refused() -> None:
    batch = _hand_batch()
    batch.data[OWNED_BY].edge_index = torch.tensor([[1], [1]])
    with pytest.raises(ValueError, match="no ownership edge"):
        vectors(batch)


def test_the_vectors_of_the_batch_world_read_its_rows(world: Any) -> None:
    """Every example's own vector is its scored account's row, and the graph vector's customer
    and community parts its customer's row and its block."""
    found = vectors_of(world.training)
    assert len(found.keys) == 448 and found.own.shape == (448, 50)
    assert found.graph.shape == (448, 102)
    assert np.isfinite(found.graph).all()
    row = 0
    for batch in world.training:
        data = batch.data
        scored = data.scored_index.numpy()
        accounts, customers = data[OWNED_BY].edge_index.numpy()
        for index, position in enumerate(scored):
            raw = data[ACCOUNT].x[position, :50].numpy().copy()
            raw[[18, 45, 47, 49]] /= 1000
            np.testing.assert_array_equal(found.own[row], raw.astype(np.float32))
            customer = customers[accounts == position][0]
            np.testing.assert_array_equal(
                found.graph[row, 50:58], data[CUSTOMER].x[customer, :8].numpy().astype(np.float32)
            )
            np.testing.assert_array_equal(
                found.graph[row, 94:], data.community[index].numpy().astype(np.float32)
            )
            row += 1
    # A ring member's customer is enrolled on the ring's device, which reaches its vector.
    ring = found.keys.index(next(key for key in found.keys if key[1] == "a_r1"))
    assert found.graph[ring, GRAPH_COLUMNS.index("device_max.customers_90d")] >= 2
    with pytest.raises(ValueError, match="at least one batch"):
        vectors_of([])


def test_a_training_batchs_labels_and_weights_travel_with_its_vectors() -> None:
    """The hand-built batch's examples are labelled 1 and 0 and weigh 2 and 0.5: row 0 of every
    array is the first example whatever order the batch holds them in."""
    batch = _hand_batch()
    batch.data.label = torch.tensor([1, 0])
    batch.data.weight = torch.tensor([2.0, 0.5], dtype=torch.float64)
    found = vectors(batch)
    assert found.labelled and found.labels.tolist() == [1, 0] and found.weights.tolist() == [2, 0.5]
    assert found.labels.dtype == np.int64 and found.weights.dtype == np.float64
    # A scoring batch carries none.
    scoring = vectors(_hand_batch())
    assert not scoring.labelled
    assert scoring.labels.shape == scoring.weights.shape == (0,)


def test_the_labels_and_weights_of_the_shuffled_batches_follow_their_keys(world: Any) -> None:
    """The batch world's training batches hold their examples shuffled across both weeks, so
    their order is neither the examples' nor sorted, yet every row's label is whether its own
    account is a mule and its weight is its own example's."""
    found = vectors_of(world.training)
    assert list(found.keys) != sorted(found.keys) and len(found.keys) == 448
    assert found.labelled and len(found.labels) == len(found.weights) == 448
    for row, (t_ms, account) in enumerate(found.keys):
        assert found.labels[row] == int(account in world.mules)
        assert found.weights[row] == world.weights[t_ms, account]
    assert found.labels.sum() == 2 * len(world.mules)


def test_vectors_refuse_mixed_batches_and_mismatched_rows(world: Any) -> None:
    scoring = next(iter(world.selection.batches))
    with pytest.raises(ValueError, match="some training batches and some scoring batches"):
        vectors_of([world.training[0], scoring])
    only = vectors(world.training[0])
    with pytest.raises(ValueError, match="keys for"):
        Vectors(only.keys[:-1], only.own, only.graph, only.labels, only.weights)
    with pytest.raises(ValueError, match="labels and"):
        Vectors(only.keys, only.own, only.graph, only.labels[:-1], only.weights[:-1])
    with pytest.raises(ValueError, match="labels and"):
        Vectors(only.keys, only.own, only.graph, only.labels, only.weights[:-1])
