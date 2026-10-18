"""The degree columns: per node, the edges of each relation arriving at its type in its own
example, counted from the batch's edge lists and counted again after the edges change
(docs/features.md, "Degree columns")."""

from __future__ import annotations

import torch

from mule_pattern_learner.subgraph import degrees
from mule_pattern_learner.subgraph.build import Batch, put_relation
from mule_pattern_learner.subgraph.layouts import (
    ACCOUNT_LAYOUT,
    CUSTOMER_LAYOUT,
    DEGREES,
    NODE_LAYOUTS,
    OWNED_BY,
    PAYS,
    degree_name,
    reverse,
)

# The hand-built batch's account nodes (conftest.py): a_co, a_hub, a_own, a_s1, a_two and
# x_payer at T, then a_s2, a_three and x_payer at T + 1 week.
A_HUB, A_OWN, A_S1, X_PAYER_0, X_PAYER_1 = 1, 2, 3, 5, 8


def _degree(batch: Batch, kind: str, relation: tuple[str, str, str]) -> list[float]:
    column = NODE_LAYOUTS[kind].index(degree_name(relation))
    return batch.data[kind].x[:, column].tolist()


def test_each_node_counts_the_edges_arriving_in_its_own_example(hand_batch: Batch) -> None:
    received = _degree(hand_batch, "account", PAYS)
    sent = _degree(hand_batch, "account", reverse(PAYS))
    # x_payer received p3 and sent p1 at T; its copy at T + 1 week only sent p5.
    assert (received[X_PAYER_0], sent[X_PAYER_0]) == (1, 1)
    assert (received[X_PAYER_1], sent[X_PAYER_1]) == (0, 1)
    assert sent[A_S1] == 2
    # p_c1 owns a_own and a_s1; p_co owns none of the subgraph's accounts; p_c2 owns a_s2.
    assert _degree(hand_batch, "customer", OWNED_BY) == [2, 0, 1]
    assert _degree(hand_batch, "account", reverse(OWNED_BY)) == [0, 0, 1, 1, 0, 0, 1, 0, 0]


def test_the_degree_columns_are_counted_again_after_the_edges_change(hand_batch: Batch) -> None:
    data = hand_batch.data.clone()
    # Drop p2, a_s1's payment to a_hub, and a_own's ownership edge.
    pays = data[PAYS].edge_index
    kept = ~((pays[0] == A_S1) & (pays[1] == A_HUB))
    put_relation(data, PAYS, pays[:, kept], data[PAYS].edge_attr[kept])
    owned = data[OWNED_BY].edge_index
    put_relation(data, OWNED_BY, owned[:, owned[0] != A_OWN], None)
    before = data["account"].x.clone()
    degrees.count(data)
    after = Batch(hand_batch.keys, data)
    assert _degree(after, "account", PAYS)[A_HUB] == 0
    assert _degree(after, "account", reverse(PAYS))[A_S1] == 1
    assert _degree(after, "account", reverse(OWNED_BY))[A_OWN] == 0
    assert _degree(after, "customer", OWNED_BY)[0] == 1
    # Nothing but the degree columns changed.
    first = ACCOUNT_LAYOUT.index("degree_pays")
    assert torch.equal(data["account"].x[:, :first], before[:, :first])
    customers = CUSTOMER_LAYOUT.index("degree_owned_by")
    assert torch.equal(
        data["customer"].x[:, :customers], hand_batch.data["customer"].x[:, :customers]
    )


def test_counting_twice_counts_the_same(hand_batch: Batch) -> None:
    data = hand_batch.data.clone()
    degrees.count(data)
    for kind in DEGREES:
        assert torch.equal(data[kind].x, hand_batch.data[kind].x)
