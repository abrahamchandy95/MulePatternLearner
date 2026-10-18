"""The degree columns: every node carries, for each relation arriving at its type, the number
of that relation's edges ending at it in its example's subgraph (docs/features.md, "Degree
columns"). Attention's softmax averages over neighbours and cannot count them.

They are the one count Python makes, which the owner agreed to on 2026-10-17: they describe
the built batch's own structure, so they are counted from its edge lists once the batch is
built and again after any ablation switch. Raw they are whole numbers; the encoding makes them
log(1 + n). Each example is its own copy of its subgraph, so a node's degrees count the edges
of its own example only.
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from .layouts import DEGREES, NODE_LAYOUTS, degree_name


def count(data: HeteroData) -> None:
    """Write each node's degree columns, in place, from data's edges."""
    for node_type, relations in DEGREES.items():
        x = data[node_type].x
        layout = NODE_LAYOUTS[node_type]
        for relation in relations:
            ends = data[relation].edge_index[1]
            column = layout.index(degree_name(relation))
            x[:, column] = torch.bincount(ends, minlength=x.shape[0]).to(x.dtype)
