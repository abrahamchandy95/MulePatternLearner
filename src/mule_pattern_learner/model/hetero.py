"""The heterogeneous temporal attention model (docs/design.md, "The model").

It reads an encoded batch (model.encode.encode_batch): each node type's rows, every relation's
edges with its attributes, the position of each example's scored account, and each example's
community block.

- Each node type's rows are projected to the hidden width by a linear layer of their own.
- Two HeteroConv layers each run one TransformerConv per relation, 4 heads of 16, reading the
  relation's edge attributes (every Fourier-encoded duration among them) as edge attributes;
  ownership and its reverse have none. What the relations bring to a node is summed, and each
  layer is followed by a ReLU and dropout.
- The head reads the scored account's final 64 numbers joined with its 8-number community
  block, Linear(72, 32), ReLU, Linear(32, 1); without the community block, Linear(64, 32).

There is no per-account embedding and no memory, so an account the model never saw is scored
like any other, and no batch-statistics layer, so a score never depends on its batch: each
example is its own copy of its subgraph, which no other example's edges reach. No label is an
input.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleDict, ModuleList, ReLU, Sequential, functional
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing, TransformerConv

from ..config import ModelConfig
from .encode import BatchLayout, EdgeType


class HeteroModel(Module):
    """One score per example of an encoded batch of layout, from the subgraph around its
    scored account. Under config.community the head reads the community block too."""

    def __init__(self, layout: BatchLayout, config: ModelConfig) -> None:
        super().__init__()
        self.layout = layout
        self.config = config
        self.project = ModuleDict(
            {kind: Linear(rows.width, config.hidden) for kind, rows in layout.nodes.items()}
        )
        self.layers = ModuleList()
        for _ in range(config.layers):
            convs: dict[EdgeType, MessagePassing] = {
                relation: TransformerConv(
                    config.hidden,
                    config.hidden // config.heads,
                    heads=config.heads,
                    edge_dim=layout.edges[relation].width if relation in layout.edges else None,
                )
                for relation in layout.relations
            }
            self.layers.append(HeteroConv(convs, aggr="sum"))
        width = config.hidden + (layout.community.width if config.community else 0)
        self.head = Sequential(
            Linear(width, config.head_hidden), ReLU(), Linear(config.head_hidden, 1)
        )

    def _check(self, data: HeteroData) -> None:
        """Refuse a batch that is not an encoded batch of the model's layout."""
        layout = self.layout
        if set(data.node_types) != set(layout.nodes) or set(data.edge_types) != set(
            layout.relations
        ):
            raise ValueError("The batch's node and edge types are not the model's")
        for kind, rows in layout.nodes.items():
            if data[kind].x.shape[1] != rows.width:
                raise ValueError(
                    f"The {kind} rows are {data[kind].x.shape[1]} wide, not {rows.width}"
                )
        for relation in layout.relations:
            store = data[relation]
            width = layout.edges[relation].width if relation in layout.edges else None
            found = store.edge_attr.shape[1] if "edge_attr" in store else None
            if found != width:
                raise ValueError(f"The attributes of {relation[1]} are {found} wide, not {width}")
        if data.community.shape[1] != layout.community.width:
            raise ValueError(f"The community block is not {layout.community.width} wide")

    @override
    def forward(self, data: HeteroData) -> Tensor:
        """The score of each example, a raw logit, in the order of data.scored_index."""
        self._check(data)
        x: dict[str, Tensor] = {
            kind: self.project[kind](data[kind].x) for kind in self.layout.nodes
        }
        for layer in self.layers:
            x = layer(x, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
            x = {
                kind: functional.dropout(functional.relu(h), self.config.dropout, self.training)
                for kind, h in x.items()
            }
        scored = x[self.layout.scored][data.scored_index]
        if self.config.community:
            scored = torch.cat([scored, data.community], dim=1)
        return self.head(scored).squeeze(-1)
