"""The heterogeneous temporal attention model (docs/design.md, "The model").

It reads the HeteroData of subgraph.build: one component per seed, with each node type's
features x, each relation and its reverse with edge_index and edge_attr, and seed_index.
Two HeteroConv layers each run one TransformerConv per relation, with the relation's edge
features (their Fourier encodings among them) as edge attributes, and sum what the
relations bring to each node. Each layer is followed by a ReLU and dropout, and the seed's
final representation goes through a head of HEAD_WIDTH units to one score. There is no
per-account embedding and no memory, so any account, a new one too, can be scored.

The label-input variant also reads, for each account node, whether it was a confirmed mule
by its seed's as-of time and the Fourier encoding of the time since, appended to the
account's features. The confirmations are kept apart from the subgraph: forward takes
them as since_confirmed_ms, the time from each account's confirmation to its seed's as-of
time, or -1 when it was not confirmed by then. Every relation has its own attention, so
the model weighs a confirmation by how the account is linked: through payments, a shared
owner, a device, an IP or an address. The seed and every account of its owners always read
as unknown. While training, each other confirmed account is hidden with probability
label_mask, drawn on the tensors' device each step, so the same built subgraphs serve every
epoch; nothing is hidden at evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, ReLU, Sequential, functional
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing, TransformerConv

from ..config import ModelConfig
from .time_encoding import WIDTH, encode

type EdgeType = tuple[str, str, str]

# The node type of the seeds, and the relation from an account to its owner.
ACCOUNT = "account"
OWNED_BY: EdgeType = ("account", "owned_by", "party")
# The units of the head between the seed's representation and its score.
HEAD_WIDTH = 32
# The columns the label-input variant adds to an account: the confirmed flag, then the
# Fourier encoding of the time since the confirmation, zero when it is unknown.
LABEL_WIDTH = 1 + WIDTH
# Two layers, so a seed hears from accounts two hops away.
LAYERS = 2


def protected(data: HeteroData) -> Tensor:
    """Whether each account node is a seed or an account of a seed's owner: those always
    read as unknown. Each seed's subgraph is a component of its own, so a seed's owners
    are the parties its owned_by edges reach, and their accounts those whose owned_by
    edges reach them."""
    seeds: Tensor = data.seed_index
    source, target = data[OWNED_BY].edge_index
    owners = target[torch.isin(source, seeds)]
    found = torch.zeros(data[ACCOUNT].num_nodes, dtype=torch.bool, device=seeds.device)
    found[seeds] = True
    found[source[torch.isin(target, owners)]] = True
    return found


def label_inputs(data: HeteroData, since_confirmed_ms: Tensor, hide: float) -> Tensor:
    """The LABEL_WIDTH label columns of each account node of data: an account confirmed by
    its seed's as-of time (since_confirmed_ms not negative) that is not protected reads
    as confirmed, unless it is hidden, which happens to each with probability hide."""
    if since_confirmed_ms.shape != (data[ACCOUNT].num_nodes,):
        raise ValueError("since_confirmed_ms needs one time per account node")
    confirmed = (since_confirmed_ms >= 0) & ~protected(data)
    if hide > 0:
        shown = torch.rand(confirmed.shape, device=confirmed.device) >= hide
        confirmed = confirmed & shown
    flag = confirmed[:, None].to(torch.float32)
    return torch.cat([flag, encode(since_confirmed_ms.clamp(min=0)) * flag], dim=1)


class HeteroModel(Module):
    """One score per seed of a HeteroData, from the subgraph around it.

    node_dims gives each node type's feature width and edge_dims each relation's, the
    reverse relations among them; a batch must have exactly these types. With
    label_inputs, the account nodes also read the confirmations given to forward.
    """

    def __init__(
        self,
        node_dims: Mapping[str, int],
        edge_dims: Mapping[EdgeType, int],
        config: ModelConfig,
        *,
        label_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_inputs = label_inputs
        self.node_types = frozenset(node_dims)
        self.edge_types = frozenset(edge_dims)
        widths = dict(node_dims)
        if label_inputs:
            widths[ACCOUNT] += LABEL_WIDTH
        self.layers = ModuleList()
        for layer in range(LAYERS):
            convs: dict[EdgeType, MessagePassing] = {
                edge_type: TransformerConv(
                    (widths[edge_type[0]], widths[edge_type[2]]) if layer == 0 else config.hidden,
                    config.hidden // config.heads,
                    heads=config.heads,
                    edge_dim=edge_dim,
                )
                for edge_type, edge_dim in edge_dims.items()
            }
            self.layers.append(HeteroConv(convs, aggr="sum"))
        self.head = Sequential(Linear(config.hidden, HEAD_WIDTH), ReLU(), Linear(HEAD_WIDTH, 1))

    @override
    def forward(self, data: HeteroData, since_confirmed_ms: Tensor | None = None) -> Tensor:
        """The score of each seed, in the order of data.seed_index. since_confirmed_ms is
        given exactly when the model has label inputs."""
        types = frozenset(data.node_types), frozenset(data.edge_types)
        if types != (self.node_types, self.edge_types):
            raise ValueError("The batch's node and edge types are not the model's")
        x: dict[str, Tensor] = dict(data.x_dict)
        if self.label_inputs:
            if since_confirmed_ms is None:
                raise ValueError("The label-input model needs the confirmations")
            hide = self.config.label_mask if self.training else 0.0
            labels = label_inputs(data, since_confirmed_ms, hide)
            x[ACCOUNT] = torch.cat([x[ACCOUNT], labels], dim=1)
        elif since_confirmed_ms is not None:
            raise ValueError("The model has no label inputs to read the confirmations")
        for layer in self.layers:
            x = layer(x, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
            x = {
                kind: functional.dropout(functional.relu(h), self.config.dropout, self.training)
                for kind, h in x.items()
            }
        return self.head(x[ACCOUNT][data.seed_index]).squeeze(-1)
