from typing import cast, override

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module, ModuleDict
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, MessagePassing
from torch_geometric.typing import EdgeType, NodeType

# The six heterogeneous edge types the sampler produces. Same-type edges
# (Account->Account) admit self-loops; the four cross-type (bipartite) relations
# do not, since a self-loop across distinct node types is undefined.
_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")
_ACCOUNT_ACCOUNT: EdgeType = ("Account", "Account_Account", "Account")
_ACCOUNT_PARTY: EdgeType = ("Account", "Account_Party", "Party")
_PARTY_ENTITY: EdgeType = ("Party", "Party_Entity", "Resolved_Entity")
_ENTITY_PARTY: EdgeType = ("Resolved_Entity", "Entity_Party", "Party")
_PARTY_ACCOUNT: EdgeType = ("Party", "Party_Account", "Account")

_EDGE_TYPES: tuple[EdgeType, ...] = (
    _HAS_PAID,
    _ACCOUNT_ACCOUNT,
    _ACCOUNT_PARTY,
    _PARTY_ENTITY,
    _ENTITY_PARTY,
    _PARTY_ACCOUNT,
)
_SAME_TYPE_EDGES: frozenset[EdgeType] = frozenset({_HAS_PAID, _ACCOUNT_ACCOUNT})

_ACCOUNT: NodeType = "Account"


class MulePatternModel(Module):
    """
    Heterogeneous GATv2 model that scores accounts by mule-likelihood.

    The attention mathematics below lives INSIDE each GATv2Conv (PyG implements
    it); this class declares the convs with the right dimensions and wires them
    together. For one attention head, a destination node i is updated from its
    neighbors j as:

        score:      e_ij = a^T . LeakyReLU( W . h_i + W . h_j + W_e . edge_ij )
        normalize:  alpha_ij = softmax_j(e_ij) = exp(e_ij) / sum_k exp(e_ik)
        aggregate:  h_i' = sigma( sum_j alpha_ij * (W . h_j) )

    "v2" puts the learnable vector a AFTER the LeakyReLU (dynamic attention), so
    node i can rank its neighbors differently than another node would. With H
    heads the per-head outputs are concatenated:

        h_i' = concat_{m=1..H} sigma( sum_j alpha_ij^(m) * W^(m) . h_j )

    HeteroConv runs one such GATv2 PER edge type r and sums over relations at
    each destination node:

        h_i' = sum_{r in relations(type(i))} GATv2_r( h_i, {h_j : j --r--> i} )
    """

    _account_in: Linear
    _featureless_embeds: ModuleDict
    _conv1: HeteroConv
    _conv2: HeteroConv
    _dropout: Dropout
    _head: Linear
    _featureless_types: tuple[NodeType, ...]

    def __init__(
        self,
        account_in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        featureless_types: tuple[NodeType, ...] = ("Party", "Resolved_Entity"),
    ) -> None:
        super().__init__()
        self._featureless_types = featureless_types

        # Account input projection:  x_i = W_in . features_i + b
        #   features_i : [account_in_dim]            (= [31])
        #   W_in       : [hidden_dim, account_in_dim] (= [64, 31]; PyTorch stores
        #                Linear weight as [out, in] because y = x @ W^T + b)
        #   x_i        : [hidden_dim]                (= [64])
        self._account_in = Linear(account_in_dim, hidden_dim)

        # Featureless types (Party, Resolved_Entity) have no input features, so
        # each type gets ONE learned vector e_type in R^{hidden_dim}, broadcast
        # to all its nodes:  x_i = e_type  for every node i of that type.
        #   each Embedding(1, hidden_dim) weight : [1, 64]
        embeds: dict[str, Module] = {ntype: Embedding(1, hidden_dim) for ntype in featureless_types}
        self._featureless_embeds = ModuleDict(embeds)

        # With concat=True the H heads are concatenated, so each head must output
        # hidden_dim / H to reassemble to hidden_dim:
        #   per_head = hidden_dim / heads        (64 / 4 = 16)
        #   concat of H heads : H * per_head = hidden_dim  (4 * 16 = 64)
        # heads is a hyperparameter (must divide hidden_dim). More heads = more
        # independent attention patterns but thinner per-head dim; per_head ~ 8-16
        # is typical, so 4 or 8 heads suit hidden_dim = 64.
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        per_head = hidden_dim // heads

        # Two layers, because the sampler fetches 2 hops. Layer L lets each node
        # see L hops away: after conv1 a node summarizes its 1-hop neighborhood;
        # conv2 aggregates those summaries -> 2-hop reach.
        self._conv1 = self._make_conv(hidden_dim, per_head, heads, edge_dim, dropout)
        self._conv2 = self._make_conv(hidden_dim, per_head, heads, edge_dim, dropout)

        self._dropout = Dropout(dropout)

        # Readout head on Account vectors:  logit_i = w_out . h_i + b
        #   W_out : [1, hidden_dim] (= [1, 64]);  logit_i : scalar
        self._head = Linear(hidden_dim, 1)

    def _make_conv(
        self,
        in_dim: int,
        per_head: int,
        heads: int,
        edge_dim: int,
        dropout: float,
    ) -> HeteroConv:
        # One GATv2Conv per edge type. Each conv independently holds the W and a
        # of the attention equations (its own weights per relation), and each
        # runs `heads` parallel heads. Internally, per head:
        #     e_ij = a^T . LeakyReLU(W.h_i + W.h_j [+ W_e.edge_ij])
        #     alpha_ij = softmax_j(e_ij)
        #     head_out_i = sum_j alpha_ij * (W . h_j)        -> [per_head]
        # then concat over heads -> [heads * per_head] = [in_dim].
        convs: dict[EdgeType, MessagePassing] = {}
        for etype in _EDGE_TYPES:
            same = etype in _SAME_TYPE_EDGES
            convs[etype] = GATv2Conv(
                # (src_dim, dst_dim) bipartite in-channels; both = hidden_dim
                (in_dim, in_dim),
                # per-head output dim; concat of `heads` of these = hidden_dim
                per_head,
                heads=heads,  # H parallel attention heads
                concat=True,  # concat heads (-> hidden_dim), not average
                dropout=dropout,  # dropout on attention coefficients alpha_ij
                add_self_loops=same,  # include the W.h_i self term only for same-type edges
                # edge term W_e.edge_ij only on HAS_PAID, whose edge_attr is [E, edge_dim]=[E,34]
                edge_dim=edge_dim if etype == _HAS_PAID else None,
            )
        # aggr="sum": combine the per-relation results at each node by summation,
        #     h_i' = sum_r GATv2_r(...)
        return HeteroConv(convs, aggr="sum")

    def _initial_x(
        self, x_dict: dict[NodeType, Tensor], counts: dict[NodeType, int]
    ) -> dict[NodeType, Tensor]:
        out: dict[NodeType, Tensor] = {}

        # Account:  X_account @ W_in^T + b
        #   x_dict["Account"] : [N_account, 31]
        #   out["Account"]    : [N_account, 64]
        out[_ACCOUNT] = cast(Tensor, self._account_in(x_dict[_ACCOUNT]))

        # Featureless types: look up the single learned vector for every node.
        #   idx          : [n]   (all zeros -> the one embedding row)
        #   out[ntype]   : [n, 64]   (e_type broadcast to all n nodes)
        # idx must sit on the model's device (the Account projection already is),
        # or indexing the on-device embedding with a CPU index fails on MPS/CUDA.
        device = out[_ACCOUNT].device
        for ntype in self._featureless_types:
            n = counts.get(ntype, 0)
            embed = self._featureless_embeds[ntype]
            idx = torch.zeros(n, dtype=torch.long, device=device)
            out[ntype] = cast(Tensor, embed(idx))
        return out

    @override
    def forward(
        self,
        x_dict: dict[NodeType, Tensor],  # {"Account": [N,31], ...}
        edge_index_dict: dict[EdgeType, Tensor],  # each etype -> [2, E_etype] (COO)
        edge_attr_dict: dict[EdgeType, Tensor],  # {_HAS_PAID: [E_hp, 34]}
        node_counts: dict[NodeType, int],  # {"Account": N, "Party": ..., ...}
    ) -> Tensor:
        # Project / embed every node type to hidden_dim.
        #   h_dict[type] : [N_type, 64]
        h_dict: dict[NodeType, Tensor] = self._initial_x(x_dict, node_counts)

        # Layer 1: per-relation GATv2 attention, summed per node (1-hop mixing).
        #   for each type, h_i^(1) = sum_r sigma( sum_j alpha_ij^r * W^r . h_j )
        #   h1[type] : [N_type, 64]
        h1: dict[NodeType, Tensor] = cast(
            dict[NodeType, Tensor],
            self._conv1(h_dict, edge_index_dict, edge_attr_dict),
        )
        # ELU nonlinearity:  ELU(x) = x if x > 0 else (exp(x) - 1)
        h1 = {k: F.elu(v) for k, v in h1.items()}
        # Dropout (training only): randomly zero a fraction of activations.
        h1 = {k: cast(Tensor, self._dropout(v)) for k, v in h1.items()}

        # Layer 2: aggregates the 1-hop summaries -> each node reflects 2 hops.
        #   h2[type] : [N_type, 64]
        h2: dict[NodeType, Tensor] = cast(
            dict[NodeType, Tensor],
            self._conv2(h1, edge_index_dict, edge_attr_dict),
        )
        h2 = {k: F.elu(v) for k, v in h2.items()}

        # Readout on accounts only:  logit_i = w_out . h_i + b
        #   account_h : [N_account, 64]
        #   logits    : [N_account, 1] -> reshape -> [N_account]
        account_h = h2[_ACCOUNT]
        logits: Tensor = cast(Tensor, self._head(account_h))
        return logits.reshape(-1)
