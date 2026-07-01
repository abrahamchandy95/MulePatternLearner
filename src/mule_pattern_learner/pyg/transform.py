from typing import Protocol, cast

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from mule_pattern_learner.features.edge_spec import flat_edge_dim
from mule_pattern_learner.features.temporal import (
    build_edge_features,
    flat_edge_features,
)
from mule_pattern_learner.indexing.node_id_mapper import NodeIDMapper
from mule_pattern_learner.pyg.fetch import fetch_has_paid_edges
from mule_pattern_learner.tigergraph.client import Client

_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")


class _EdgeStore(Protocol):
    edge_index: Tensor
    edge_attr: Tensor


class _NodeStore(Protocol):
    n_id: Tensor
    x: Tensor


def _edge_store(data: HeteroData, key: EdgeType) -> _EdgeStore:
    return cast(_EdgeStore, cast(object, data[key]))


def _node_store(data: HeteroData, key: NodeType) -> _NodeStore:
    return cast(_NodeStore, cast(object, data[key]))


class EdgeFeatureError(RuntimeError):
    pass


class HasPaidEdgeFeatureAttacher:
    """
    Batch transform that attaches HAS_PAID temporal edge features.

    PyG's remote FeatureStore does not serve edge features (node features only),
    so per PyG's design edge features are attached with a loader transform. For
    each sampled batch this fetches the HAS_PAID edges among the batch's Account
    nodes, builds their temporal features (temporal.py), and writes them to
    edge_attr aligned row-for-row to the batch's existing HAS_PAID edge_index.

    It shares the backend's mapper to turn the batch's local Account integers
    back into global ids for the fetch and the alignment.
    """

    _client: Client
    _mapper: NodeIDMapper
    _reference_epoch_s: float
    _max_bins: int

    def __init__(
        self,
        client: Client,
        mapper: NodeIDMapper,
        reference_epoch_s: float,
        max_bins: int,
    ) -> None:
        self._client = client
        self._mapper = mapper
        self._reference_epoch_s = reference_epoch_s
        self._max_bins = max_bins

    def __call__(self, data: HeteroData) -> HeteroData:
        if _HAS_PAID not in data.edge_types:
            return data
        # A degenerate tail batch can carry a HAS_PAID edge type but no "Account"
        # node store (no Account nodes survived sampling); accessing data["Account"]
        # below would raise KeyError: 'Account'. Without Account nodes there are no
        # edges to attach features to, so return unchanged. This transform runs
        # INSIDE loader iteration (applied as each batch is produced), so the crash
        # it prevents fires before any caller-side loop guard can see the batch --
        # which is why guarding only in the eval/train loop did not stop it.
        if "Account" not in data.node_types:
            return data

        edge_store = _edge_store(data, _HAS_PAID)
        edge_index = edge_store.edge_index
        num_edges = int(edge_index.shape[1])
        dim = flat_edge_dim(self._max_bins)

        if num_edges == 0:
            edge_store.edge_attr = torch.zeros((0, dim), dtype=torch.float32)
            return data

        account_int = _node_store(data, "Account").n_id
        account_int_ids: list[int] = [int(i) for i in account_int.tolist()]
        account_ids = self._mapper.to_strings("Account", account_int_ids)

        raw_edges = fetch_has_paid_edges(self._client, account_ids)
        features = build_edge_features(raw_edges, self._reference_epoch_s, self._max_bins)
        flat = flat_edge_features(features)

        lookup: dict[tuple[str, str], int] = {}
        for i, (s, d) in enumerate(zip(features.src_ids, features.dst_ids)):
            lookup[(s, d)] = i

        row: list[int] = [int(r) for r in edge_index[0].tolist()]
        col: list[int] = [int(c) for c in edge_index[1].tolist()]
        order: list[int] = []
        for r, c in zip(row, col):
            src_id = account_ids[r]
            dst_id = account_ids[c]
            pos = lookup.get((src_id, dst_id))
            if pos is None:
                raise EdgeFeatureError(f"no fetched features for edge {src_id} -> {dst_id}")
            order.append(pos)

        edge_store.edge_attr = flat[torch.tensor(order, dtype=torch.long)]
        return data
