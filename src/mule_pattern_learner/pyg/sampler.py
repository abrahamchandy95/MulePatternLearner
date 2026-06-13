from typing import override

import torch
from torch import Tensor
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NodeSamplerInput,
)
from torch_geometric.typing import EdgeType, NodeType

from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.indexing.node_id_mapper import NodeIDMapper
from mule_pattern_learner.indexing.reindex import (
    LocalGraph,
    parse_raw_result,
    reindex_neighborhood,
)
from mule_pattern_learner.tigergraph.client import Client


class TigerGraphSamplerError(RuntimeError):
    pass


class TigerGraphHeteroSampler(BaseSampler):
    """
    A PyG BaseSampler that delegates k-hop neighborhood sampling to TigerGraph.

    Samples each batch's neighborhood with a server-side query and returns it as
    PyG structure, registering the batch's nodes in the shared NodeIDMapper so the
    FeatureStore can fetch their features. seed_ids defines the index<->id mapping,
    bounded by the seed set, not the full graph.
    allow_val/allow_test gate strict-
    inductive filtering: when False, neighborhoods don't traverse into val/test
    accounts (no leakage).
    Train passes both False; val (True, False); test both True.
    """

    _client: Client
    _fanout: NeighborFanout
    _seed_ids: tuple[str, ...]
    _mapper: NodeIDMapper
    _query_name: str
    _allow_val: bool
    _allow_test: bool

    def __init__(
        self,
        client: Client,
        seed_ids: tuple[str, ...],
        mapper: NodeIDMapper,
        fanout: NeighborFanout | None = None,
        query_name: str = "sample_khop_neighborhood",
        allow_val: bool = True,
        allow_test: bool = True,
    ) -> None:
        super().__init__()
        self._client = client
        self._seed_ids = seed_ids
        self._mapper = mapper
        self._fanout = fanout if fanout is not None else NeighborFanout()
        self._query_name = query_name
        self._allow_val = allow_val
        self._allow_test = allow_test

    def _seed_indices_to_ids(self, node: Tensor) -> list[str]:
        indices: list[int] = node.tolist()
        n = len(self._seed_ids)
        ids: list[str] = []
        for idx in indices:
            if idx < 0 or idx >= n:
                raise TigerGraphSamplerError(f"seed index {idx} out of range for {n} seed ids")
            ids.append(self._seed_ids[idx])
        return ids

    def _run_query(self, seed_ids: list[str]) -> list[object]:
        params: dict[str, object] = {"seeds": [(sid,) for sid in seed_ids]}
        for key, value in self._fanout.as_query_params().items():
            params[key] = value
        # strict-inductive split flags: gate traversal into held-out accounts
        params["allow_val"] = self._allow_val
        params["allow_test"] = self._allow_test
        return self._client.conn.runInstalledQuery(self._query_name, params)

    def _to_hetero_output(self, local: LocalGraph, index: NodeSamplerInput) -> HeteroSamplerOutput:
        """
        Pack a reindexed LocalGraph into PyG's HeteroSamplerOutput.

        Registers each node's global id in the shared mapper and writes the
        assigned integers into the node tensors, so the FeatureStore can reverse
        them to fetch features. edge is all-None (edge features are attached
        later by the transform, not at sample time).
        """
        node: dict[NodeType, Tensor] = {}
        for ntype, ids in local.node.items():
            int_ids = self._mapper.register(ntype, list(ids))
            node[ntype] = torch.tensor(int_ids, dtype=torch.long)
        row: dict[EdgeType, Tensor] = {
            etype: torch.tensor(rows, dtype=torch.long) for etype, rows in local.row.items()
        }
        col: dict[EdgeType, Tensor] = {
            etype: torch.tensor(cols, dtype=torch.long) for etype, cols in local.col.items()
        }
        edge: dict[EdgeType, Tensor | None] = {etype: None for etype in local.row}
        batch_size = int(index.node.shape[0])
        metadata: tuple[Tensor | None, int] = (index.input_id, batch_size)
        return HeteroSamplerOutput(node=node, row=row, col=col, edge=edge, metadata=metadata)

    @override
    def sample_from_nodes(self, index: NodeSamplerInput, **kwargs: object) -> HeteroSamplerOutput:
        _ = kwargs
        # Reset the shared mapper so it holds only this batch's nodes, not every
        # node sampled across the run. Loader is synchronous (num_workers=0), so
        # the prior batch's feature fetch is already done and no live id is lost.
        self._mapper.reset()
        seed_ids = self._seed_indices_to_ids(index.node)
        raw = self._run_query(seed_ids)
        # order seeds first so the first len(seeds) Account rows ARE the seeds
        # (the contract the loader relies on to slice seed logits)
        local = reindex_neighborhood(parse_raw_result(raw), seed_ids=seed_ids)
        return self._to_hetero_output(local, index)

    @override
    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling: NegativeSampling | None = None,
        **kwargs: object,
    ) -> HeteroSamplerOutput:
        _ = index
        _ = neg_sampling
        _ = kwargs
        raise NotImplementedError(
            "TigerGraphHeteroSampler supports node-level sampling only; use NodeLoader."
        )
