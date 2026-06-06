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
    """A PyG BaseSampler that delegates k-hop sampling to TigerGraph.

    Per batch, sample_from_nodes maps the seed indices PyG provides to their
    global account ids, runs the sample_khop_neighborhood installed query on
    the server, reindexes the raw global-id result to per-type local indices
    (the pure reindex core), and returns structure as a HeteroSamplerOutput.
    A fresh NodeIDMapper for the batch (the FalkorDB shared-mapper pattern)
    rides along in metadata, so a FeatureStore can recover each node's global
    id from its integer to fetch features, following PyG's structure/feature
    split.

    seed_ids defines the seed index<->id mapping: position i in seed_ids is
    PyG integer index i. It is bounded by the labeled/seed set, never the full
    graph, so it stays small at any graph scale.

    allow_val / allow_test enforce strict-inductive split filtering in the
    query: when False, a sampled neighborhood does not traverse INTO val / test
    accounts, so their features cannot leak into a seed's embedding through
    message passing. Train batches pass both False; validation passes
    allow_val=True, allow_test=False; test passes both True. The seed accounts
    themselves are never filtered.
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
        # Register this batch's global ids into the shared persistent mapper and
        # write the assigned integers into the node tensor. The mapper is shared
        # with the feature store via the backend (not via metadata), so those
        # integers are reversible to global ids when features are fetched.
        #
        # metadata carries what PyG's filter_fn expects for a NodeLoader batch:
        # (input_id, batch_size). PyG reads these to set data.input_id and
        # data.batch_size; it is not a free slot for the mapper.
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
