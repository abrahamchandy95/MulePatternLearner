from typing import cast, override

import torch
from torch_geometric.data import EdgeAttr, EdgeLayout, GraphStore
from torch_geometric.typing import EdgeTensorType, EdgeType

from mule_pattern_learner.indexing.node_id_mapper import NodeIDMapper
from mule_pattern_learner.indexing.reindex import edge_type_schema
from mule_pattern_learner.tigergraph.client import Client


class GraphStoreError(RuntimeError):
    pass


# PyG (src, relation, dst) triple -> raw GSQL relation name (reverse of the
# reindex schema). This is the single place the PyG<->GSQL edge naming lives.
_RELATION_BY_TYPE: dict[EdgeType, str] = {
    triple: name for name, triple in edge_type_schema().items()
}


class TigerGraphGraphStore(GraphStore):
    """
    PyG GraphStore over TigerGraph, sharing a backend's mapper.

    Used on the training path via get_all_edge_attrs: PyG's NodeLoader calls it
    at construction to learn the heterogeneous edge types. (Batch connectivity
    itself comes from the k-hop sampler, which queries the server directly, not
    from this store.)

    _get_edge_index exports one edge type's full COO connectivity, mapping global
    string ids to shared integer ids. PyG's GraphStore interface requires it, but
    nothing on the training path calls it — only scripts/demos/graph_store.py
    exercises it, and it's available for whole-type export. It queries one edge
    type at a time, never the whole graph.

    Edges are static, so put/remove are unsupported.
    """

    _client: Client
    _mapper: NodeIDMapper
    _export_query: str

    def __init__(
        self,
        client: Client,
        mapper: NodeIDMapper,
        export_query: str = "export_edges_by_type",
    ) -> None:
        super().__init__()
        self._client = client
        self._mapper = mapper
        self._export_query = export_query

    @override
    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        return [
            EdgeAttr(edge_type=triple, layout=EdgeLayout.COO, is_sorted=False)
            for triple in edge_type_schema().values()
        ]

    @override
    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType | None:
        edge_type = edge_attr.edge_type
        relation = _RELATION_BY_TYPE.get(edge_type)
        if relation is None:
            raise GraphStoreError(f"unknown edge type {edge_type!r}")

        src_type = edge_type[0]
        dst_type = edge_type[2]
        src_ids, dst_ids = self._export_all(relation, src_type, dst_type)

        row = self._mapper.register(src_type, src_ids)
        col = self._mapper.register(dst_type, dst_ids)
        row_t = torch.tensor(row, dtype=torch.long)
        col_t = torch.tensor(col, dtype=torch.long)
        return (row_t, col_t)

    def _export_all(
        self, relation: str, src_type: str, dst_type: str
    ) -> tuple[list[str], list[str]]:
        # Page through the keyset cursor until the source set is exhausted, so
        # an arbitrarily large relation is pulled in bounded pages rather than
        # one unbounded response. src_type/dst_type are used by the caller to map
        # ids; the query itself derives endpoints from the relation name.
        _ = src_type
        _ = dst_type
        src_ids: list[str] = []
        dst_ids: list[str] = []
        cursor = ""
        while True:
            params: dict[str, object] = {
                "relation": relation,
                "cursor": cursor,
            }
            raw = self._client.conn.runInstalledQuery(self._export_query, params)
            page_src, page_dst, next_cursor = self._parse_page(raw)
            src_ids.extend(page_src)
            dst_ids.extend(page_dst)
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return src_ids, dst_ids

    def _parse_page(self, raw: list[object]) -> tuple[list[str], list[str], str]:
        src_ids: list[str] = []
        dst_ids: list[str] = []
        next_cursor = ""
        for block in raw:
            if not isinstance(block, dict):
                continue
            b = _string_keyed(block)
            if "edges" in b:
                edges = b["edges"]
                if not isinstance(edges, list):
                    raise GraphStoreError(f"'edges' is not a list: {edges!r}")
                for edge in cast("list[object]", edges):
                    if not isinstance(edge, dict):
                        raise GraphStoreError(f"edge is not a dict: {edge!r}")
                    e = _string_keyed(edge)
                    frm = e.get("from_id")
                    to = e.get("to_id")
                    if not isinstance(frm, str) or not isinstance(to, str):
                        raise GraphStoreError(f"edge missing endpoints: {e!r}")
                    src_ids.append(frm)
                    dst_ids.append(to)
            if "next_cursor" in b:
                nc = b["next_cursor"]
                if isinstance(nc, str):
                    next_cursor = nc
        return src_ids, dst_ids, next_cursor

    @override
    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> bool:
        _ = edge_index
        _ = edge_attr
        return False

    @override
    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        _ = edge_attr
        return False


def _string_keyed(value: dict[object, object]) -> dict[str, object]:
    return {str(k): v for k, v in value.items()}
