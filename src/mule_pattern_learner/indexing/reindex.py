from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

NodeType = str
EdgeType = tuple[str, str, str]

_ACCOUNT: NodeType = "Account"


class ReindexError(ValueError):
    pass


_EDGE_TYPE_SCHEMA: dict[str, EdgeType] = {
    "HAS_PAID": ("Account", "HAS_PAID", "Account"),
    "Account_Account": ("Account", "Account_Account", "Account"),
    "Account_Party": ("Account", "Account_Party", "Party"),
    "Party_Entity": ("Party", "Party_Entity", "Resolved_Entity"),
    "Entity_Party": ("Resolved_Entity", "Entity_Party", "Party"),
    "Party_Account": ("Party", "Party_Account", "Account"),
}


def edge_type_schema() -> dict[str, EdgeType]:
    """Mapping from raw GSQL e_type string to PyG (src, relation, dst) triple."""
    return dict(_EDGE_TYPE_SCHEMA)


@dataclass(frozen=True, slots=True)
class RawNeighborhood:
    """Parsed GSQL sampler output: per-type global id lists, plus edges as
    (from_global_id, to_global_id, e_type) triples referencing those ids.
    """

    node_ids: dict[NodeType, list[str]]
    edges: list[tuple[str, str, str]]


@dataclass(frozen=True, slots=True)
class LocalGraph:
    """Local-indexed neighborhood, ready to become a PyG HeteroSamplerOutput.

    node: ordered global ids per type (position == local id).
    row/col: local indices into the per-type lists, per edge type.
    """

    node: dict[NodeType, list[str]]
    row: dict[EdgeType, list[int]]
    col: dict[EdgeType, list[int]]

    def num_nodes(self, node_type: NodeType) -> int:
        return len(self.node.get(node_type, []))

    def num_edges(self, edge_type: EdgeType) -> int:
        return len(self.row.get(edge_type, []))


def _build_index(node_ids: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, nid in enumerate(node_ids):
        if nid in index:
            raise ReindexError(f"duplicate node id in input: {nid!r}")
        index[nid] = i
    return index


def _order_seeds_first(account_ids: Sequence[str], seed_ids: Sequence[str]) -> list[str]:
    """
    Reorder account_ids with seeds first, so logits[:batch_size] hits the seeds,
    not neighbors.
    """
    present = set(account_ids)
    seen: set[str] = set()
    ordered_seeds: list[str] = []
    for s in seed_ids:
        if s in present and s not in seen:
            seen.add(s)
            ordered_seeds.append(s)
    rest = [a for a in account_ids if a not in seen]
    return ordered_seeds + rest


def reindex_neighborhood(raw: RawNeighborhood, seed_ids: Sequence[str] | None = None) -> LocalGraph:
    """
    Convert global-id neighborhood to local indices, seeds ordered first.

    Each edge's endpoints are looked up in the per-type node lists to produce
    local row/col indices; an endpoint absent from its list is rejected. With
    seed_ids, Accounts are reordered (seeds first) before indices are built, so
    the first len(seed_ids) Account rows are the seeds.
    """
    node: dict[NodeType, list[str]] = {ntype: list(ids) for ntype, ids in raw.node_ids.items()}
    if seed_ids is not None and _ACCOUNT in node:
        node[_ACCOUNT] = _order_seeds_first(node[_ACCOUNT], seed_ids)

    indices: dict[NodeType, dict[str, int]] = {
        ntype: _build_index(ids) for ntype, ids in node.items()
    }

    row: dict[EdgeType, list[int]] = {}
    col: dict[EdgeType, list[int]] = {}

    for from_id, to_id, e_type in raw.edges:
        schema = _EDGE_TYPE_SCHEMA.get(e_type)
        if schema is None:
            raise ReindexError(f"unknown edge type: {e_type!r}")
        src_type, _, dst_type = schema

        src_index = indices.get(src_type)
        dst_index = indices.get(dst_type)
        if src_index is None:
            raise ReindexError(
                f"edge type {e_type!r} needs source node type {src_type!r}, "
                + "which is absent from the node sets"
            )
        if dst_index is None:
            raise ReindexError(
                f"edge type {e_type!r} needs destination node type {dst_type!r}, "
                + "which is absent from the node sets"
            )

        local_src = src_index.get(from_id)
        local_dst = dst_index.get(to_id)
        if local_src is None:
            raise ReindexError(f"edge {e_type!r}: source id {from_id!r} not in {src_type!r} nodes")
        if local_dst is None:
            raise ReindexError(
                f"edge {e_type!r}: destination id {to_id!r} not in {dst_type!r} nodes"
            )

        row.setdefault(schema, []).append(local_src)
        col.setdefault(schema, []).append(local_dst)

    return LocalGraph(node=node, row=row, col=col)


def parse_raw_result(result: Sequence[object]) -> RawNeighborhood:
    """
    Parse the GSQL sampler result into a RawNeighborhood.

    Reads the per-type id blocks (account_ids/party_ids/entity_ids) and the
    edges block of {from_id, to_id, e_type} objects.
    """
    name_to_type: dict[str, NodeType] = {
        "account_ids": "Account",
        "party_ids": "Party",
        "entity_ids": "Resolved_Entity",
    }
    node_ids: dict[NodeType, list[str]] = {}
    edges: list[tuple[str, str, str]] = []

    for block in result:
        if not isinstance(block, dict):
            continue
        b = cast(dict[str, object], block)
        for key, ntype in name_to_type.items():
            if key in b:
                raw_ids = b[key]
                ids: list[str] = []
                if isinstance(raw_ids, list):
                    for item in cast(list[object], raw_ids):
                        ids.append(str(item))
                node_ids[ntype] = ids
        if "edges" in b:
            raw_edges = b["edges"]
            if isinstance(raw_edges, list):
                for e in cast(list[object], raw_edges):
                    if not isinstance(e, dict):
                        continue
                    ed = cast(dict[str, object], e)
                    from_id = ed.get("from_id")
                    to_id = ed.get("to_id")
                    e_type = ed.get("e_type")
                    if from_id is not None and to_id is not None and e_type is not None:
                        edges.append((str(from_id), str(to_id), str(e_type)))

    return RawNeighborhood(node_ids=node_ids, edges=edges)
