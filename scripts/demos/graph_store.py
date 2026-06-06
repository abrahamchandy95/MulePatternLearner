from __future__ import annotations

from torch_geometric.data import EdgeAttr

from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings


def _show_attr(attr: EdgeAttr) -> str:
    src, rel, dst = attr.edge_type
    return f"({src})-[{rel}]->({dst})  layout={attr.layout}"


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"connected: {client.graphname}")

    backend = TigerGraphRemoteBackend(client)
    graph_store = backend.make_graph_store()

    print("=" * 70)
    print("STEP 1: get_all_edge_attrs() - the metadata NodeLoader reads at init")
    print("=" * 70)
    attrs = graph_store.get_all_edge_attrs()
    print(f"  {len(attrs)} edge types reported:")
    for a in attrs:
        print(f"    {_show_attr(a)}")

    for relation, src_t, dst_t in (
        ("HAS_PAID", "Account", "Account"),
        ("Account_Party", "Account", "Party"),
    ):
        print()
        print("=" * 70)
        print(f"STEP 2: _get_edge_index for logical relation '{relation}'")
        print("=" * 70)
        edge_index = graph_store.get_edge_index(edge_type=(src_t, relation, dst_t), layout="coo")
        row, col = edge_index
        n = int(row.shape[0])
        print(f"  exported {n} edges (COO)")
        if n > 0:
            r0 = int(row.tolist()[0])
            c0 = int(col.tolist()[0])
            src_id = backend.mapper.to_string(src_t, r0)
            dst_id = backend.mapper.to_string(dst_t, c0)
            print(f"  first edge: local ({r0},{c0}) -> {src_id} -> {dst_id}")
            print(f"  row int dtype: {row.tolist()[:5]}  (mapper-assigned ids)")

    print()
    print("Graph store works: edge-type metadata for NodeLoader, plus on-demand")
    print("per-type COO export (logical relations translated to real edges).")
    print("Note: batch sampling uses the sampler, not this store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
