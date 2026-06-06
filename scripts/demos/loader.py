from __future__ import annotations

from typing import Protocol, cast

from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

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


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"connected: {client.graphname}")

    backend = TigerGraphRemoteBackend(client)

    # A few Account seeds. reference_epoch_s is the leakage-safe snapshot moment;
    # max_bins matches the data (13). In the real pipeline these come from the
    # seed parquet and derive_temporal_spec.
    seed_ids: tuple[str, ...] = ("A0000000001", "A0000000003", "A0000000012")
    loader = backend.make_loader(
        seed_ids=seed_ids,
        reference_epoch_s=1_600_000_000.0,
        max_bins=13,
        batch_size=2,
    )

    print("=" * 70)
    print("Pulling one HeteroData batch from the NodeLoader")
    print("=" * 70)
    batch = cast(HeteroData, next(iter(loader)))

    print("NODE STORES:")
    for ntype in batch.node_types:
        store = _node_store(batch, ntype)
        n = int(store.n_id.shape[0])
        if hasattr(store, "x"):
            print(f"  {ntype:16s}: {n} nodes, x={tuple(store.x.shape)}")
        else:
            print(f"  {ntype:16s}: {n} nodes, (no features - structural)")

    print()
    print("EDGE STORES:")
    for etype in batch.edge_types:
        store = _edge_store(batch, etype)
        ne = int(store.edge_index.shape[1])
        line = f"  {str(etype):48s}: {ne} edges"
        if etype == _HAS_PAID and hasattr(store, "edge_attr"):
            line += f", edge_attr={tuple(store.edge_attr.shape)}"
        print(line)

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    acct = _node_store(batch, "Account")
    print(f"  Account node features: {tuple(acct.x.shape)}  (node fetch via FeatureStore)")
    hp = _edge_store(batch, _HAS_PAID)
    if hasattr(hp, "edge_attr"):
        print(f"  HAS_PAID edge features: {tuple(hp.edge_attr.shape)}  (transform via temporal.py)")
    print("  Full HeteroData batch assembled: structure + node x + edge_attr.")
    print("  This is what the GATv2 model will consume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
