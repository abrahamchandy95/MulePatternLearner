from __future__ import annotations

from mule_pattern_learner.pyg.feature_fetch import (
    fetch_account_vertices,
    fetch_has_paid_edges,
)
from mule_pattern_learner.schema.node_features import build_account_features
from mule_pattern_learner.tigergraph.temporal import build_edge_features
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"connected: {client.graphname}")

    # A few known accounts (in real use these come from the sampled batch's mapper).
    account_ids = ["A0000000001", "A0000000003", "A0000000012"]

    print("=" * 64)
    print("STEP 1: fetch raw Account vertices by id from TigerGraph")
    print(f"  ids: {account_ids}")
    print("=" * 64)
    vertices = fetch_account_vertices(client, account_ids)
    print(f"  fetched {len(vertices)} vertices")
    if vertices:
        first = vertices[0]
        if isinstance(first, dict):
            vid = first.get("v_id")
            attrs = first.get("attributes")
            n_attrs = len(attrs) if isinstance(attrs, dict) else 0
            print(f"  first vertex: v_id={vid}, {n_attrs} attributes")

    print()
    print("=" * 64)
    print("STEP 2: build feature tensor via node_features.py")
    print("=" * 64)
    features = build_account_features(vertices)
    print(f"  node_ids: {features.node_ids}")
    print(f"  feature tensor shape: {tuple(features.feats.shape)}  (N x 31)")
    print(f"  feature names (first 6): {features.feature_names[:6]}")

    print()
    print("EXAMPLE: first account's first 6 transformed features")
    if int(features.feats.shape[0]) > 0:
        row = features.feats[0].tolist()[:6]
        for name, val in zip(features.feature_names[:6], row):
            print(f"  {name:20s}: {val:.4f}")

    # ---- edge-feature half: fetch HAS_PAID edges + build temporal features ----
    print()
    print("=" * 64)
    print("STEP 3: fetch HAS_PAID edges (in-batch) + build edge features")
    print("=" * 64)
    edges = fetch_has_paid_edges(client, account_ids)
    print(f"  fetched {len(edges)} in-batch HAS_PAID edges")

    # reference_epoch_s is the snapshot moment (leakage-safe); use a fixed value
    # for the demo. max_bins is derived from the graph in real use; the live data
    # carries 13 bins per edge.
    reference_epoch_s = 1_600_000_000.0
    max_bins = 13
    edge_features = build_edge_features(edges, reference_epoch_s, max_bins)
    print(f"  scalar feats shape: {tuple(edge_features.scalar_feats.shape)}  (E x 8)")
    print(f"  bin sequence shape: {tuple(edge_features.bin_seq.shape)}  (E x bins x 2)")
    if edge_features.num_edges > 0:
        print(f"  first edge: {edge_features.src_ids[0]} -> {edge_features.dst_ids[0]}")

    print()
    print("Both halves now work: node features (fetch->node_features.py) and")
    print("edge features (fetch->temporal.py). The FeatureStore wraps both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
