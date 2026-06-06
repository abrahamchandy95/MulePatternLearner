from __future__ import annotations

from typing import cast

import torch
from torch_geometric.sampler import NodeSamplerInput

from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.fanout import NeighborFanout
from mule_pattern_learner.pyg.node_id_mapper import NodeIDMapper
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"connected: {client.graphname}")

    # One backend holds the shared client + persistent mapper.
    backend = TigerGraphRemoteBackend(client)
    print("=" * 66)
    print("STEP 1: backend starts with an empty shared mapper")
    print("=" * 66)
    print(f"  mapper node types so far: {backend.mapper.node_types()}")

    # Build a sampler from the backend; it shares the backend's mapper.
    seed_ids: tuple[str, ...] = ("A0000000001", "A0000000003")
    sampler = backend.make_sampler(seed_ids=seed_ids, fanout=NeighborFanout())
    print()
    print("=" * 66)
    print("STEP 2: sampler built from backend shares the SAME mapper object")
    print("=" * 66)
    sampler_mapper = cast(NodeIDMapper, cast(object, getattr(sampler, "_mapper")))
    print(f"  sampler._mapper is backend.mapper: {sampler_mapper is backend.mapper}")

    # Sample a batch (seed indices 0,1).
    index = NodeSamplerInput(
        input_id=None,
        node=torch.tensor([0, 1], dtype=torch.long),
        time=None,
        input_type="Account",
    )
    print()
    print("=" * 66)
    print("STEP 3: sample a batch -> sampler registers ids into shared mapper")
    print("=" * 66)
    out = sampler.sample_from_nodes(index)
    n_accounts = int(out.node["Account"].shape[0])
    print(f"  sampled Account nodes: {n_accounts}")

    # The key proof: the BACKEND's mapper now knows these nodes, because the
    # sampler registered into the shared object. This is what lets a feature
    # store (built from the same backend) reverse the node-tensor integers.
    print()
    print("=" * 66)
    print("STEP 4: backend.mapper now knows the sampled nodes (the shared state)")
    print("=" * 66)
    for ntype in backend.mapper.node_types():
        count = backend.mapper.num_nodes(ntype)
        print(f"  {ntype:16s}: {count} ids registered")

    # Round-trip: take an integer from the node tensor, reverse via the
    # backend's mapper to the global id (what the feature store will do).
    first_int = int(out.node["Account"].tolist()[0])
    recovered = backend.mapper.to_string("Account", first_int)
    print()
    print("EDGE OF THE CONTRACT: reverse a node-tensor integer via backend mapper")
    print(f"  node tensor Account[0] = integer {first_int}")
    print(f"  backend.mapper.to_string('Account', {first_int}) = {recovered}")
    print("  -> the feature store will use exactly this to fetch features by id.")

    print()
    print("Backend works: it wires the sampler to a shared, persistent mapper,")
    print("so node-tensor integers stay reversible to global ids across the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
