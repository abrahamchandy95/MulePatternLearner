import torch
from torch_geometric.sampler import NodeSamplerInput

from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.indexing.node_id_mapper import NodeIDMapper
from mule_pattern_learner.pyg.sampler import TigerGraphHeteroSampler
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"connected: {client.graphname}")

    # Seed vocabulary: ordered list of seed account ids. Position i == index i.
    # In real training this comes from the PU-label parquet; here, two known seeds.
    seed_ids: tuple[str, ...] = ("A0000000001", "A0000000003")

    # The sampler shares a NodeIDMapper (in real use this comes from the backend,
    # shared with the feature store). Here we create it explicitly to inspect it.
    mapper = NodeIDMapper()
    sampler = TigerGraphHeteroSampler(
        client=client,
        seed_ids=seed_ids,
        mapper=mapper,
        fanout=NeighborFanout(),
    )

    # PyG hands the sampler integer seed indices; sample from both seeds (0, 1).
    index = NodeSamplerInput(
        input_id=None,
        node=torch.tensor([0, 1], dtype=torch.long),
        time=None,
        input_type="Account",
    )

    print("=" * 66)
    print("TigerGraphHeteroSampler.sample_from_nodes  (seed indices 0, 1)")
    print(f"  index 0 -> {seed_ids[0]}")
    print(f"  index 1 -> {seed_ids[1]}")
    print("=" * 66)

    out = sampler.sample_from_nodes(index)

    print("STRUCTURE returned as HeteroSamplerOutput")
    print("  node tensors (contiguous local indices, per type):")
    for ntype, node_tensor in out.node.items():
        n = int(node_tensor.shape[0])
        print(f"    {ntype:16s}: {n} nodes  (local 0..{n - 1})")

    print("  edges per type (row/col local-index pairs):")
    total_edges = 0
    for etype in out.row:
        n_edges = int(out.row[etype].shape[0])
        total_edges += n_edges
        print(f"    {str(etype):48s}: {n_edges}")
    print(f"    {'TOTAL':48s}: {total_edges}")

    # The mapper (shared with the sampler) now knows every sampled node, since
    # the sampler registered them during sampling. metadata holds PyG's
    # (input_id, batch_size), not the mapper.
    meta = mapper

    print()
    print("MAPPER (NodeIDMapper: integer id -> global string id, per type)")
    for ntype in meta.node_types():
        n = meta.num_nodes(ntype)
        sample_ids = meta.to_strings(ntype, list(range(min(5, n))))
        print(f"  {ntype:16s}: {n} ids, e.g. {sample_ids}")

    # Decode one real edge end-to-end to prove structure correctness:
    # take the first HAS_PAID edge and resolve its local indices to global ids
    # through the mapper (int -> string), exactly as the FeatureStore will.
    has_paid = ("Account", "HAS_PAID", "Account")
    if has_paid in out.row and int(out.row[has_paid].shape[0]) > 0:
        r = int(out.row[has_paid].tolist()[0])
        c = int(out.col[has_paid].tolist()[0])
        src_id = meta.to_string("Account", r)
        dst_id = meta.to_string("Account", c)
        print()
        print("EDGE DECODE CHECK (first HAS_PAID edge):")
        print(f"  local row={r}, col={c}")
        print(f"  -> {src_id}  --HAS_PAID-->  {dst_id}")
        print("  (resolved via the mapper, the same int->string the FeatureStore uses)")

    print()
    print("Index i in each node tensor maps via mapper.to_string(type, i).")
    print("The FeatureStore reverses these ints to global ids to fetch features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
