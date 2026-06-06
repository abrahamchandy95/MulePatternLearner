from typing import Protocol, cast

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from mule_pattern_learner.features.edge_spec import flat_edge_dim
from mule_pattern_learner.features.nodes import NUM_ACCOUNT_FEATURES
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.model import MulePatternModel
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")
_MAX_BINS = 13


class _NodeStore(Protocol):
    n_id: Tensor
    x: Tensor


class _EdgeStore(Protocol):
    edge_index: Tensor
    edge_attr: Tensor


def _node_store(data: HeteroData, key: NodeType) -> _NodeStore:
    return cast(_NodeStore, cast(object, data[key]))


def _edge_store(data: HeteroData, key: EdgeType) -> _EdgeStore:
    return cast(_EdgeStore, cast(object, data[key]))


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"connected: {client.graphname}")

    backend = TigerGraphRemoteBackend(client)
    loader = backend.make_loader(
        seed_ids=("A0000000001", "A0000000003", "A0000000012"),
        reference_epoch_s=1_600_000_000.0,
        max_bins=_MAX_BINS,
        batch_size=2,
    )
    batch = cast(HeteroData, next(iter(loader)))

    x_dict: dict[NodeType, Tensor] = {
        "Account": _node_store(batch, "Account").x,
    }
    node_counts: dict[NodeType, int] = {
        ntype: int(_node_store(batch, ntype).n_id.shape[0]) for ntype in batch.node_types
    }
    edge_index_dict: dict[EdgeType, Tensor] = {
        etype: _edge_store(batch, etype).edge_index for etype in batch.edge_types
    }
    edge_attr_dict: dict[EdgeType, Tensor] = {
        _HAS_PAID: _edge_store(batch, _HAS_PAID).edge_attr,
    }

    model = MulePatternModel(
        account_in_dim=NUM_ACCOUNT_FEATURES,
        edge_dim=flat_edge_dim(_MAX_BINS),
        hidden_dim=64,
        heads=4,
    )

    print("=" * 64)
    print("FORWARD PASS (untrained) -> per-account logits")
    print("=" * 64)
    account_x = x_dict["Account"]
    hp_attr = edge_attr_dict[_HAS_PAID]
    print(f"  Account x in        : {tuple(account_x.shape)}  [N, {NUM_ACCOUNT_FEATURES}]")
    print(f"  HAS_PAID edge_attr  : {tuple(hp_attr.shape)}  [E, {flat_edge_dim(_MAX_BINS)}]")
    print(f"  node counts         : {node_counts}")

    _ = model.eval()
    logits: Tensor
    with torch.no_grad():
        logits = cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))

    n_account = node_counts.get("Account", 0)
    n_logits = int(logits.shape[0])
    print()
    print(f"  logits shape        : {tuple(logits.shape)}  (expected [{n_account}])")
    print(f"  one logit per account: {n_logits == n_account}")
    sample = cast(list[float], logits[:5].tolist())
    print(f"  first logits        : {[round(v, 4) for v in sample]}")

    print()
    print("Model produces one mule-likelihood logit per Account node.")
    print("Next: nnPU loss + training loop turn these logits into a trained ranker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
