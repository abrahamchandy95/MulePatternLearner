"""The heterogeneous attention model on subgraphs of the reference graph's standard world
(testing.world), at T (Monday 2024-07-01) in training.

Ann's subgraph holds her deposit account, her credit card (an account of the same owner),
Ben, who shares her home, and Old, who paid her. Max's holds Mia, found 5 days before T.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from mule_pattern_learner.config import DEFAULT_CONFIG, ModelConfig
from mule_pattern_learner.model.hetero import (
    LABEL_WIDTH,
    EdgeType,
    HeteroModel,
    label_inputs,
    protected,
)
from mule_pattern_learner.model.time_encoding import encode_reference
from mule_pattern_learner.subgraph.build import build
from mule_pattern_learner.testing.reference_graph import DAY_MS
from mule_pattern_learner.testing.world import T, standard_world
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    Phase,
    SubgraphBatch,
    SubgraphRequest,
    parse_subgraphs,
    subgraph_params,
)

GRAPH = standard_world().graph()
# No dropout, so in training only the label mask is random.
NO_DROPOUT = replace(DEFAULT_CONFIG.model, dropout=0.0)
ACCELERATORS = [
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


def _batch(seeds: list[str]) -> SubgraphBatch:
    request = SubgraphRequest(tuple(seeds), T, Phase.TRAINING)
    params = subgraph_params(request, DEFAULT_CONFIG.subgraph)
    return parse_subgraphs(GRAPH.run(FETCH_SUBGRAPHS, params), request)


def _accounts(batch: SubgraphBatch) -> list[str]:
    """The account of each account node, in the order build lays them out."""
    return [node.id for subgraph in batch.subgraphs for node in subgraph.accounts]


def _model(
    data: HeteroData, config: ModelConfig = DEFAULT_CONFIG.model, *, labels: bool = False
) -> HeteroModel:
    torch.manual_seed(0)
    nodes = {kind: int(data[kind].x.shape[1]) for kind in data.node_types}
    edges: dict[EdgeType, int] = {
        edge_type: int(data[edge_type].edge_attr.shape[1]) for edge_type in data.edge_types
    }
    return HeteroModel(nodes, edges, config, label_inputs=labels)


def _all_confirmed(data: HeteroData) -> Tensor:
    """Every account node confirmed a day before its seed's as-of time."""
    return torch.full((data["account"].num_nodes,), DAY_MS, dtype=torch.long)


def test_a_forward_pass_gives_each_seed_one_score_and_trains_every_layer() -> None:
    data = build([_batch(["A_MIA", "A_ANN", "A_QUY"])])
    model = _model(data)
    scores = model(data)
    assert scores.shape == (3,) and bool(torch.isfinite(scores).all())
    scores.sum().backward()
    # Mia's subgraph has payments, so what both layers' payment attention passes on, with
    # the payments' edge features, and the head learn. (A seed that receives one payment
    # gives all its attention to it, so the keys of the last layer may learn nothing.)
    learn = ("lin_value", "lin_edge", "lin_skip")
    for name, parameter in model.named_parameters():
        if ("pays" in name and any(part in name for part in learn)) or name.startswith("head."):
            assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name


def test_a_seeds_score_does_not_depend_on_the_other_seeds_of_its_batch() -> None:
    together = build([_batch(["A_MIA", "A_ANN", "A_QUY"])])
    model = _model(together).eval()
    alone = [model(build([_batch([seed])])) for seed in ("A_MIA", "A_ANN", "A_QUY")]
    torch.testing.assert_close(model(together), torch.cat(alone))


def test_the_model_refuses_a_batch_of_other_types_and_misplaced_confirmations() -> None:
    data = build([_batch(["A_MIA"])])
    plain, labelled = _model(data), _model(data, labels=True)
    with pytest.raises(ValueError, match="needs the confirmations"):
        labelled(data)
    with pytest.raises(ValueError, match="no label inputs"):
        plain(data, _all_confirmed(data))
    with pytest.raises(ValueError, match="one time per account node"):
        labelled(data, _all_confirmed(data)[1:])
    del data["account", "at_address", "address"]
    with pytest.raises(ValueError, match="node and edge types"):
        plain(data)


def test_the_seed_and_every_account_of_its_owner_are_protected() -> None:
    """Ann's card is hers, so it is protected in her subgraph; Ben shares her home but is
    another customer, so he is not. Max's and Mia's accounts each have only themselves."""
    batch = _batch(["A_ANN", "A_MAX", "A_MIA"])
    data = build([batch])
    accounts = _accounts(batch)
    shielded = [accounts[i] for i in protected(data).nonzero().flatten().tolist()]
    assert shielded == ["A_ANN", "A_ANN_CARD", "A_MAX", "A_MIA"]
    ann = batch.subgraphs[0]
    assert {"A_BEN", "A_OLD"} <= {node.id for node in ann.accounts}


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_masking_never_shows_the_seed_or_its_owners_accounts(device: str) -> None:
    data = build([_batch(["A_ANN", "A_MAX", "A_MIA"])], device=device)
    since = _all_confirmed(data).to(device)
    shielded = protected(data)
    shown = []
    torch.manual_seed(0)
    for _ in range(200):
        columns = label_inputs(data, since, 0.5)
        assert columns.shape == (len(since), LABEL_WIDTH) and columns.device.type == device
        assert not columns[shielded].any()
        shown.append(columns[~shielded, 0])
    # Every other confirmed account is hidden in about half of the training steps...
    share = torch.stack(shown).mean(dim=0).cpu()
    assert bool(((share > 0.35) & (share < 0.65)).all())
    # ...and none is hidden at evaluation, each with the encoding of its confirmation's age.
    columns = label_inputs(data, since, 0.0).cpu()
    assert not columns[shielded.cpu()].any()
    assert bool((columns[~shielded.cpu(), 0] == 1).all())
    expected = encode_reference(np.array(DAY_MS))
    assert np.abs(columns[~shielded.cpu(), 1:].numpy() - expected).max() < 1e-5


def test_an_account_not_confirmed_by_the_as_of_time_reads_as_unknown() -> None:
    data = build([_batch(["A_MAX"])])
    since = torch.full((data["account"].num_nodes,), -1, dtype=torch.long)
    assert not label_inputs(data, since, 0.0).any()


def test_only_other_customers_confirmations_reach_the_score() -> None:
    """Changing what the seed's or its card's confirmations say leaves Ann's score as it
    is, in training and at evaluation; Ben's changes it."""
    batch = _batch(["A_ANN"])
    data = build([batch])
    accounts = _accounts(batch)
    model = _model(data, NO_DROPOUT, labels=True)
    unknown = torch.full((len(accounts),), -1, dtype=torch.long)

    def scored(confirmed: list[str], *, training: bool) -> Tensor:
        since = unknown.clone()
        since[[accounts.index(account) for account in confirmed]] = DAY_MS
        model.train(training)
        torch.manual_seed(1)
        return model(data, since)

    for training in (True, False):
        nobody = scored([], training=training)
        torch.testing.assert_close(scored(["A_ANN", "A_ANN_CARD"], training=training), nobody)
    assert not torch.equal(scored(["A_BEN"], training=False), scored([], training=False))


def test_confirmations_are_hidden_at_random_in_training_and_never_at_evaluation() -> None:
    data = build([_batch(["A_ANN", "A_MAX", "A_MIA"])])
    since = _all_confirmed(data)
    model = _model(data, NO_DROPOUT, labels=True)
    model.eval()
    torch.testing.assert_close(model(data, since), model(data, since))
    model.train()
    draws = torch.stack([model(data, since) for _ in range(8)])
    assert bool((draws.std(dim=0) > 0).all())
