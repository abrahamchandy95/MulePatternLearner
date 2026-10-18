"""The heterogeneous attention model on batches built from the reference world of conftest.py:
its parts as docs/design.md's "The model" gives them, the shape of what each relation passes
on, a score that never depends on its batch, the head with and without the community block,
every ablation's batch, and the batches it refuses."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from torch.nn import BatchNorm1d, Embedding, LayerNorm, Linear, Module
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, TransformerConv

from mule_pattern_learner.config import DEFAULT_CONFIG, ModelConfig
from mule_pattern_learner.model.encode import encode_batch
from mule_pattern_learner.model.hetero import HeteroModel
from mule_pattern_learner.subgraph.build import Batch, ExampleRows, build
from mule_pattern_learner.subgraph.layouts import LAYOUT, OWNED_BY, PAYS, RELATIONS, reverse
from mule_pattern_learner.subgraph.switches import Switch, apply

# No dropout, so a model in training mode gives the same scores twice.
NO_DROPOUT = replace(DEFAULT_CONFIG.model, dropout=0.0)
WITHOUT_COMMUNITY = replace(DEFAULT_CONFIG.model, community=False)


def _model(config: ModelConfig = DEFAULT_CONFIG.model, seed: int = 0) -> HeteroModel:
    torch.manual_seed(seed)
    return HeteroModel(LAYOUT, config)


def _encoded(batch: Batch) -> HeteroData:
    return encode_batch(batch.data, LAYOUT)


def test_the_parts_are_the_designs() -> None:
    """Per node type a projection to 64; two layers of one TransformerConv per relation, 4
    heads of 16, with the relation's encoded attributes, none on ownership; the head
    Linear(72, 32), ReLU, Linear(32, 1)."""
    model = _model()
    assert {kind: layer.in_features for kind, layer in model.project.items()} == {
        "account": 309,
        "customer": 11,
        "device": 12,
        "ip": 12,
    }
    assert {layer.out_features for layer in model.project.values()} == {64}
    assert len(model.layers) == 2
    widths = {"pays": 401, "seen_on_device": 131, "seen_on_ip": 131}
    for layer in model.layers:
        assert isinstance(layer, HeteroConv)
        convs: dict[Any, Module] = dict(layer.convs.items())
        assert len(convs) == 16
        for key, conv in convs.items():
            assert isinstance(conv, TransformerConv)
            assert (conv.heads, conv.out_channels, conv.concat) == (4, 16, True)
            name = key[1].removeprefix("rev_")
            if name == "owned_by":
                assert conv.lin_edge is None
            else:
                assert conv.lin_edge is not None
                assert conv.lin_edge.in_channels == widths.get(name, 130)
    head = [part for part in model.head if isinstance(part, Linear)]
    assert [(part.in_features, part.out_features) for part in head] == [(72, 32), (32, 1)]
    without = _model(WITHOUT_COMMUNITY)
    first = without.head[0]
    assert isinstance(first, Linear) and first.in_features == 64


def test_there_is_no_embedding_memory_or_batch_statistics() -> None:
    model = _model()
    kinds = {type(module) for module in model.modules()}
    assert not kinds & {Embedding, BatchNorm1d, LayerNorm}
    assert not any(buffer.numel() for buffer in model.buffers())


def test_each_relation_passes_on_one_row_per_node_it_reaches(raw_batch: Batch) -> None:
    data = _encoded(raw_batch)
    model = _model()
    shapes: dict[tuple[int, Any], tuple[int, ...]] = {}
    hooks = []
    for depth, layer in enumerate(model.layers):
        assert isinstance(layer, HeteroConv)
        for key, conv in layer.convs.items():

            def record(
                _: Module, __: tuple[Any, ...], out: Any, depth: int = depth, key: Any = key
            ) -> None:
                shapes[(depth, key)] = tuple(out.shape)

            hooks.append(conv.register_forward_hook(record))
    scores = model(data)
    for hook in hooks:
        hook.remove()
    assert scores.shape == (32,) and bool(torch.isfinite(scores).all())
    assert len(shapes) == 2 * len(RELATIONS)
    for (_, key), shape in shapes.items():
        assert shape == (data[key[2]].num_nodes, 64), key
    # The batch holds edges of every relation.
    for relation in RELATIONS:
        assert data[relation].edge_index.shape[1] > 0, relation


def test_training_reaches_every_part(raw_batch: Batch) -> None:
    data = _encoded(raw_batch)
    model = _model(NO_DROPOUT)
    model(data).sum().backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("head.", "project.account")) or (
            "pays" in name and "lin_value" in name
        ):
            assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name


def test_a_score_never_depends_on_its_batch(
    model_rows: list[ExampleRows], raw_batch: Batch
) -> None:
    """Each of the 32 examples alone gives the logit it gives among the 31 others."""
    model = _model().eval()
    with torch.no_grad():
        together = model(_encoded(raw_batch))
        alone = torch.cat([model(_encoded(build([rows], "w"))) for rows in model_rows])
    torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-5)
    assert len(set(together.tolist())) > 1


def test_the_community_block_reaches_the_score_only_through_the_head(raw_batch: Batch) -> None:
    data = _encoded(raw_batch)
    changed = data.clone()
    changed.community = changed.community + 1
    with torch.no_grad():
        model = _model().eval()
        assert not torch.allclose(model(data), model(changed))
        without = _model(WITHOUT_COMMUNITY).eval()
        torch.testing.assert_close(without(data), without(changed), rtol=0, atol=0)


@pytest.mark.parametrize("switch", list(Switch))
def test_every_ablation_batch_is_scored(raw_batch: Batch, switch: Switch) -> None:
    switched = apply(raw_batch, switch)
    model = _model(replace(DEFAULT_CONFIG.model, community=switch.community)).eval()
    with torch.no_grad():
        scores = model(_encoded(switched))
    assert scores.shape == (len(raw_batch.keys),) and bool(torch.isfinite(scores).all())


def test_the_scored_account_alone_reads_only_its_own_row_and_payments(raw_batch: Batch) -> None:
    """Under the switch, what any node but the scored account holds changes no score."""
    model = _model(WITHOUT_COMMUNITY).eval()
    with torch.no_grad():
        before = model(_encoded(apply(raw_batch, Switch.SCORED_ALONE)))
        data = raw_batch.data.clone()
        others = torch.ones(data["account"].num_nodes, dtype=torch.bool)
        others[data.scored_index] = False
        data["account"].x[others] += 1
        for kind in ("customer", "device", "ip"):
            data[kind].x += 1
        data.community += 1
        after = model(_encoded(apply(Batch(raw_batch.keys, data), Switch.SCORED_ALONE)))
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_dropout_acts_only_in_training(raw_batch: Batch) -> None:
    data = _encoded(raw_batch)
    model = _model()
    with torch.no_grad():
        model.eval()
        torch.testing.assert_close(model(data), model(data), rtol=0, atol=0)
        model.train()
        draws = torch.stack([model(data) for _ in range(4)])
    assert bool((draws.std(dim=0) > 0).any())


def test_a_batch_not_encoded_for_the_model_is_refused(raw_batch: Batch) -> None:
    model = _model()
    with pytest.raises(ValueError, match="account rows are 57 wide, not 309"):
        model(raw_batch.data)
    data = _encoded(raw_batch)
    del data[reverse(PAYS)]
    with pytest.raises(ValueError, match="node and edge types"):
        model(data)
    data = _encoded(raw_batch)
    data[OWNED_BY].edge_attr = torch.zeros((data[OWNED_BY].edge_index.shape[1], 3))
    with pytest.raises(ValueError, match="owned_by are 3 wide, not None"):
        model(data)
    data = _encoded(raw_batch)
    data.community = data.community[:, :7]
    with pytest.raises(ValueError, match="community block"):
        model(data)
