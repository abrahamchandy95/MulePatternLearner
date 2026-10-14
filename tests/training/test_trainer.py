"""Training on the reference graph's standard world (testing.world), and model.pt.

With no maturity gap the training examples are Mia, found 5 days before T (Monday
2024-07-01), on her 8 dates before it, beside the other open accounts; validation ranks
Max and Roy, found after T, on the dates they were active.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from functools import cache
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from mule_pattern_learner.config import DEFAULT_CONFIG, SubgraphConfig, TrainingConfig
from mule_pattern_learner.data.examples import training_examples, validation_examples
from mule_pattern_learner.data.grid import windows
from mule_pattern_learner.data.supervision import read_supervision
from mule_pattern_learner.model.loss import NonNegativePULoss
from mule_pattern_learner.subgraph.build import LAYOUTS
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.reference_graph import DAY_MS
from mule_pattern_learner.testing.world import T, standard_world
from mule_pattern_learner.tigergraph.queries import (
    SUBGRAPH_VERSION,
    AccountNode,
    PaymentEdge,
    Phase,
    SubgraphBatch,
)
from mule_pattern_learner.training import trainer
from mule_pattern_learner.training.selection import Recall
from mule_pattern_learner.training.trainer import (
    MODEL_FILE,
    Batch,
    Trained,
    batches,
    fingerprint,
    load,
    save,
    scaled,
    scores,
    since_confirmed,
    train,
)

D = DAY_MS
DATA = replace(DEFAULT_CONFIG.data, maturity_gap_days=0)
MODEL = DEFAULT_CONFIG.model
SUBGRAPH = DEFAULT_CONFIG.subgraph
# A few epochs of one prior, so the tests stay short.
SHORT = replace(DEFAULT_CONFIG.training, max_epochs=3, patience=3, priors=(0.01,), batch_size=32)


@cache
def _world() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, SubgraphBatch], dict[str, int]]:
    """The training and validation examples, their subgraphs, and the discovery of each
    mule revealed by the validation labels' cutoff."""
    graph = standard_world().graph()
    window = windows(DATA)
    training = training_examples(graph, window.training, DATA)
    validation = validation_examples(graph, window.validation, DATA)
    seeds = [
        (account_id, int(as_of_ms))
        for examples in (training, validation)
        for account_id, as_of_ms in zip(examples.account_id, examples.as_of_ms, strict=True)
    ]
    subgraphs = fetch(graph, seeds, Phase.TRAINING, DEFAULT_CONFIG.subgraph)
    assert window.validation.labels_before_ms is not None
    revealed = read_supervision(graph, window.validation.labels_before_ms, Phase.TRAINING)
    discovered = dict(zip(revealed.account_id, revealed.discovered_ms.tolist(), strict=True))
    return training, validation, subgraphs, discovered


def _batches(
    *, labels: bool = False, discovered: Mapping[str, int] | None = None
) -> tuple[list[Batch], list[Batch]]:
    """The training and validation batches, with the confirmations of discovered, or of
    the mules revealed, when labels."""
    training, validation, subgraphs, revealed = _world()
    found = (revealed if discovered is None else discovered) if labels else None
    return (
        batches(training, subgraphs, size=SHORT.batch_size, discovered=found, shuffle=0),
        batches(validation, subgraphs, size=64, discovered=found),
    )


def _trained(
    config: TrainingConfig = SHORT,
    *,
    seed: int = 0,
    labels: bool = False,
    discovered: Mapping[str, int] | None = None,
) -> Trained:
    made = _batches(labels=labels, discovered=discovered)
    return train(*made, MODEL, config, label_inputs=labels, seed=seed)


def _same_weights(left: Trained, right: Trained) -> bool:
    ours, theirs = left.model.state_dict(), right.model.state_dict()
    return ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_batches_keep_each_example_with_its_label_weight_and_as_of_time() -> None:
    _, validation, subgraphs, _ = _world()
    made = batches(validation, subgraphs, size=40)
    assert [len(batch.account_ids) for batch in made] == [40, 40, len(validation) - 80]
    assert [account for batch in made for account in batch.account_ids] == list(
        validation.account_id
    )
    joined = {
        "as_of_ms": torch.cat([batch.data.as_of_ms for batch in made]),
        "label": torch.cat([batch.label for batch in made]),
        "weight": torch.cat([batch.weight for batch in made]),
    }
    assert joined["as_of_ms"].tolist() == validation.as_of_ms.tolist()
    assert joined["label"].tolist() == validation.label.tolist()
    assert joined["weight"].tolist() == pytest.approx(validation.weight.tolist())
    assert all(batch.since_confirmed_ms is None for batch in made)


def test_shuffled_batches_hold_the_same_examples_in_a_seeded_order() -> None:
    training, _, subgraphs, _ = _world()

    def examples(made: list[Batch]) -> list[tuple[str, int, int]]:
        return [
            (account, int(as_of_ms), int(label))
            for batch in made
            for account, as_of_ms, label in zip(
                batch.account_ids, batch.data.as_of_ms, batch.label, strict=True
            )
        ]

    first = examples(batches(training, subgraphs, size=32, shuffle=0))
    assert first == examples(batches(training, subgraphs, size=32, shuffle=0))
    assert first != examples(batches(training, subgraphs, size=32, shuffle=1))
    rows = zip(training.account_id, training.as_of_ms, training.label, strict=True)
    assert sorted(first) == sorted((account, int(t), int(label)) for account, t, label in rows)


def test_batches_refuse_an_example_without_its_subgraph_or_with_a_bad_label() -> None:
    training, _, subgraphs, _ = _world()
    stranger = pd.concat([training, training.head(1).assign(account_id="A_NEW")])
    with pytest.raises(ValueError, match="No subgraph was fetched"):
        batches(stranger, subgraphs, size=32)
    with pytest.raises(ValueError, match="label of 0 or 1"):
        batches(training.assign(label=2), subgraphs, size=32)


def test_a_confirmation_counts_from_the_discovery_at_or_before_the_as_of_time() -> None:
    """In Max's subgraph at T, Mia was found 5 days before; Max himself is found 20 days
    after, so he is not confirmed yet; Vic, said here to be found at T, reads 0."""
    _, validation, subgraphs, _ = _world()
    assert ((validation.account_id == "A_MAX") & (validation.as_of_ms == T)).any()
    [max_at_t] = [sub for sub in subgraphs[T].subgraphs if sub.seed == "A_MAX"]
    batch = replace(subgraphs[T], subgraphs=(max_at_t,))
    discovered = {"A_MIA": T - 5 * D, "A_MAX": T + 20 * D, "A_VIC": T}
    since = dict(zip([node.id for node in max_at_t.accounts], since_confirmed([batch], discovered)))
    assert {account: int(value) for account, value in since.items()} == {
        "A_MAX": -1,
        "A_MIA": 5 * D,
        "A_QUY": -1,
        "A_VIC": 0,
        "X_CASH": -1,
    }


def test_each_labels_weights_are_scaled_to_a_mean_of_1() -> None:
    training, _ = _batches()
    label = torch.cat([batch.label for batch in training])
    weight = torch.cat([batch.weight for batch in training])
    rescaled = torch.cat([batch.weight for batch in scaled(training)])
    for value in (0, 1):
        assert float(rescaled[label == value].mean()) == pytest.approx(1)
        ratio = rescaled[label == value] / weight[label == value]
        assert torch.allclose(ratio, ratio[0])


def test_two_mules_weigh_the_same_however_many_dates_they_span() -> None:
    """1000 mules positive on 1 date and 100 on 26, each weighing 1 in all as
    data.examples weighs them, with 15 unlabelled per positive sharing its weight, in
    random batches of 128 as batches makes them, shuffled four ways. Once train scales
    the weights, the loss pulls a mule of 26 dates up as hard in all as a mule of 1,
    whatever else shares their batches; dividing by a batch's weight of positives
    instead made it 2.4 times as hard."""
    dates = torch.tensor([1] * 1000 + [26] * 100)
    mule = torch.repeat_interleave(torch.arange(len(dates)), dates)  # each positive's mule
    share = 1 / dates[mule].double()
    label = torch.cat([torch.ones_like(mule), torch.zeros(15 * len(mule), dtype=torch.long)])
    weight = torch.cat([share, share.repeat_interleave(15) / 15])
    loss = NonNegativePULoss(0.01)
    pull = torch.zeros(len(dates), dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    for _ in range(4):
        chunks = torch.randperm(len(label), generator=generator).split(128)
        made = [Batch((), HeteroData(), label[chunk], weight[chunk], None) for chunk in chunks]
        for batch, chunk in zip(scaled(made), chunks, strict=True):
            logits = torch.zeros(len(chunk), dtype=torch.float64, requires_grad=True)
            step, _ = loss(logits, batch.label, batch.weight)
            (gradient,) = torch.autograd.grad(step, logits)
            positive = batch.label == 1
            pull.index_add_(0, mule[chunk[positive]], -gradient[positive])
    assert float(pull[1000:].mean() / pull[:1000].mean()) == pytest.approx(1, abs=0.03)


def test_a_short_training_run_lowers_the_loss_and_learns_from_the_graph() -> None:
    """A falling loss alone is no evidence: the head's bias lowers it with the graph
    layers cut off. The trained model also scores the training positives above the
    unlabelled, which a model that ignores its input cannot."""
    config = replace(SHORT, max_epochs=8, patience=8)
    trained = _trained(config)
    losses = [epoch.loss for epoch in trained.history]
    assert len(losses) == 8 and all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
    training, _ = _batches()
    score = scores(trained.model, training)
    label = torch.cat([batch.label for batch in training])
    weight = torch.cat([batch.weight for batch in training])

    def mean(value: int) -> float:
        on = label == value
        return float((score[on] * weight[on]).sum() / weight[on].sum())

    assert mean(1) > mean(0)


def _confirmed_long_ago() -> dict[str, int]:
    """Every account of the subgraphs confirmed at the start of time, so the label inputs
    of training read confirmations (the revealed mules are found after every training
    date) and the masks drawn while training change the scores."""
    _, _, subgraphs, _ = _world()
    return {
        node.id: 1
        for batch in subgraphs.values()
        for subgraph in batch.subgraphs
        for node in subgraph.accounts
    }


@pytest.mark.parametrize("labels", [False, True])
def test_training_is_deterministic_under_a_seed(labels: bool) -> None:
    config = replace(SHORT, priors=(0.003, 0.01))
    found = _confirmed_long_ago() if labels else None
    first, again, other = (
        _trained(config, seed=seed, labels=labels, discovered=found) for seed in (0, 0, 1)
    )
    assert (first.prior, first.epoch, first.history) == (again.prior, again.epoch, again.history)
    assert _same_weights(first, again)
    assert not _same_weights(first, other)


def _scripted(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    """Makes validation give the scores of values, in order, one per epoch."""
    script: Iterator[float] = iter(values)

    def recall(*_: Any) -> Recall:
        value = next(script)
        return Recall(value, value)

    monkeypatch.setattr(trainer, "validation_recall", recall)


def test_validation_chooses_the_best_epoch_of_each_prior_then_the_best_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With patience 2 the first prior stops two epochs after its best, its second; the
    second prior's best ties it, so the earlier prior is kept."""
    config = replace(SHORT, max_epochs=6, patience=2, priors=(0.003, 0.01))
    _scripted(monkeypatch, [0.1, 0.3, 0.2, 0.3] + [0.3, 0.2, 0.25])
    trained = _trained(config)
    assert [(epoch.prior, epoch.epoch) for epoch in trained.history] == [
        (0.003, 1),
        (0.003, 2),
        (0.003, 3),
        (0.003, 4),
        (0.01, 1),
        (0.01, 2),
        (0.01, 3),
    ]
    assert (trained.prior, trained.epoch) == (0.003, 2)
    assert trained.recall_by_prior() == {0.003: Recall(0.3, 0.3), 0.01: Recall(0.3, 0.3)}
    # The model kept is the first prior's after its second epoch.
    _scripted(monkeypatch, [0.1, 0.3])
    two_epochs = _trained(replace(config, max_epochs=2, priors=(0.003,)))
    assert _same_weights(trained, two_epochs)


def test_label_input_training_needs_batches_with_confirmations() -> None:
    trained = _trained(replace(SHORT, max_epochs=1), labels=True)
    assert trained.model.label_inputs
    plain, labelled = _batches(), _batches(labels=True)
    with pytest.raises(ValueError, match="training batches"):
        train(plain[0], labelled[1], MODEL, SHORT, label_inputs=True, seed=0)
    with pytest.raises(ValueError, match="validation batches"):
        train(plain[0], labelled[1], MODEL, SHORT, label_inputs=False, seed=0)


@pytest.mark.parametrize("labels", [False, True])
def test_a_saved_model_loads_back_the_same(tmp_path: Path, labels: bool) -> None:
    trained = _trained(replace(SHORT, max_epochs=2), labels=labels)
    path = tmp_path / MODEL_FILE
    save(trained, path, SUBGRAPH)
    loaded = load(path, SUBGRAPH)
    assert (loaded.prior, loaded.epoch, loaded.history) == (
        trained.prior,
        trained.epoch,
        trained.history,
    )
    assert (loaded.model.config, loaded.model.label_inputs) == (MODEL, labels)
    assert _same_weights(loaded, trained) and not loaded.model.training
    validation = _batches(labels=labels)[1]
    assert torch.equal(scores(loaded.model, validation), scores(trained.model, validation))


def _rewritten(path: Path, **changes: object) -> None:
    stored = torch.load(path, weights_only=True)
    torch.save(stored | changes, path)


def test_the_contract_lock_refuses_a_model_of_another_query_version(tmp_path: Path) -> None:
    path = tmp_path / MODEL_FILE
    save(_trained(replace(SHORT, max_epochs=1)), path, SUBGRAPH)
    assert torch.load(path, weights_only=True)["version"] == SUBGRAPH_VERSION
    _rewritten(path, version="subgraphs-000000000000")
    with pytest.raises(ValueError, match="fetch_subgraphs 'subgraphs-000000000000'"):
        load(path, SUBGRAPH)


def test_the_contract_lock_refuses_a_model_of_another_feature_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / MODEL_FILE
    save(_trained(replace(SHORT, max_epochs=1)), path, SUBGRAPH)
    # The code now lays the accounts out without one of their counts.
    account = LAYOUTS[AccountNode]
    fewer = replace(account, counts=account.counts[1:])
    monkeypatch.setattr(trainer, "LAYOUTS", LAYOUTS | {AccountNode: fewer})
    with pytest.raises(ValueError, match="another feature layout"):
        load(path, SUBGRAPH)
    monkeypatch.undo()
    load(path, SUBGRAPH)
    # The label inputs are part of the layout.
    assert fingerprint(True) != fingerprint(False)
    _rewritten(path, fingerprint=fingerprint(True))
    with pytest.raises(ValueError, match="another feature layout"):
        load(path, SUBGRAPH)


def test_the_fingerprint_holds_what_each_column_means_not_only_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The columns keep their names when another field says a duration is present, or
    when counts are transformed another way, but the features change."""
    before = fingerprint(False)
    payment = LAYOUTS[PaymentEdge]
    durations = tuple(
        (name, "" if name == "pair_age_ms" else present) for name, present in payment.durations
    )
    changed = replace(payment, durations=durations)
    assert changed.columns == payment.columns
    monkeypatch.setattr(trainer, "LAYOUTS", LAYOUTS | {PaymentEdge: changed})
    assert fingerprint(False) != before
    monkeypatch.undo()
    monkeypatch.setattr(trainer, "COUNT_TRANSFORM", "log")
    assert fingerprint(False) != before


def test_the_contract_lock_refuses_a_model_of_other_subgraph_settings(tmp_path: Path) -> None:
    """A model trained on a year of payments reads in_365d, gap_ms or pair_age_ms with
    that year's meaning, so it is not scored on subgraphs of 180 days."""
    path = tmp_path / MODEL_FILE
    save(_trained(replace(SHORT, max_epochs=1)), path, SUBGRAPH)
    shorter = SubgraphConfig(lookback_days=180)
    with pytest.raises(ValueError, match=r"other settings, \{'lookback_days': 365\}"):
        load(path, shorter)
    load(path, SUBGRAPH)
