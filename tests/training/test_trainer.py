"""The trainer on the batch world of conftest.py: it learns the planted mules, every run of one
seed is the same, a run resumed from its checkpoint ends with the weights of one that never
stopped, priors and epochs are kept only when strictly better, the model files keep their
lock, ablations train on switched batches, and no query of any kind is run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG, TrainingConfig
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.switches import Switch
from mule_pattern_learner.tigergraph import executor as executor_module
from mule_pattern_learner.training import trainer as trainer_module
from mule_pattern_learner.training.batches import training_batches
from mule_pattern_learner.training.priors import Prior
from mule_pattern_learner.training.selection import Found
from mule_pattern_learner.training.trainer import (
    GNN,
    SelectionWorld,
    Trained,
    check_determinism,
    gnn_lock,
    load_trained,
    model_config,
    model_file,
    ranker_name,
    save_trained,
    train,
    train_rotation,
)

MODEL = DEFAULT_CONFIG.model
SUBGRAPH = DEFAULT_CONFIG.subgraph
# Short runs at the design's learning rate.
SHORT = replace(DEFAULT_CONFIG.training, batch_size=32, max_epochs=3, patience=3)
PRIOR = Prior(1.0, 0.03)


def _selection(world: Any, size: int = 64) -> SelectionWorld:
    return SelectionWorld(world.cache, world.identity, world.selection_keys(), size)


def _small(world: Any) -> list[Batch]:
    """Three batches of 16 of the training examples of T: the six mules and 42 others."""
    frame = world.examples()
    frame = frame[frame["t_ms"] == world.mondays[0]].reset_index(drop=True)
    chosen = frame[(frame["label"] == 1) | (frame.index < 42)]
    assert len(chosen) == 48
    return training_batches(world.cache, world.identity, chosen, 16)


def _train(
    batches: Sequence[Batch],
    selection: SelectionWorld,
    settings: TrainingConfig = SHORT,
    priors: Sequence[Prior] = (PRIOR,),
    **changes: Any,
) -> Trained:
    return train(
        batches,
        selection,
        priors,
        0,
        model=MODEL,
        settings=settings,
        subgraph=SUBGRAPH,
        **changes,
    )


def _same_weights(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    ours, theirs = left.state_dict(), right.state_dict()
    return ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_training_learns_the_planted_mules_and_not_shuffled_labels(
    world: Any, planted: list[Batch]
) -> None:
    """Trained on the mules, the model ranks them far higher on the selection world than one
    trained from the same start on six ordinary accounts drawn as positives. The best a ranking
    can score is (2.24 / 6 + 6 / 6) / 2, about 0.687: the top 1% of 224 accounts holds 2.24 of
    them."""
    selection = _selection(world)
    learned = _train(planted, selection)
    labels = world.examples(world.shuffled())
    shuffled = _train(training_batches(world.cache, world.identity, labels, 32), selection)
    assert learned.score >= 0.5
    assert shuffled.score <= 0.25
    assert learned.score > shuffled.score + 0.25
    # The history holds every epoch, and the kept epoch is the first with the best score.
    scores = [epoch.score for epoch in learned.history]
    assert [epoch.epoch for epoch in learned.history] == [1, 2, 3]
    assert learned.score == max(scores) and learned.epoch == scores.index(max(scores)) + 1
    assert learned.found.score == learned.score


def test_two_runs_of_one_seed_are_identical(world: Any) -> None:
    batches, selection = _small(world), _selection(world)
    settings = replace(SHORT, max_epochs=2)
    first = _train(batches, selection, settings)
    second = _train(batches, selection, settings)
    assert _same_weights(first.model, second.model)
    assert first.history == second.history
    other = train(batches, selection, [PRIOR], 1, model=MODEL, settings=settings, subgraph=SUBGRAPH)
    assert not _same_weights(first.model, other.model)


def test_a_resumed_run_ends_with_the_weights_of_one_that_never_stopped(
    world: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches, selection = _small(world), _selection(world)
    settings = replace(SHORT, max_epochs=3, patience=3)
    whole = _train(batches, selection, settings, checkpoints=tmp_path / "whole")
    # A run that stops in its third epoch, after the checkpoint of its second.
    real = trainer_module.selection_found
    scored: list[int] = []

    def stopping(*args: Any) -> Found:
        scored.append(1)
        if len(scored) == 3:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(trainer_module, "selection_found", stopping)
    with pytest.raises(KeyboardInterrupt):
        _train(batches, selection, settings, checkpoints=tmp_path / "stopped")
    monkeypatch.setattr(trainer_module, "selection_found", real)
    checkpoint = tmp_path / "stopped" / "gnn_seed0_prior1.pt"
    assert checkpoint.is_file()
    assert not [path for path in (tmp_path / "stopped").iterdir() if path.name.startswith(".")]
    resumed = _train(batches, selection, settings, checkpoints=tmp_path / "stopped")
    assert _same_weights(whole.model, resumed.model)
    assert whole.history == resumed.history
    assert (whole.epoch, whole.score) == (resumed.epoch, resumed.score)
    # A finished prior is read back from its checkpoint, not trained again.
    monkeypatch.setattr(
        "mule_pattern_learner.training.trainer._PriorTraining._epoch", None, raising=True
    )
    again = _train(batches, selection, settings, checkpoints=tmp_path / "stopped")
    assert _same_weights(whole.model, again.model)


def test_a_checkpoint_of_another_lock_or_prior_is_refused(world: Any, tmp_path: Path) -> None:
    batches, selection = _small(world), _selection(world)
    settings = replace(SHORT, max_epochs=1)
    _train(batches, selection, settings, checkpoints=tmp_path)
    moved = replace(SUBGRAPH, own_accounts=7)
    with pytest.raises(ValueError, match="settings.own_accounts"):
        train(
            batches,
            selection,
            [PRIOR],
            0,
            model=MODEL,
            settings=settings,
            subgraph=moved,
            checkpoints=tmp_path,
        )
    with pytest.raises(ValueError, match="another prior"):
        _train(batches, selection, settings, [Prior(1.0, 0.04)], checkpoints=tmp_path)


def test_an_epoch_and_a_prior_are_kept_only_when_strictly_better(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every selection score equal, the first epoch of the first prior is kept, and a
    prior stops after patience epochs without a better one."""
    batches, selection = _small(world), _selection(world)
    flat = Found(("a",), np.ones(1), np.full((1, 2), 0.5))

    def flat_found(*args: object) -> Found:
        return flat

    monkeypatch.setattr(trainer_module, "selection_found", flat_found)
    settings = replace(SHORT, max_epochs=5, patience=2)
    trained = _train(batches, selection, settings, [PRIOR, Prior(3.0, 0.09)])
    assert [(epoch.multiple, epoch.epoch) for epoch in trained.history] == [
        (1.0, 1),
        (1.0, 2),
        (1.0, 3),
        (3.0, 1),
        (3.0, 2),
        (3.0, 3),
    ]
    assert (trained.prior, trained.epoch, trained.score) == (PRIOR, 1, 0.5)
    rising = iter([0.1, 0.2, 0.3, 0.1, 0.3, 0.2, 0.2])

    def better(*args: Any) -> Found:
        return Found(("a",), np.ones(1), np.full((1, 2), next(rising)))

    monkeypatch.setattr(trainer_module, "selection_found", better)
    settings = replace(SHORT, max_epochs=4, patience=2)
    trained = _train(batches, selection, settings, [PRIOR, Prior(3.0, 0.09)])
    # The first prior climbs to 0.3 at epoch 3 and stops at its 4th, the second ties 0.3.
    assert [round(epoch.score, 1) for epoch in trained.history] == [
        0.1,
        0.2,
        0.3,
        0.1,
        0.3,
        0.2,
        0.2,
    ]
    assert (trained.prior, trained.epoch) == (PRIOR, 3)


def test_the_model_file_keeps_its_lock_and_reads_back(world: Any, tmp_path: Path) -> None:
    batches, selection = _small(world), _selection(world)
    trained = _train(batches, selection, replace(SHORT, max_epochs=1))
    lock = gnn_lock(None, MODEL, SUBGRAPH, SHORT)
    path = model_file(tmp_path, GNN, 0)
    save_trained(trained, path, lock)
    saved = load_trained(path, lock)
    assert _same_weights(saved.model, trained.model) and not saved.model.training
    assert (saved.ranker, saved.seed, saved.prior, saved.epoch, saved.score) == (
        GNN,
        0,
        PRIOR,
        trained.epoch,
        trained.score,
    )
    assert saved.found.accounts == trained.found.accounts
    assert saved.history == trained.history
    assert [(run.prior, run.epoch, run.score) for run in saved.priors] == [
        (run.prior, run.epoch, run.score) for run in trained.runs
    ]
    np.testing.assert_array_equal(saved.priors[0].found.shares, trained.runs[0].found.shares)
    with pytest.raises(ValueError, match="settings.hub_threshold"):
        load_trained(path, gnn_lock(None, MODEL, replace(SUBGRAPH, hub_threshold=150), SHORT))
    with pytest.raises(ValueError, match="ranker"):
        load_trained(path, gnn_lock(Switch.WITHOUT_COMMUNITY, MODEL, SUBGRAPH, SHORT))


@pytest.mark.parametrize("switch", list(Switch))
def test_an_ablation_trains_on_switched_batches(
    world: Any, switch: Switch, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches, selection = _small(world), _selection(world)
    seen: list[bool] = []
    real = trainer_module.encode_batch

    def recording(data: Any, *args: Any) -> Any:
        seen.append(bool(data.community.abs().sum() > 0))
        return real(data, *args)

    monkeypatch.setattr(trainer_module, "encode_batch", recording)
    trained = _train(batches, selection, replace(SHORT, max_epochs=1), switch=switch)
    assert trained.ranker == ranker_name(switch) == f"gnn_{switch.value}"
    assert trained.model.config.community == switch.community
    assert model_config(MODEL, switch).community == switch.community
    # Every batch read in training and selection carries the community block only when the
    # switch keeps it.
    assert seen and set(seen) == {switch.community}


def test_a_rotation_trains_every_seed_and_ablation_and_keeps_epochs_of_one(
    world: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches, selection = _small(world), _selection(world)
    settings = replace(SHORT, max_epochs=1, seeds=(0, 1))
    priors = [PRIOR, Prior(3.0, 0.09)]
    rising = iter(np.linspace(0.1, 0.9, 64))

    def scored(*args: Any) -> Found:
        return Found(("a",), np.ones(1), np.full((1, 2), next(rising)))

    monkeypatch.setattr(trainer_module, "selection_found", scored)
    found = train_rotation(
        batches,
        selection,
        priors,
        tmp_path,
        model=MODEL,
        settings=settings,
        subgraph=SUBGRAPH,
        keep_epochs_of_seed=0,
    )
    assert list(found) == [GNN, *(ranker_name(switch) for switch in Switch)]
    for runs in found.values():
        assert [run.seed for run in runs] == [0, 1]
    # The second prior scores higher, so each ablation trains at it alone.
    for seed in (0, 1):
        assert found[GNN][seed].prior == priors[1]
        for switch in Switch:
            ablation = found[ranker_name(switch)][seed]
            assert ablation.prior == priors[1] and len(ablation.runs) == 1
            assert model_file(tmp_path, ablation.ranker, seed).is_file()
        assert model_file(tmp_path, GNN, seed).is_file()
    assert sorted(path.name for path in (tmp_path / "epochs").iterdir()) == [
        "gnn_seed0_prior1_epoch1.pt",
        "gnn_seed0_prior3_epoch1.pt",
    ]


def test_training_runs_no_query(
    world: Any, planted: list[Batch], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training and selection read the cache alone: no executor of any kind runs a query, the
    evaluation executor least of all."""
    calls: list[str] = []

    def spy(self: object, name: str, *args: object, **keywords: object) -> list[Any]:
        calls.append(f"{type(self).__name__}.{name}")
        raise AssertionError("training ran a query")

    for kind in (
        executor_module.TigerGraphExecutor,
        executor_module.AllowedQueries,
        executor_module.EvaluationExecutor,
        executor_module.FetchExecutor,
    ):
        monkeypatch.setattr(kind, "run", spy)
    _train(_small(world), _selection(world), replace(SHORT, max_epochs=1))
    assert calls == []


def test_the_attention_is_deterministic_on_the_cpu() -> None:
    """The check passes, restores torch's settings and leaves its generator as it was."""
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    check_determinism(torch.device("cpu"))
    assert torch.equal(torch.rand(3), expected)
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_attention_is_deterministic_on_cuda() -> None:
    check_determinism(torch.device("cuda"))


def test_training_refuses_batches_without_labels_and_empty_inputs(world: Any) -> None:
    selection = _selection(world)
    scoring = next(iter(selection.batches))
    with pytest.raises(ValueError, match="no labels"):
        _train([scoring], selection)
    with pytest.raises(ValueError, match="at least one batch"):
        _train([], selection)
    with pytest.raises(ValueError, match="at least one prior"):
        _train(_small(world), selection, priors=())
