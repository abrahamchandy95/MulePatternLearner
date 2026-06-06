import pytest

from mule_pattern_learner.training.loop import EarlyStopper


def test_early_stopper_detects_improvement() -> None:
    s = EarlyStopper(patience=3, min_delta=0.01)
    assert s.update(0.50, 0) is True  # first score is always the best
    assert s.best == pytest.approx(0.50)
    assert s.update(0.55, 1) is True  # improved by more than min_delta
    assert s.best_epoch == 1
    assert s.update(0.553, 2) is False  # gain below min_delta -> no improvement


def test_early_stopper_stops_after_patience() -> None:
    s = EarlyStopper(patience=2, min_delta=0.0)
    _ = s.update(0.5, 0)  # best
    assert not s.should_stop
    _ = s.update(0.4, 1)  # bad epoch 1
    assert not s.should_stop
    _ = s.update(0.4, 2)  # bad epoch 2 -> reaches patience
    assert s.should_stop


def test_early_stopper_resets_on_new_best() -> None:
    s = EarlyStopper(patience=2, min_delta=0.0)
    _ = s.update(0.5, 0)
    _ = s.update(0.4, 1)  # bad 1
    assert not s.should_stop
    _ = s.update(0.6, 2)  # new best resets the counter
    assert s.best_epoch == 2
    _ = s.update(0.5, 3)  # bad 1 again (not yet at patience)
    assert not s.should_stop


def test_best_epoch_tracks_argmax() -> None:
    s = EarlyStopper(patience=10, min_delta=0.0)
    for epoch, score in enumerate([0.3, 0.7, 0.5, 0.9, 0.2]):
        _ = s.update(score, epoch)
    assert s.best_epoch == 3  # 0.9 occurs at epoch 3
    assert s.best == pytest.approx(0.9)
