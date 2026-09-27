"""Rejected roots: their counts, the limit that fails a split and a training epoch."""

from __future__ import annotations

import numpy as np
import pytest

from mule_pattern_learner.inference.rejections import (
    TrainingRejections,
    check_split_rejections,
    exceeds_rejection_limit,
    rejection_counts,
)

LABELS = np.array([1, 0, 0, 0, 1, 0])
STATUSES = {"missing_entity": 1}


def test_counts_split_rejected_roots_into_positives_and_unlabeled() -> None:
    accepted = np.array([True, False, True, True, False, True])
    assert rejection_counts(LABELS, accepted) == {
        "requested": 6,
        "rejected": 2,
        "positive": 1,
        "unlabeled": 1,
    }
    assert exceeds_rejection_limit(1, 0, 6, 0.2) is False
    assert exceeds_rejection_limit(2, 0, 6, 0.2) is True
    assert exceeds_rejection_limit(1, 1, 6, 1.0) is True


def test_a_split_fails_on_a_rejected_positive_the_limit_or_lost_classes() -> None:
    everything = np.ones(6, dtype=bool)
    check_split_rejections("test", LABELS, everything, 0.0, STATUSES)
    one_unlabeled = np.array([True, False, True, True, True, True])
    check_split_rejections("test", LABELS, one_unlabeled, 0.2, STATUSES)
    with pytest.raises(ValueError, match=r"test: .*1 of 6 roots \(above .*=0.0\); statuses"):
        check_split_rejections("test", LABELS, one_unlabeled, 0.0, STATUSES)
    positive = np.array([False, True, True, True, True, True])
    with pytest.raises(ValueError, match="1 observed positives among them"):
        check_split_rejections("test", LABELS, positive, 1.0, STATUSES)
    # Validation keeps both observed classes after its rejections.
    negatives = np.array([True, False, False, False, True, False])
    check_split_rejections("test", LABELS, negatives, 1.0, STATUSES)
    with pytest.raises(ValueError, match="no longer has both observed classes"):
        check_split_rejections("validation", LABELS, negatives, 1.0, STATUSES)


def test_training_rejections_count_the_epoch_against_the_limit_and_resume() -> None:
    rejections = TrainingRejections(limit=0.25)
    step = np.array([True, False, True, True])
    rejections.count(0, np.zeros(4, dtype=np.int64), step, 8, lambda: STATUSES)
    assert rejections.totals() == {"requested": 4, "rejected": 1, "positive": 0, "unlabeled": 1}
    saved = rejections.saved()
    with pytest.raises(ValueError, match=r"Epoch 1: .*rejected 4 of 8 training roots"):
        rejections.count(0, np.zeros(4, dtype=np.int64), ~step, 8, lambda: STATUSES)
    # A resumed run continues from the saved counts; a new epoch starts its own.
    resumed = TrainingRejections(limit=0.25)
    resumed.restore(saved)
    assert resumed.totals() == {"requested": 4, "rejected": 1, "positive": 0, "unlabeled": 1}
    resumed.start_epoch()
    resumed.count(1, np.zeros(4, dtype=np.int64), step, 8, lambda: STATUSES)
    assert resumed.totals()["rejected"] == 2 and resumed.epoch["rejected"] == 1
    with pytest.raises(ValueError, match="1 observed positives"):
        resumed.count(1, np.array([0, 1, 0, 0]), step, 100, lambda: STATUSES)
