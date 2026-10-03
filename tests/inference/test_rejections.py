"""Rejected roots: their counts, the limit that fails a split and a training epoch."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.inference.rejections import (
    SourceRejections,
    TrainingRejections,
    check_split_rejections,
    exceeds_rejection_limit,
    rejection_counts,
    rejection_summary,
)
from mule_pattern_learner.testing.builders import unit_config
from mule_pattern_learner.testing.fake_graph import FakeSource

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
    with pytest.raises(ValueError, match="1 observed positive among them"):
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
    with pytest.raises(ValueError, match=r"\(1 observed positive;"):
        resumed.count(1, np.array([0, 1, 0, 0]), step, 100, lambda: STATUSES)


def test_a_summary_since_a_copy_of_the_counters_leaves_out_what_came_before() -> None:
    source = FakeSource(unit_config(), reject=frozenset({"ghost", "gone"}))
    ghost, gone = ContextKey("Account", "ghost", 1, 1), ContextKey("Account", "gone", 1, 1)
    source.fetch([ghost, gone], hop=2)
    before = SourceRejections.of(source)
    source.fetch([ghost])
    source.fetch([gone], hop=2)
    # The copy keeps its counts while the source's grow.
    assert before.by_status == Counter({"missing_entity": 2})
    assert before.by_hop == {2: Counter({"missing_entity": 2})}
    later = rejection_summary(source, 1, Counter({"rejected_children": 1}), since=before)
    assert later["rejected_roots_by_status"] == {"missing_entity": 1}
    assert later["rejected_children_by_status"] == {"missing_entity": 1}
    assert later["rejection_events_by_status"] == {"missing_entity": 2}
    # Without since, the summary is the source's whole life.
    whole = rejection_summary(source, 1, Counter())
    assert whole["rejected_children_by_status"] == {"missing_entity": 3}
    assert whole["rejection_events_by_status"] == {"missing_entity": 4}
    # A hop with nothing new reports no statuses.
    assert rejection_summary(source, 0, Counter(), since=SourceRejections.of(source)) == {
        "rejected": 0,
        "rejected_roots_by_status": {},
        "rejected_children": 0,
        "rejected_children_by_status": {},
        "stub_children": 0,
        "rejection_events_by_status": {},
    }
