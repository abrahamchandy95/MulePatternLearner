import numpy as np
import pytest

from mule_pattern_learner.training.metrics import (
    Bucket,
    evaluate_hidden,
    evaluate_hidden_multi_k,
    evaluate_ranking,
)


def _bucket_from(true_label: list[int], hidden_idx: set[int]) -> np.ndarray:
    out: list[int] = []
    for i, y in enumerate(true_label):
        if y == 1 and i in hidden_idx:
            out.append(int(Bucket.HIDDEN_POS))
        elif y == 1:
            out.append(int(Bucket.REVEALED_POS))
        else:
            out.append(int(Bucket.UNLABELED_NEG))
    return np.array(out, dtype=np.int64)


def test_perfect_ranking_gives_ap_one() -> None:
    # Scores perfectly separate labelled positives (high) from unlabeled (low).
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    pu_label = np.array([1, 1, 0, 0], dtype=np.int64)
    m = evaluate_ranking(scores, pu_label, k=2)
    assert m.average_precision == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.precision_at_k == pytest.approx(1.0)  # top-2 are both positive
    assert (m.num_evaluated, m.num_labeled_positives, m.k) == (4, 2, 2)


def test_inverted_ranking_low_ap() -> None:
    # Negatives rank above positives: AUC 0 and no positive in the top 2.
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    pu_label = np.array([1, 1, 0, 0], dtype=np.int64)
    m = evaluate_ranking(scores, pu_label, k=2)
    assert m.roc_auc == pytest.approx(0.0)
    assert m.precision_at_k == pytest.approx(0.0)


def test_precision_at_k_partial() -> None:
    # Top 2 by score: idx3 (0.9, positive) and idx2 (0.7, unlabeled), so 1 of 2.
    scores = np.array([0.1, 0.2, 0.7, 0.9], dtype=np.float64)
    pu_label = np.array([0, 0, 0, 1], dtype=np.int64)
    assert evaluate_ranking(scores, pu_label, k=2).precision_at_k == pytest.approx(0.5)


def test_hidden_recall_detects_generalization() -> None:
    # Two true positives, idx0 hidden. Ranking idx0 in the top 2 finds it.
    scores = np.array([0.95, 0.3, 0.9, 0.1], dtype=np.float64)
    true = np.array([1, 0, 1, 0], dtype=np.int64)
    bucket = _bucket_from([1, 0, 1, 0], {0})
    m = evaluate_hidden(scores, true, bucket, k=2)
    assert m.num_hidden_positives == 1 and m.num_true_positives == 2
    assert m.hidden_recall_at_k == pytest.approx(1.0)
    assert m.roc_auc_true == pytest.approx(1.0)

    # Ranking the hidden mule low (outside the top 2) misses it.
    scores_bad = np.array([0.05, 0.3, 0.9, 0.8], dtype=np.float64)
    assert evaluate_hidden(scores_bad, true, bucket, k=2).hidden_recall_at_k == pytest.approx(0.0)


def test_multi_k_recall_reports_the_achievable_ceiling() -> None:
    # Three hidden mules; with k=1 a perfect ranker reaches at most one third.
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2], dtype=np.float64)
    true = np.array([1, 1, 1, 0, 0], dtype=np.int64)
    bucket = _bucket_from([1, 1, 1, 0, 0], {0, 1, 2})
    report = evaluate_hidden_multi_k(scores, true, bucket, ks=(1, 3), primary_k=3)
    rows = {row.k: row for row in report.rows}
    assert rows[1].recall == pytest.approx(1 / 3)
    assert rows[1].ceiling == pytest.approx(1 / 3)
    assert rows[1].normalized_recall == pytest.approx(1.0)
    assert rows[3].recall == pytest.approx(1.0) and rows[3].hidden_in_top_k == 3


def test_single_class_raises() -> None:
    scores = np.array([0.5, 0.6], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.int64)  # no positives
    with pytest.raises(ValueError, match="both classes"):
        _ = evaluate_ranking(scores, labels, k=1)
    with pytest.raises(ValueError, match="both classes"):
        _ = evaluate_hidden(scores, labels, labels, k=1)


def test_shape_mismatch_raises() -> None:
    scores = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    labels = np.array([1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        _ = evaluate_ranking(scores, labels, k=1)
    with pytest.raises(ValueError, match="shape"):
        _ = evaluate_hidden(scores, labels, labels, k=1)
