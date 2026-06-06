import numpy as np
import pytest

from mule_pattern_learner.training.metrics import Bucket, evaluate


def _bucket_from(true_label: list[int], hidden_idx: set[int]) -> list[int]:
    out: list[int] = []
    for i, y in enumerate(true_label):
        if y == 1 and i in hidden_idx:
            out.append(int(Bucket.HIDDEN_POS))
        elif y == 1:
            out.append(int(Bucket.REVEALED_POS))
        else:
            out.append(int(Bucket.UNLABELED_NEG))
    return out


def test_perfect_ranking_gives_ap_one() -> None:
    # scores perfectly separate positives (high) from negatives (low)
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    true = np.array([1, 1, 0, 0], dtype=np.int64)
    bucket = np.array(_bucket_from([1, 1, 0, 0], set()), dtype=np.int64)
    m = evaluate(scores, true, bucket, k=2)
    assert m.average_precision == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.precision_at_k == pytest.approx(1.0)  # top-2 are both positive


def test_inverted_ranking_low_ap() -> None:
    # scores rank negatives above positives -> AUC ~ 0, AP low
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    true = np.array([1, 1, 0, 0], dtype=np.int64)
    bucket = np.array(_bucket_from([1, 1, 0, 0], set()), dtype=np.int64)
    m = evaluate(scores, true, bucket, k=2)
    assert m.roc_auc == pytest.approx(0.0)
    assert m.precision_at_k == pytest.approx(0.0)  # top-2 are both negative


def test_precision_at_k_partial() -> None:
    # top-2 by score: idx3 (0.9, pos) and idx2 (0.7, neg) -> 1 of 2 correct
    scores = np.array([0.1, 0.2, 0.7, 0.9], dtype=np.float64)
    true = np.array([0, 0, 0, 1], dtype=np.int64)
    bucket = np.array(_bucket_from([0, 0, 0, 1], set()), dtype=np.int64)
    m = evaluate(scores, true, bucket, k=2)
    assert m.precision_at_k == pytest.approx(0.5)


def test_hidden_recall_detects_generalization() -> None:
    # 4 accounts: two positives, idx0 hidden. scores rank idx0 high (top-2)
    # -> hidden mule is found -> hidden_recall = 1.0 (1 of 1 hidden in top-2)
    scores = np.array([0.95, 0.3, 0.9, 0.1], dtype=np.float64)
    true = np.array([1, 0, 1, 0], dtype=np.int64)
    bucket = np.array(_bucket_from([1, 0, 1, 0], {0}), dtype=np.int64)
    m = evaluate(scores, true, bucket, k=2)
    assert m.num_hidden_positives == 1
    assert m.hidden_recall_at_k == pytest.approx(1.0)

    # now rank the hidden mule LOW (not in top-2) -> hidden_recall = 0.0
    scores_bad = np.array([0.05, 0.3, 0.9, 0.8], dtype=np.float64)
    m2 = evaluate(scores_bad, true, bucket, k=2)
    assert m2.hidden_recall_at_k == pytest.approx(0.0)


def test_single_class_raises() -> None:
    scores = np.array([0.5, 0.6], dtype=np.float64)
    true = np.array([0, 0], dtype=np.int64)  # no positives
    bucket = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        _ = evaluate(scores, true, bucket, k=1)


def test_shape_mismatch_raises() -> None:
    scores = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    true = np.array([1, 0], dtype=np.int64)
    bucket = np.array([1, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        _ = evaluate(scores, true, bucket, k=1)
