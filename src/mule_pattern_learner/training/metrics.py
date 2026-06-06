from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


def _int(value: np.intp | int) -> int:
    # numpy reductions / shapes are typed Any under strict checking; a cast at
    # the call site (cast(np.intp, ...)) erases the Any, and this funnels the
    # concrete numpy integer to a Python int in one place.
    return int(value)


def _count(array: NDArray[np.bool_] | NDArray[np.int64]) -> int:
    # number of truthy / nonzero entries, laundered to a plain int
    return _int(cast("np.intp", array.sum()))


def _length(array: NDArray[np.float64] | NDArray[np.int64]) -> int:
    return _int(cast("int", array.shape[0]))


def _precision(scores: NDArray[np.float64], labels: NDArray[np.int64], k: int) -> float:
    # fraction of the top-k highest-scored accounts that are positive
    if k <= 0:
        return 0.0
    order = np.argsort(-scores)
    top = order[:k]
    hits = _count(labels[top])
    denom = min(k, _length(labels))
    return hits / float(denom)


# ── PRODUCTION metrics: computed against pu_label only ──
# These use only the labels the model also trains on (pu_label: 1 = known
# positive, 0 = unlabeled), so they exist unchanged on real data. This is what
# drives early stopping and what you would report in production.


@dataclass(frozen=True, slots=True)
class ValScores:
    """
    Validation ranking quality, computed against pu_label (production-valid).

    roc_auc is the model-selection number and the early-stop signal. In PU data
    the unlabeled validation points are treated as (corrupted) negatives, and
    ROC-AUC computed that way is the Proxy AUC (PAUC): ranking under the true
    labels would order two classifiers the same way PAUC does, because AUC is
    provably robust to this label corruption -- a classifier with higher PAUC
    has higher true AUC. (Sakai et al.; PU model-selection literature.) That is
    why early stopping uses roc_auc, NOT average_precision: AP is sensitive to
    the unlabeled-as-negative assumption (the unlabeled set still contains real
    positives, which depress precision unpredictably), so it is reported as a
    diagnostic only, not optimized against. precision_at_k mirrors an
    investigation queue (of the top-k flagged, the fraction that are labelled
    positives) and is likewise diagnostic.

    Every field here derives from pu_label, so nothing in this object depends on
    knowing the true (hidden) mules -- it is identical in production.
    """

    average_precision: float
    roc_auc: float
    precision_at_k: float
    k: int
    num_evaluated: int
    num_labeled_positives: int


def evaluate_ranking(scores: NDArray[np.float64], pu_label: NDArray[np.int64], k: int) -> ValScores:
    """
    Score model output against pu_label (the labels the model also trains on).

    scores:   model output per account (higher = more mule-like); shape [M].
    pu_label: 1 if the account is a known/labelled positive, else 0; shape [M].
    k:        cutoff for precision_at_k.

    average_precision / roc_auc are computed against pu_label; precision_at_k
    uses the score ranking. roc_auc is the Proxy AUC used for model selection
    (see ValScores); average_precision is diagnostic. Raises ValueError on shape
    mismatch or if pu_label has only one class (ranking metrics undefined).
    Production-valid: uses no ground-truth / hidden-mule information.
    """
    if scores.shape != pu_label.shape:
        raise ValueError("scores and pu_label must have the same shape.")
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D.")

    n_total = _length(pu_label)
    n_pos = _count(pu_label)
    if n_pos == 0 or n_pos == n_total:
        raise ValueError(
            f"pu_label must contain both classes; got {n_pos} positives of {n_total}. "
            + "PU model selection (Proxy AUC) needs at least one revealed positive in "
            + "the evaluated split; train.py guards this before training, so reaching "
            + "here means the split lost its positives."
        )

    ap = float(average_precision_score(pu_label, scores))
    auc = float(roc_auc_score(pu_label, scores))
    p_at_k = _precision(scores, pu_label, k)

    return ValScores(
        average_precision=ap,
        roc_auc=auc,
        precision_at_k=p_at_k,
        k=k,
        num_evaluated=n_total,
        num_labeled_positives=n_pos,
    )


# ── SYNTHETIC-ONLY evaluation: requires the answer key (true_label, bucket) ──
# This is NOT part of training. It exists to measure, on synthetic data where
# ground truth is known, whether the model generalizes to mules it was never
# told about. On real data there is no answer key, so this is never run and
# these symbols are never imported by the training loop.


class Bucket(IntEnum):
    """Per-account evaluation bucket (synthetic answer key; from the masking step)."""

    UNLABELED_NEG = 0
    REVEALED_POS = 1
    HIDDEN_POS = 2


@dataclass(frozen=True, slots=True)
class HiddenScores:
    """
    Generalization measure for the PU experiment (synthetic data only).

    hidden_recall_at_k is the key number: among the HIDDEN mules (truly mules
    but never labelled to the model), the fraction the model ranks in its top-k.
    A model that merely memorized the few revealed positives scores low here.
    average_precision_true / roc_auc_true recompute ranking quality against the
    TRUE labels (not pu_label), so they reveal how well the model found the
    unlabelled mules too. All of this depends on the synthetic answer key.
    """

    hidden_recall_at_k: float
    average_precision_true: float
    roc_auc_true: float
    k: int
    num_hidden_positives: int
    num_true_positives: int


def _hidden_recall(scores: NDArray[np.float64], is_hidden: NDArray[np.bool_], k: int) -> float:
    # of all HIDDEN positives, the fraction ranked within the top-k overall
    n_hidden = _count(is_hidden)
    if k <= 0 or n_hidden == 0:
        return 0.0
    order = np.argsort(-scores)
    top = order[:k]
    hidden_in_top = _count(is_hidden[top])
    return hidden_in_top / float(n_hidden)


def evaluate_hidden(
    scores: NDArray[np.float64],
    true_label: NDArray[np.int64],
    bucket: NDArray[np.int64],
    k: int,
) -> HiddenScores:
    """
    Score generalization to hidden mules, using the synthetic answer key.

    scores:     model output per account (higher = more mule-like); shape [M].
    true_label: 1 if the account is truly a mule, else 0; shape [M].
    bucket:     REVEALED_POS / HIDDEN_POS / UNLABELED_NEG per account; shape [M].
    k:          cutoff for hidden_recall_at_k.

    SYNTHETIC ONLY: requires ground truth that does not exist in production. Run
    this as a separate evaluation step, never inside the training loop.
    """
    if not (scores.shape == true_label.shape == bucket.shape):
        raise ValueError("scores, true_label, and bucket must have the same shape.")
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D.")

    n_total = _length(true_label)
    n_pos = _count(true_label)
    if n_pos == 0 or n_pos == n_total:
        raise ValueError(
            f"true_label must contain both classes; got {n_pos} positives of {n_total}."
        )

    is_hidden: NDArray[np.bool_] = np.equal(bucket, int(Bucket.HIDDEN_POS))

    return HiddenScores(
        hidden_recall_at_k=_hidden_recall(scores, is_hidden, k),
        average_precision_true=float(average_precision_score(true_label, scores)),
        roc_auc_true=float(roc_auc_score(true_label, scores)),
        k=k,
        num_hidden_positives=_count(is_hidden),
        num_true_positives=n_pos,
    )
