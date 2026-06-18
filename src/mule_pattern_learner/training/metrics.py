from collections.abc import Sequence
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


# ── SYNTHETIC-ONLY: multi-k recall + ceiling-normalized recall ──
# Extends evaluate_hidden's single-k report to several review-queue depths and
# reports, alongside each raw recall, the FRACTION OF THE ACHIEVABLE CEILING it
# captures. The ceiling exists because recall@k is capped at k / n_hidden when
# k < n_hidden: with k slots and more than k hidden mules, a perfect ranker
# still cannot place all hidden mules in the top k. The normalized number
# divides observed recall by that cap, so 1.0 means "ranked as many hidden
# mules into the top k as is mathematically possible," NOT "found every mule."
# It is a diagnostic that separates model ranking quality from the keyhole
# effect of a small k; the production metric remains recall at the single k an
# analyst can actually review.

_DEFAULT_KS: tuple[int, ...] = (100, 250, 500, 1000)


def recall_ceiling_at_k(n_hidden: int, k: int) -> float:
    """Maximum achievable hidden-recall at cutoff k.

    A perfect ranker puts every hidden mule above every negative, so it captures
    min(k, n_hidden) of them; dividing by n_hidden gives the ceiling. Equals
    min(1.0, k / n_hidden). Returns 0.0 when there are no hidden mules (recall is
    undefined) and 0.0 for non-positive k.
    """
    if n_hidden <= 0 or k <= 0:
        return 0.0
    return min(1.0, k / float(n_hidden))


@dataclass(frozen=True, slots=True)
class RecallAtK:
    """One (k, recall, ceiling, normalized) row of the multi-k report.

    recall:
        fraction of ALL hidden mules ranked within the top k (the same quantity
        evaluate_hidden reports at its single k).
    ceiling:
        min(1.0, k / num_hidden_positives) -- the most recall any ranker can
        achieve at this k. Below n_hidden the top-k simply has too few slots.
    normalized_recall:
        recall / ceiling, clamped to [0, 1]; the share of the achievable maximum
        actually captured. Read as ranking quality at this depth, not as
        coverage of the true mule population.
    hidden_in_top_k:
        raw count of hidden mules in the top k (recall * num_hidden_positives),
        surfaced so the report can show "37 of 169 hidden mules in the top 250".
    """

    k: int
    recall: float
    ceiling: float
    normalized_recall: float
    hidden_in_top_k: int


@dataclass(frozen=True, slots=True)
class HiddenScoresMultiK:
    """evaluate_hidden, reported across several review-queue depths.

    rows holds one RecallAtK per requested k (ascending). primary_k names the k
    treated as the production / headline number (the depth an analyst can review)
    so a caller can highlight it; the others are diagnostic context. The TRUE-label
    ranking metrics (average_precision_true, roc_auc_true) are k-independent and
    are reported once, identical to evaluate_hidden's.
    """

    rows: tuple[RecallAtK, ...]
    primary_k: int
    average_precision_true: float
    roc_auc_true: float
    num_hidden_positives: int
    num_true_positives: int

    def row_for(self, k: int) -> RecallAtK:
        for row in self.rows:
            if row.k == k:
                return row
        raise KeyError(f"no RecallAtK row for k={k}; have {[r.k for r in self.rows]}")

    @property
    def primary(self) -> RecallAtK:
        return self.row_for(self.primary_k)


def evaluate_hidden_multi_k(
    scores: NDArray[np.float64],
    true_label: NDArray[np.int64],
    bucket: NDArray[np.int64],
    ks: Sequence[int] = _DEFAULT_KS,
    primary_k: int = 100,
) -> HiddenScoresMultiK:
    """Multi-k generalization report for the PU experiment (synthetic only).

    Computes hidden-recall at each k in ks, plus the ceiling and the
    ceiling-normalized recall at that k, and reports the k-independent TRUE-label
    AP / ROC-AUC once. Sorting the scores once and reusing the order keeps this
    O(n log n + sum_k k) rather than re-sorting per k.

    scores:     model output per account (higher = more mule-like); shape [M].
    true_label: 1 if the account is truly a mule, else 0; shape [M].
    bucket:     REVEALED_POS / HIDDEN_POS / UNLABELED_NEG per account; shape [M].
    ks:         review-queue depths to report (deduped, sorted ascending). A k
                larger than M is kept but its top-k is the whole array, so recall
                there is 1.0 and the ceiling is 1.0.
    primary_k:  the k flagged as the headline/production number. Must be one of
                the (deduplicated) ks, so HiddenScoresMultiK.primary is well
                defined; a primary_k absent from ks is a caller error.

    SYNTHETIC ONLY: requires the answer key (true_label, bucket) that does not
    exist in production. Never call inside the training loop.
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

    # dedupe + sort so rows are stable and a repeated k is not double-reported
    sorted_ks = tuple(sorted({int(k) for k in ks}))
    if not sorted_ks:
        raise ValueError("ks must contain at least one cutoff.")
    if not all(k > 0 for k in sorted_ks):
        raise ValueError(f"all ks must be positive; got {sorted_ks}.")
    if primary_k not in sorted_ks:
        raise ValueError(
            f"primary_k ({primary_k}) must be one of ks ({sorted_ks}); it is the "
            + "headline depth and must have a computed row."
        )

    is_hidden: NDArray[np.bool_] = np.equal(bucket, int(Bucket.HIDDEN_POS))
    n_hidden = _count(is_hidden)

    # sort once (descending score); slice prefixes for each k
    order = np.argsort(-scores)
    hidden_sorted = is_hidden[order]
    # cumulative count of hidden mules down the ranking; cumulative[t-1] is the
    # number of hidden mules within the top t
    cumulative_hidden = np.cumsum(hidden_sorted.astype(np.int64))

    rows: list[RecallAtK] = []
    for k in sorted_ks:
        top = min(k, n_total)
        hidden_in_top = _int(cast("np.intp", cumulative_hidden[top - 1])) if top > 0 else 0
        recall = hidden_in_top / float(n_hidden) if n_hidden > 0 else 0.0
        ceiling = recall_ceiling_at_k(n_hidden, k)
        normalized = min(1.0, recall / ceiling) if ceiling > 0.0 else 0.0
        rows.append(
            RecallAtK(
                k=k,
                recall=recall,
                ceiling=ceiling,
                normalized_recall=normalized,
                hidden_in_top_k=hidden_in_top,
            )
        )

    return HiddenScoresMultiK(
        rows=tuple(rows),
        primary_k=primary_k,
        average_precision_true=float(average_precision_score(true_label, scores)),
        roc_auc_true=float(roc_auc_score(true_label, scores)),
        num_hidden_positives=n_hidden,
        num_true_positives=n_pos,
    )


def format_hidden_multi_k(report: HiddenScoresMultiK) -> str:
    """Render a HiddenScoresMultiK as an aligned text block for logs/console.

    Mirrors the existing evaluate_hidden printout style: a small header with the
    k-independent figures, then one line per k. The normalized column is labelled
    'recall/ceil' and the header states explicitly that 1.0 means 'best possible
    at this k', so 0.55 is never misread as 'found 55% of all mules'.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("HIDDEN-MULE GENERALIZATION (multi-k)")
    lines.append("=" * 60)
    lines.append(f"  true mules             : {report.num_true_positives}")
    lines.append(f"  hidden mules           : {report.num_hidden_positives}")
    lines.append(f"  AP (vs true labels)    : {report.average_precision_true:.4f}")
    lines.append(f"  AUC (vs true labels)   : {report.roc_auc_true:.4f}")
    lines.append("")
    lines.append("  recall/ceil = share of the BEST POSSIBLE recall at that k")
    lines.append("  (1.0 = every hidden mule that could fit in the top-k is there,")
    lines.append("   NOT that every mule in the bank was found)")
    lines.append("")
    header = (
        f"  {'k':>6}  {'recall':>8}  {'ceiling':>8}  {'recall/ceil':>12}  {'hidden_in_top_k':>16}"
    )
    lines.append(header)
    for row in report.rows:
        star = "  <- primary" if row.k == report.primary_k else ""
        lines.append(
            f"  {row.k:>6}  {row.recall:>8.4f}  {row.ceiling:>8.4f}  "
            + f"{row.normalized_recall:>12.4f}  "
            + f"{row.hidden_in_top_k:>6} / {report.num_hidden_positives:<7}{star}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# CAPACITY-SWEEP (gains / lift) EVALUATION
# ══════════════════════════════════════════════════════════════════════════
# Operating-point metrics reported across a RANGE of review capacities rather
# than one arbitrary k. Each capacity is an alert rate (fraction of accounts an
# analyst team reviews), because in real AML programs the cutoff is set by
# staffing, not by a statistic -- and no single rate is "correct" for a summary report
# on synthetic data, so we report the whole curve and let each viewer read off
# their own column.
#
# Real-world anchors (for sizing the default sweep and the reference line):
#   * Production bank alert rates are tiny: a multi-bank survey saw ~0.04% of
#     transactions alerted. Account-level review budgets are larger but still
#     well under a few percent.
#   * Genuine-suspicion base rates run ~1-5%, sometimes <1%.
#   * A skilled human investigator's referrals convert to a SAR ~40-50% of the
#     time -- the realistic precision ceiling to benchmark against.
#
# WHICH METRICS, AND WHY THESE:
#   recall    -- the ONLY operating-point metric provably estimable on PU data
#                (under the standard PU assumption precision is biased, so recall
#                is the unattackable detection number).
#   precision -- reported because THIS dataset is synthetic and carries a true-
#                label answer key; on real PU data it would be biased downward
#                (correctly-flagged hidden positives counted as false positives).
#                Always present it flagged as synthetic-only.
#   lift      -- precision divided by the population base rate: "Nx better than
#                random at this depth". Imbalance-native and intuitive.
#   alert_rate/k -- the operating context that ties everything to staffing and
#                kills the "arbitrary k" objection.
#
# All four are computed against TRUE labels here (synthetic answer key). On real
# data you would compute recall/lift against revealed positives and treat them
# as PU estimates; precision would not be trustworthy. This module does not hide
# that -- the formatter prints the caveat.

# Default capacity sweep: fractions of the scored population to review. Brackets
# the realistic range from "true bank alert rate" (~0.05%) up to a "generous
# review budget" (2%). Override per audience.
_DEFAULT_ALERT_RATES: tuple[float, ...] = (0.0005, 0.001, 0.005, 0.01, 0.02)

# Reference precision a strong human analyst achieves (SAR conversion ~40-50%).
# Drawn on the precision curve as a benchmark line; not a model output.
_HUMAN_ANALYST_PRECISION: float = 0.45


@dataclass(frozen=True, slots=True)
class GainsRow:
    """One operating point in a capacity sweep, against TRUE labels.

    alert_rate:
        fraction of the population reviewed at this cutoff (the x-axis).
    k:
        number of accounts reviewed = round(alert_rate * n_total); the top-k by
        score. Reported alongside alert_rate so the audience sees both the
        percentage and the concrete headcount.
    recall:
        of all true positives, the fraction inside the top k. PU-estimable.
    precision:
        of the top k, the fraction that are true positives. SYNTHETIC-ONLY
        trustworthy (biased downward on real PU data).
    lift:
        precision / base_rate. 1.0 = no better than random; higher is better.
    true_positives_in_k:
        raw count of true positives in the top k (recall * num_true_positives).
    """

    alert_rate: float
    k: int
    recall: float
    precision: float
    lift: float
    true_positives_in_k: int


@dataclass(frozen=True, slots=True)
class SummaryReport:
    """The full fool-proof metric panel for the synthetic PU experiment.

    Two threshold-free headline numbers (computed against true labels):
        pr_auc  -- Average Precision; the PRIMARY ranking metric for imbalanced
                   fraud/AML. Lead with this.
        roc_auc -- shown only with a caveat; inflated under heavy class imbalance
                   and deliberately NOT the headline.
    Plus the capacity sweep (gains rows) and the context needed to read it:
        base_rate            -- num_true_positives / num_accounts; the random
                                baseline lift is measured against.
        human_analyst_precision -- reference SAR-conversion ceiling (~0.45) to
                                draw on the precision curve. A constant, not a
                                model output.
    Nothing here is estimable in production unchanged: precision and lift use the
    synthetic answer key. The formatter states this explicitly.
    """

    pr_auc: float
    roc_auc: float
    rows: tuple[GainsRow, ...]
    base_rate: float
    num_accounts: int
    num_true_positives: int
    human_analyst_precision: float


def gains_table(
    scores: NDArray[np.float64],
    true_label: NDArray[np.int64],
    alert_rates: Sequence[float] = _DEFAULT_ALERT_RATES,
) -> tuple[GainsRow, ...]:
    """Compute recall / precision / lift at each alert-rate cutoff (true labels).

    Sorts scores once and reuses cumulative true-positive counts down the
    ranking, so the whole sweep is O(n log n) regardless of how many rates are
    requested. k for each rate is round(rate * n_total), clamped to [1, n_total].

    scores:      model output per account (higher = more mule-like); shape [M].
    true_label:  1 if truly a mule else 0; shape [M].
    alert_rates: fractions of the population to review (deduped, sorted asc).
                 Each must be in (0, 1].

    Returns one GainsRow per distinct alert_rate. Raises ValueError on shape
    mismatch, out-of-range rate, or a single-class label vector (lift undefined
    when base_rate is 0 or 1).
    """
    if scores.shape != true_label.shape:
        raise ValueError("scores and true_label must have the same shape.")
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D.")

    n_total = _length(true_label)
    n_pos = _count(true_label)
    if n_pos == 0 or n_pos == n_total:
        raise ValueError(
            f"true_label must contain both classes; got {n_pos} positives of {n_total}. "
            + "Lift / precision are undefined when the base rate is 0 or 1."
        )

    rates = tuple(sorted({float(r) for r in alert_rates}))
    if not rates:
        raise ValueError("alert_rates must contain at least one rate.")
    if not all(0.0 < r <= 1.0 for r in rates):
        raise ValueError(f"every alert_rate must be in (0, 1]; got {rates}.")

    base_rate = n_pos / float(n_total)

    order = np.argsort(-scores)
    pos_sorted = true_label[order].astype(np.int64)
    cumulative_pos = np.cumsum(pos_sorted)  # cumulative_pos[t-1] = TPs in top t

    rows: list[GainsRow] = []
    for rate in rates:
        k = int(round(rate * n_total))
        k = max(1, min(k, n_total))
        tp_in_k = _int(cast("np.intp", cumulative_pos[k - 1]))
        recall = tp_in_k / float(n_pos)
        precision = tp_in_k / float(k)
        lift = precision / base_rate  # base_rate > 0 guaranteed above
        rows.append(
            GainsRow(
                alert_rate=rate,
                k=k,
                recall=recall,
                precision=precision,
                lift=lift,
                true_positives_in_k=tp_in_k,
            )
        )
    return tuple(rows)


def evaluate_summary(
    scores: NDArray[np.float64],
    true_label: NDArray[np.int64],
    alert_rates: Sequence[float] = _DEFAULT_ALERT_RATES,
    human_analyst_precision: float = _HUMAN_ANALYST_PRECISION,
) -> SummaryReport:
    """Assemble the full fool-proof panel: PR-AUC + caveated ROC-AUC + sweep.

    scores / true_label:        shape [M]; the synthetic answer key.
    alert_rates:                capacity sweep (see gains_table).
    human_analyst_precision:    reference SAR-conversion ceiling for the chart.

    SYNTHETIC ONLY for the precision/lift parts (true-label answer key). PR-AUC
    and recall are the production-meaningful figures; ROC-AUC is reported only to
    be visibly de-emphasised.
    """
    if scores.shape != true_label.shape:
        raise ValueError("scores and true_label must have the same shape.")
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D.")

    n_total = _length(true_label)
    n_pos = _count(true_label)
    if n_pos == 0 or n_pos == n_total:
        raise ValueError(
            f"true_label must contain both classes; got {n_pos} positives of {n_total}."
        )

    rows = gains_table(scores, true_label, alert_rates)
    return SummaryReport(
        pr_auc=float(average_precision_score(true_label, scores)),
        roc_auc=float(roc_auc_score(true_label, scores)),
        rows=rows,
        base_rate=n_pos / float(n_total),
        num_accounts=n_total,
        num_true_positives=n_pos,
        human_analyst_precision=human_analyst_precision,
    )


def format_summary_report(report: SummaryReport) -> str:
    """Render a SummaryReport as an aligned console block, caveats included."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("MULE-DETECTION EVALUATION  (synthetic; true-label answer key)")
    lines.append("=" * 72)
    lines.append(f"  accounts scored        : {report.num_accounts}")
    lines.append(f"  true mules             : {report.num_true_positives}")
    lines.append(
        f"  base rate              : {report.base_rate:.5f}  "
        + f"({report.base_rate * 100:.3f}% of accounts)"
    )
    lines.append("")
    lines.append("  PRIMARY (threshold-free, robust to imbalance):")
    lines.append(f"    PR-AUC / Avg Precision : {report.pr_auc:.4f}   <- headline")
    lines.append(
        f"    ROC-AUC                : {report.roc_auc:.4f}   "
        + "(de-emphasised: inflated under imbalance; not the headline)"
    )
    lines.append("")
    lines.append("  OPERATING POINTS across review capacity:")
    lines.append("    recall is PU-estimable; PRECISION & LIFT are trustworthy")
    lines.append("    ONLY because this is synthetic (biased on real PU data).")
    lines.append(f"    human-analyst precision reference ~ {report.human_analyst_precision:.2f}")
    lines.append("")
    header = (
        f"  {'alert_rate':>10}  {'k':>7}  {'recall':>8}  "
        + f"{'precision':>10}  {'lift':>8}  {'TP_in_k':>9}"
    )
    lines.append(header)
    for row in report.rows:
        beats = " *" if row.precision >= report.human_analyst_precision else "  "
        lines.append(
            f"  {row.alert_rate * 100:>9.3f}%  {row.k:>7}  {row.recall:>8.4f}  "
            + f"{row.precision:>10.4f}  {row.lift:>7.1f}x  "
            + f"{row.true_positives_in_k:>4}/{report.num_true_positives:<4}{beats}"
        )
    lines.append("")
    lines.append("  ( * = precision at this depth meets/beats the human-analyst")
    lines.append("    reference; the slide-worthy 'matches expert triage' point )")
    lines.append("=" * 72)
    return "\n".join(lines)


# ── Charting (optional; requires matplotlib) ──
# Kept in the module so the figure is reproducible from a SummaryReport rather
# than ad-hoc per script. matplotlib is imported INSIDE the function so the rest
# of metrics.py has no hard dependency on it; environments without matplotlib
# can still compute every number above.


def save_summary_chart(
    report: "SummaryReport",
    out_path: str,
    title: str = "Mule detection - gains across review capacity",
    note: str | None = None,
) -> str:
    """Render a two-panel gains/lift chart from a SummaryReport and save to disk.

    Left panel: recall and precision vs. alert rate, with the human-analyst
    precision reference line. Right panel: lift vs. alert rate, with the random
    (1x) baseline. Both x-axes are the alert rate in percent, so the figure is
    read in operational (staffing) terms, not arbitrary k.

    report:   a SummaryReport (use one built from a DENSE alert_rates sweep, e.g.
              numpy.linspace, for smooth curves -- the default 5-point sweep
              produces a coarse plot).
    out_path: file path to write (extension sets the format, e.g. .png/.pdf).
    title:    figure suptitle.
    note:     optional caption drawn under the title; use it to record the data
              provenance (e.g. the run / checkpoint the scores came from). If the
              scores are not real model output, say so here.

    Returns out_path. Raises ImportError (with guidance) if matplotlib is absent.
    Does not fabricate data: every point comes from report.rows.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe; no display needed
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "save_summary_chart needs matplotlib; install it or compute the "
            "numbers with format_summary_report instead (no plotting required)."
        ) from exc

    if not report.rows:
        raise ValueError("report has no rows to plot.")

    ar = [row.alert_rate * 100.0 for row in report.rows]
    rec = [row.recall for row in report.rows]
    pre = [row.precision for row in report.rows]
    lift = [row.lift for row in report.rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    suptitle = title
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    if note:
        # small caption line under the suptitle for provenance / caveats
        fig.text(0.5, 0.93, note, ha="center", va="top", fontsize=8, color="#555555")

    ax = axes[0]
    ax.plot(ar, rec, color="#1b5e9c", lw=2.2, label="Recall (detection rate) - PU-estimable")
    ax.plot(ar, pre, color="#c0392b", lw=2.2, label="Precision - synthetic-only")
    ax.axhline(
        report.human_analyst_precision,
        color="#7f8c8d",
        ls="--",
        lw=1.4,
        label=f"Human-analyst precision ~{report.human_analyst_precision:.2f}",
    )
    ax.set_xlabel("Alert rate  (% of accounts reviewed)")
    ax.set_ylabel("Recall / Precision")
    ax.set_title("Recall & precision vs. capacity")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(ar, lift, color="#27772f", lw=2.2)
    ax.axhline(1.0, color="#7f8c8d", ls="--", lw=1.2, label="Random (1x)")
    ax.set_xlabel("Alert rate  (% of accounts reviewed)")
    ax.set_ylabel("Lift  (x better than random)")
    ax.set_title("Lift vs. capacity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92 if note else 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
