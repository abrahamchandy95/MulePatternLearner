"""The score figures of an audit: the threshold, the score densities, revealed and hidden mules.

Each plot function draws on the Axes it is given and returns it; none reads or saves a
file (reporting.report does). Scores sit near 0 and 1, so the threshold and density
figures put them on a log10-odds axis, where 0 is a score of 0.5 and 5 a score of
0.99999.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
import numpy as np
from numpy.typing import NDArray

from ..metrics import precision_recall_curve
from .ranking import SplitScores, share_label, top_share_axis
from .style import HIDDEN, INK, MULE, NON_MULE, REVEALED, SURFACE, legend_below, measure, number

# The smallest score and distance from 1 that log10_odds tells apart: a float64 score
# rounds to 1 above a logit of about 37, which would otherwise have infinite odds.
SMALLEST = 1e-16


def log10_odds(score: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """log10(score / (1 - score)), within plus or minus 16."""
    p = np.asarray(score, dtype=np.float64)
    return np.log10(np.maximum(p, SMALLEST)) - np.log10(np.maximum(1 - p, SMALLEST))


def threshold_label(threshold: float) -> str:
    """The selected threshold in the axis' units, log10 odds, and as the score it is."""
    return (
        f"selected threshold: log10 odds {float(log10_odds(threshold)):.2f} (score {threshold:.6g})"
    )


def _threshold_line(ax: Axes, threshold: float, label: str) -> None:
    ax.axvline(float(log10_odds(threshold)), color=INK, linestyle="--", linewidth=1.0, label=label)


def plot_threshold_metrics(ax: Axes, scores: SplitScores, threshold: float) -> Axes:
    """Weighted precision, recall and F1 of flagging the scores at or above each threshold.

    The curves step at each block of tied scores (metrics.precision_recall_curve); the
    dashed line is the selected threshold, labelled with its recorded precision and recall.
    """
    recall, precision, thresholds = precision_recall_curve(scores.y, scores.score, scores.weight)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # Lowest threshold first; the value between two block scores is the higher block's.
    x = log10_odds(thresholds)[::-1]
    for index, (label, values) in enumerate(
        (("precision", precision), ("recall", recall), ("F1", f1))
    ):
        ax.plot(x, values[::-1], drawstyle="steps-pre", label=label, **measure(index))
    chosen = (
        f"{threshold_label(threshold)},\nprecision {number(scores.metrics.get('precision'))}, "
        f"recall {number(scores.metrics.get('recall'))}"
    )
    _threshold_line(ax, threshold, chosen)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold, log10 odds of the score (0 is a score of 0.5)")
    ax.set_ylabel("Weighted estimate")
    # Under the axes: the threshold's label is long, and the curves cross the middle.
    legend_below(ax, *ax.get_legend_handles_labels(), ncols=2)
    ax.set_title("Test audit: precision, recall and F1 at each threshold")
    return ax


def plot_score_distribution(ax: Axes, scores: SplitScores, threshold: float) -> Axes:
    """Weighted densities of the log10 odds of mules and non-mules, and the threshold.

    Each class is normalised on its own, so the few mules show beside the many
    non-mules; the weights make the non-mules' density the population's.
    """
    odds = log10_odds(scores.score)
    edges = np.histogram_bin_edges(odds, bins=48)
    classes = (
        ("non-mules", scores.y == 0, NON_MULE),
        ("mules", scores.y == 1, MULE),
    )
    for label, rows, colour in classes:
        if not rows.any():
            continue
        density, _ = np.histogram(odds[rows], bins=edges, weights=scores.weight[rows], density=True)
        sampled, population = int(rows.sum()), float(scores.weight[rows].sum())
        stands_for = "" if round(population) == sampled else f", standing for {population:,.0f}"
        ax.stairs(density, edges, fill=True, color=colour, alpha=0.2)
        ax.stairs(
            density,
            edges,
            color=colour,
            linewidth=1.6,
            label=f"{label}: {sampled:,} sampled{stands_for}",
        )
    _threshold_line(ax, threshold, threshold_label(threshold))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("log10 odds of the score (0 is a score of 0.5)")
    ax.set_ylabel("Weighted density")
    # Under the axes: the densities and the threshold leave no corner free.
    legend_below(ax, *ax.get_legend_handles_labels(), ncols=1)
    ax.set_title("Test audit: score densities of mules and non-mules")
    return ax


def top_shares(scores: SplitScores) -> NDArray[np.float64]:
    """Each account's rank from the top: the weighted share of accounts scored above it.

    Half of the accounts tied with it (itself included) count as above, so this is one
    minus its weighted percentile rank; the top-scored of 47,749 accounts is at 0.001%.
    """
    order = np.argsort(scores.score, kind="stable")
    ranked = scores.score[order]
    cumulative = np.concatenate(([0.0], np.cumsum(scores.weight[order])))
    below = cumulative[np.searchsorted(ranked, scores.score, side="left")]
    upto = cumulative[np.searchsorted(ranked, scores.score, side="right")]
    return 1 - (below + upto) / 2 / cumulative[-1]


def plot_revealed_vs_hidden(ax: Axes, scores: SplitScores) -> Axes:
    """The empirical CDF of the ranks of revealed and of hidden mules, from the top.

    Revealed mules had their labels revealed in the graph before the split's cutoff, so
    training may have seen accounts like them; hidden mules the model never saw
    labelled. Each mule is a point at its rank (top_shares) on a log axis, as in the
    capture figure, and each group's dotted line is its median.
    """
    if scores.revealed is None:
        raise ValueError("The revealed-and-hidden figure needs the audit's revealed flags")
    ranks = top_shares(scores)
    mules = scores.y == 1
    groups = (
        ("hidden mules", mules & ~scores.revealed, HIDDEN),
        ("revealed mules", mules & scores.revealed, REVEALED),
    )
    start = 1e-4
    for label, rows, colour in groups:
        found = np.sort(ranks[rows])
        if not len(found):
            continue
        start = min(start, float(found[0]) / 2)
        share = np.arange(1, len(found) + 1) / len(found)
        median = float(np.median(found))
        ax.step(np.append(found, 1.0), np.append(share, 1.0), where="post", color=colour)
        ax.plot(
            found,
            share,
            marker="o",
            linestyle="none",
            color=colour,
            markeredgecolor=SURFACE,
            label=f"{label} ({len(found)}): median in the top {share_label(median)}",
        )
        ax.axvline(median, color=colour, linestyle=":", linewidth=1.2)
    top_share_axis(ax, start)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Rank from the top: share of accounts scored higher (log scale)")
    ax.set_ylabel("Share of the group's mules ranked within it")
    ax.legend(loc="upper left")
    ax.set_title("Test audit: where revealed and hidden mules rank")
    return ax
