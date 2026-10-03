"""The figures of a diagnostic study, from the long tables its analyses write.

Each plot function draws on the Axes it is given, from a table in the long format of
artifacts.DIAGNOSTIC_TABLES, and returns it; none reads or saves a file
(reporting.study_report.write_diagnostics_report does). The held-out splits keep their
colours (validation for decisions, test for reporting); train, which only fixes
directions and fits, is grey. The learners take MEASURES in order, and the run the study
compares with, the built-in run, is ink.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter
import numpy as np
import pandas as pd

from ..metrics import REVIEW_BUDGETS, budget_name
from .comparison import PURPOSES
from .ranking import share_label
from .style import (
    BASELINE,
    MEASURES,
    MUTED,
    SPLIT_COLOURS,
    SURFACE,
    legend_below,
    measure,
    number,
)

# The splits of the tables, in drawing order, and their colours; train is grey.
SPLITS = ("train", "validation", "test")
COLOURS = {"train": MUTED, **SPLIT_COLOURS}
# How far each split's dot sits from its row's middle (rows are 1 apart), so equal values
# of two splits stay apart.
ROW_OFFSETS = {"train": 0.0, "validation": 0.14, "test": -0.14}
# The learners of the baselines, drift and learning curve, and how the figures name them.
LEARNERS = {"lr": "logistic regression", "hgb": "gradient-boosted trees"}
# The rows a figure of one row per feature or baseline shows at most.
FEATURE_ROWS = 30
SHIFT_ROWS = 25
# A standardised mean difference this large is commonly called small (0.2) or large (0.8);
# the drift figure is linear inside LINEAR_SMD and logarithmic beyond it.
SMALL_SMD = 0.2
LINEAR_SMD = 1.0


def feature_label(name: str) -> str:
    """A feature column as the figures name it: its family, a colon, its name."""
    family, _, rest = name.partition("__")
    return f"{family}: {rest}" if rest else name


def _names(ax: Axes, names: Sequence[str]) -> np.ndarray:
    """Row names down the y axis, the first on top; returns each row's y."""
    y = np.arange(len(names), 0, -1, dtype=np.float64) - 1
    ax.set_yticks(y, list(names))
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.grid(axis="y", visible=False)
    return y


def _dot(
    colour: str, *, hollow: bool = False, marker: str = "o", size: float = 6.0
) -> dict[str, Any]:
    """The style of a point: filled with a light edge, or hollow with the colour's edge."""
    return {
        "marker": marker,
        "markersize": size,
        "markerfacecolor": SURFACE if hollow else colour,
        "markeredgecolor": colour if hollow else SURFACE,
        "markeredgewidth": 1.2,
        "linestyle": "none",
        "color": colour,
    }


def _legend(entries: Sequence[tuple[str, dict[str, Any]]]) -> tuple[list[Line2D], list[str]]:
    return [Line2D([], [], **style) for _, style in entries], [label for label, _ in entries]


def _decimal(value: float, _: object = None) -> str:
    """A tick of a logarithmic axis of shares or AP, as a plain number: 0.001, 0.05, 1."""
    return f"{value:g}"


def strongest_features(
    table: pd.DataFrame, count: int = FEATURE_ROWS, split: str = "validation"
) -> list[str]:
    """The features of a univariate table whose ROC AUC on a split is farthest from 0.5.

    Decisions use the validation split, so the features are ranked there.
    """
    auc = table[(table.split == split) & (table.metric == "roc_auc")]
    distance = (auc.value - 0.5).abs().to_numpy()
    order = np.argsort(-distance, kind="stable")
    return auc.feature.iloc[order].head(count).tolist()


def plot_univariate(ax: Axes, table: pd.DataFrame, *, count: int = FEATURE_ROWS) -> Axes:
    """Each feature alone: its weighted ROC AUC per split, the strongest on validation first.

    A feature left of 0.5 is lower for mules. Train is hollow and grey: it fixed each
    feature's direction for the AP in the table. Within a row validation sits above and
    test below (ROW_OFFSETS), so equal values stay visible.
    """
    names = strongest_features(table, count)
    auc = table[table.metric == "roc_auc"].pivot_table(
        index="feature", columns="split", values="value"
    )
    y = _names(ax, [feature_label(name) for name in names])
    for split in SPLITS:
        if split in auc.columns:
            values = auc.reindex(names)[split].to_numpy(np.float64)
            ax.plot(values, y + ROW_OFFSETS[split], **_dot(COLOURS[split], hollow=split == "train"))
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Weighted ROC AUC of the raw value (below 0.5: lower for mules)")
    entries = [(split, _dot(COLOURS[split], hollow=split == "train")) for split in SPLITS]
    entries.append(("chance", {"color": MUTED, "linestyle": "--", "linewidth": 1.0}))
    legend_below(ax, *_legend(entries))
    ax.set_title(f"Each feature alone: the {len(names)} strongest on validation")
    return ax


def strongest_shifts(table: pd.DataFrame, count: int = SHIFT_ROWS) -> list[str]:
    """The features of a drift table whose non-mules shift most from train (largest |SMD|)."""
    smd = table[table.metric == "smd"]
    largest = smd.assign(size=smd.value.abs()).groupby("feature")["size"].max()
    return largest.sort_values(ascending=False, kind="stable").head(count).index.tolist()


def plot_drift(ax: Axes, table: pd.DataFrame, *, count: int = SHIFT_ROWS) -> Axes:
    """The non-mules' standardised mean difference from train at the held-out cutoffs.

    The features that shift most first. The axis is linear within plus or minus
    LINEAR_SMD and logarithmic beyond, since the history counts shift by several
    standard deviations while most features shift by a fraction of one; the dotted lines
    mark a small difference (plus or minus 0.2).
    """
    names = strongest_shifts(table, count)
    smd = table[table.metric == "smd"].pivot_table(index="feature", columns="split", values="value")
    y = _names(ax, [feature_label(name) for name in names])
    drawn = [0.0]
    for split in ("validation", "test"):
        if split in smd.columns:
            values = smd.reindex(names)[split].to_numpy(np.float64)
            ax.plot(values, y + ROW_OFFSETS[split], **_dot(COLOURS[split]))
            drawn += values[np.isfinite(values)].tolist()
    ax.axvline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    for edge in (-SMALL_SMD, SMALL_SMD):
        ax.axvline(edge, color=MUTED, linestyle=":", linewidth=0.9, zorder=1)
    ax.set_xscale("symlog", linthresh=LINEAR_SMD, linscale=1.0)
    # Room beyond the outermost points, which a symmetric log axis does not leave.
    ax.set_xlim(min(-0.5, 1.4 * min(drawn)), max(0.5, 1.4 * max(drawn)))
    ax.xaxis.set_major_formatter(FuncFormatter(_decimal))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Standardised mean difference from train (log beyond ±1)")
    entries = [(f"{s} cutoff against train", _dot(COLOURS[s])) for s in ("validation", "test")]
    entries.append((f"a small difference (±{SMALL_SMD:g})", {"color": MUTED, "linestyle": ":"}))
    legend_below(ax, *_legend(entries))
    ax.set_title(f"Feature drift of non-mules: the {len(names)} largest")
    return ax


def baseline_rows(table: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """The rankings of a baselines table in drawing order: (label, baseline, features, model).

    The run first, then the PU baselines by feature family and learner, the floor, and
    the single features. Chance is a line, not a row.
    """
    keys = table[["baseline", "features", "model"]].drop_duplicates()
    rows: list[tuple[str, str, str, str]] = []
    for _, key in keys[keys.baseline == "model"].iterrows():
        rows.append((f"the run: {key.features}", "model", str(key.features), "audit"))
    pu = keys[(keys.baseline == "pu") & (keys.features != "attributes")]
    for family in pu.features.unique():
        for kind in LEARNERS:
            if ((pu.features == family) & (pu.model == kind)).any():
                rows.append((f"{family}, {kind.upper()}", "pu", str(family), kind))
    floor = keys[(keys.baseline == "pu") & (keys.features == "attributes")]
    for kind in floor.model:
        rows.append((f"attribute floor, {str(kind).upper()}", "pu", "attributes", str(kind)))
    for name in keys[keys.baseline == "single_feature"].features:
        rows.append((f"{feature_label(str(name))} alone", "single_feature", str(name), "raw"))
    return rows


def plot_baselines(
    ax: Axes,
    table: pd.DataFrame,
    *,
    split: str,
    labels: bool = True,
    interval: str = "ring-clustered 90% interval",
) -> Axes:
    """Each baseline's audit AP on one split, with its interval.

    The run (ink) is the model's recorded audit; the PU baselines are fitted at the train
    cutoff on the revealed train mules; the single features rank with no training; the
    dashed line is chance, a random ranking's AP (the weighted prevalence). The x axis is
    logarithmic, since AP spans chance to far above it. ``labels`` False leaves the row
    names to a panel beside this one; ``interval`` names the intervals the table holds
    (the baselines' are ring-clustered, metrics.bootstrap_intervals).
    """
    rows = baseline_rows(table)
    chosen = table[(table.split == split) & (table.metric == "average_precision")]
    y = _names(ax, [label for label, *_ in rows])
    if not labels:
        ax.tick_params(axis="y", labelleft=False)
    colour = SPLIT_COLOURS[split]
    for position, (_, baseline, features, model) in zip(y, rows, strict=True):
        found = chosen[
            (chosen.baseline == baseline) & (chosen.features == features) & (chosen.model == model)
        ]
        if not len(found):
            continue
        row = found.iloc[0]
        shade = BASELINE if baseline == "model" else colour
        if np.isfinite(row.low) and np.isfinite(row.high):
            ax.plot([row.low, row.high], [position, position], color=shade, linewidth=2.2)
        ax.plot(row.value, position, **_dot(shade, hollow=baseline == "single_feature"))
    entries = [
        ("the run's audit", _dot(BASELINE)),
        ("fitted on the revealed train mules", _dot(colour)),
        ("one feature, no training", _dot(colour, hollow=True)),
        (interval, {"color": colour, "linewidth": 2.2}),
    ]
    chance = chosen[chosen.baseline == "chance"]
    if len(chance):
        prevalence = float(chance.value.iloc[0])
        ax.axvline(prevalence, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
        entries.append(
            (f"chance (prevalence {number(prevalence)})", {"color": MUTED, "linestyle": "--"})
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_decimal))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"{split.capitalize()} audit average precision (log scale)")
    legend_below(ax, *_legend(entries))
    ax.set_title(f"{split.capitalize()} audit, {PURPOSES[split]}")
    return ax


# The metrics a learning curve can draw, and how its labels name them.
CURVE_NAMES = {"average_precision": ("average precision", "AP"), "roc_auc": ("ROC AUC", "ROC AUC")}


def plot_learning_curve(
    ax: Axes, table: pd.DataFrame, *, split: str = "test", metric: str = "average_precision"
) -> Axes:
    """An audit metric against the number of oracle-labelled train mules a learner had.

    Per learner, the mean over the random draws of k mules and the range of the draws;
    a star is the learner fitted on the revealed train mules alone, the labels training
    has, and the ink line the run's audit, at the revealed count (dotted). ``metric`` is
    the average precision, which a mule or two at the top moves, or the ROC AUC.
    """
    name, short = CURVE_NAMES[metric]
    chosen = table[(table.split == split) & (table.metric == metric)]
    random = chosen[chosen.labels == "random"]
    for index, kind in enumerate(LEARNERS):
        mine = random[random.model == kind]
        if not len(mine):
            continue
        grouped = mine.groupby("mules").value
        k = grouped.mean().index.to_numpy(np.float64)
        ax.fill_between(
            k, grouped.min(), grouped.max(), color=MEASURES[index], alpha=0.18, linewidth=0
        )
        ax.plot(
            k,
            grouped.mean(),
            marker="o",
            markersize=4,
            label=f"{LEARNERS[kind]}: mean and range over random mules",
            **measure(index),
        )
        revealed = chosen[(chosen.labels == "revealed") & (chosen.model == kind)]
        for _, row in revealed.iterrows():
            ax.plot(row.mules, row.value, **_dot(MEASURES[index], marker="*", size=12))
    run = chosen[chosen.model == "model"]
    handles, labels = ax.get_legend_handles_labels()
    if len(run):
        row = run.iloc[0]
        ax.axhline(row.value, color=BASELINE, linewidth=1.2)
        ax.axvline(row.mules, color=BASELINE, linestyle=":", linewidth=1.0)
        handles.append(Line2D([], [], color=BASELINE, linewidth=1.2))
        labels.append(
            f"the run's audit: {short} {number(row.value)}, {int(row.mules)} revealed mules"
        )
    handles.append(Line2D([], [], **_dot(MUTED, marker="*", size=12)))
    labels.append("fitted on the revealed train mules only")
    ax.set_xscale("log")
    counts = sorted(int(k) for k in chosen.mules.unique())
    ax.set_xticks(counts, [str(k) for k in counts])
    ax.xaxis.set_minor_formatter(NullFormatter())
    if metric == "roc_auc":
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
        handles.append(Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0))
        labels.append("chance")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Train mules with oracle labels (log scale)")
    ax.set_ylabel(f"{split.capitalize()} audit {name}")
    legend_below(ax, handles, labels)
    ax.set_title(f"Learning curve, {split} audit, {PURPOSES[split]}")
    return ax


def plot_ap_concentration(ax: Axes, table: pd.DataFrame) -> Axes:
    """The AP the top-ranked mules make: its running sum over the mules, best ranked first.

    Each split's dashed line is its whole AP; a curve that reaches it within a few mules
    says a handful of mules make the AP, so it moves in large steps.
    """
    chosen = table[table.metric == "cumulative_average_precision"]
    for split in ("validation", "test"):
        mine = chosen[chosen.split == split]
        if not len(mine):
            continue
        colour = COLOURS[split]
        ax.plot(mine["rank"], mine.value, drawstyle="steps-post", color=colour, label=split)
        total = float(mine.value.iloc[-1])
        ax.axhline(
            total, color=colour, linestyle="--", linewidth=0.9, label=f"{split} AP {number(total)}"
        )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mules, ranked by score, highest first")
    ax.set_ylabel("Average precision of the mules ranked so far")
    ax.legend(loc="lower right")
    ax.set_title("How few mules make the audit AP")
    return ax


def plot_ring_coverage(ax: Axes, table: pd.DataFrame) -> Axes:
    """The share of each split's rings with a member in the top 1, 5 and 10% of accounts.

    Beside each split's bar, a marker for the share of its mules there (the recall), so a
    bar above its marker says the ranking reaches more rings than mules.
    """
    width = 0.36
    x = np.arange(len(REVIEW_BUDGETS), dtype=np.float64)
    for offset, split in zip((-width / 2, width / 2), ("validation", "test"), strict=True):
        rows = table[table.split == split]
        rings = rows[rows.subset == "rings"].set_index("metric").value
        mules = rows[rows.subset == "mules"].set_index("metric").value
        if "rings" not in rings or not rings["rings"]:
            continue
        colour = COLOURS[split]
        shares = [float(rings.get(f"coverage_at_{budget_name(f)}", np.nan)) for f in REVIEW_BUDGETS]
        ax.bar(
            x + offset,
            shares,
            width * 0.9,
            color=colour,
            label=f"{split}: {int(rings['rings'])} rings",
        )
        recall = [float(mules.get(f"coverage_at_{budget_name(f)}", np.nan)) for f in REVIEW_BUDGETS]
        ax.plot(x + offset, recall, **_dot(BASELINE, marker="D"))
        for place, share in zip(x + offset, shares, strict=True):
            ax.annotate(
                f"{share:.0%}",
                (place, share),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=colour,
            )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], **_dot(BASELINE, marker="D")))
    labels.append("share of the split's mules there (recall)")
    ax.set_xticks(x, [f"top {share_label(f)}" for f in REVIEW_BUDGETS])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Rings with a member there")
    ax.set_xlabel("Top share of the split's accounts, by score")
    legend_below(ax, handles, labels)
    ax.set_title("Rings the audit's ranking reaches")
    return ax


# The subsets of the proxy validity table and how its figure names them.
PROXY_SUBSETS = {
    "all": "all predicted\naccounts",
    "hidden": "hidden mules\nagainst non-mules",
    "revealed": "revealed mules\nagainst non-mules",
}


def plot_proxy_validity(ax: Axes, table: pd.DataFrame) -> Axes:
    """The ROC AUC of the proxy predictions against the ground truth, by subset and split.

    All the predicted accounts, the hidden mules against the non-mules, and the revealed
    mules against them; each bar is labelled with its AP. A proxy that ranks revealed
    mules well and hidden ones near chance measures the reveal, not mule detection.
    """
    subsets = list(PROXY_SUBSETS)
    width = 0.36
    x = np.arange(len(subsets), dtype=np.float64)
    for offset, split in zip((-width / 2, width / 2), ("validation", "test"), strict=True):
        rows = table[table.split == split]
        if not len(rows):
            continue
        wide = rows.pivot_table(index="subset", columns="metric", values="value", dropna=False)
        wide = wide.reindex(subsets)
        auc = wide["roc_auc"].to_numpy(np.float64) if "roc_auc" in wide else np.full(3, np.nan)
        ap = (
            wide["average_precision"].to_numpy(np.float64)
            if "average_precision" in wide
            else np.full(3, np.nan)
        )
        colour = COLOURS[split]
        ax.bar(x + offset, auc, width * 0.9, color=colour, label=split)
        for place, height, value in zip(x + offset, auc, ap, strict=True):
            if np.isfinite(height):
                text = f"AP {number(value)}" if np.isfinite(value) else "AP n/a"
                ax.annotate(
                    text,
                    (place, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=colour,
                )
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xticks(x, list(PROXY_SUBSETS.values()))
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC AUC against the ground truth")
    ax.set_xlabel("The accounts of the proxy predictions scored")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0))
    labels.append("chance")
    legend_below(ax, handles, labels)
    ax.set_title("The proxy predictions against the ground truth")
    return ax


def plot_reveal_spread(
    ax: Axes, table: pd.DataFrame, *, budget: int | None, salt: int | None
) -> Axes:
    """Per split, the mules a bank would have discovered by the cutoff and those revealed.

    Over the replayed salts: the 5th to 95th percentile as a bar, the median as a dot, and
    the configured salt's outcome as an ink diamond. Each split's tick is its mules; the
    dashed line is the reveal's budget per split.
    """
    names, rows = [], []
    for split in SPLITS:
        for metric, what in (("eligible", "discovered by the cutoff"), ("revealed", "revealed")):
            names.append(f"{split}: {what}")
            rows.append((split, metric))
    y = _names(ax, names)
    for position, (split, metric) in zip(y, rows, strict=True):
        found = table[(table.split == split) & (table.metric == metric)]
        if not len(found):
            continue
        colour = COLOURS[split]
        low, middle, high = np.percentile(found.value.to_numpy(np.float64), [5, 50, 95])
        ax.plot(
            [low, high],
            [position, position],
            color=colour,
            linewidth=5,
            solid_capstyle="butt",
            alpha=0.45,
        )
        ax.plot(middle, position, **_dot(colour))
        if salt is not None and (found.salt == salt).any():
            chosen = float(found[found.salt == salt].value.iloc[0])
            ax.plot(chosen, position, **_dot(BASELINE, marker="D"))
        mules = table[(table.split == split) & (table.metric == "mules")].value
        if len(mules):
            ax.plot(
                float(mules.iloc[0]),
                position,
                marker="|",
                markersize=12,
                markeredgewidth=1.6,
                color=colour,
                linestyle="none",
            )
    if budget is not None:
        ax.axvline(budget, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Mules")
    entries = [
        ("5th to 95th percentile over salts", {"color": MUTED, "linewidth": 5, "alpha": 0.45}),
        ("median over salts", _dot(MUTED)),
        (
            "the split's mules",
            {
                "marker": "|",
                "markersize": 12,
                "markeredgewidth": 1.6,
                "color": MUTED,
                "linestyle": "none",
            },
        ),
    ]
    if salt is not None:
        entries.insert(2, (f"the configured salt ({salt})", _dot(BASELINE, marker="D")))
    if budget is not None:
        entries.append((f"budget per split ({budget})", {"color": MUTED, "linestyle": "--"}))
    legend_below(ax, *_legend(entries))
    ax.set_title(f"The label reveal replayed over {table.salt.nunique():,} salts")
    return ax


def plot_nnpu_simulation(ax: Axes, table: pd.DataFrame) -> Axes:
    """The simulated test ROC AUC of each positive weight, per seed and on average.

    The weights are categories, evenly spaced, each labelled with how many of its seeds
    collapsed (their labelled positives scored near zero). The textbook weight is the
    class prior (0.001); the built-in run's is one minus it (0.999).
    """
    chosen = table.pivot_table(index=["positive_weight", "seed"], columns="metric", values="value")
    weights = sorted(float(w) for w in chosen.index.get_level_values(0).unique())
    colour = MEASURES[0]
    means, ticks = [], []
    for place, weight in enumerate(weights):
        mine = chosen.xs(weight, level="positive_weight")
        auc = mine.test_roc_auc.to_numpy(np.float64)
        spread = np.linspace(-0.08, 0.08, len(auc)) if len(auc) > 1 else np.zeros(1)
        ax.plot(place + spread, auc, **_dot(colour, hollow=True))
        means.append(float(auc.mean()))
        collapsed = int(mine.collapsed.sum())
        role = {weights[0]: " (the prior)", weights[-1]: " (balanced)"}.get(weight, "")
        ticks.append(f"{weight:g}{role}\n{collapsed} of {len(mine)} collapsed")
    ax.plot(
        range(len(weights)),
        means,
        marker="o",
        markersize=7,
        color=colour,
        markeredgecolor=SURFACE,
    )
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(weights)), ticks)
    ax.set_xlim(-0.5, len(weights) - 0.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("nnPU positive weight")
    ax.set_ylabel("Simulated test ROC AUC")
    entries = [
        ("one seed", _dot(colour, hollow=True)),
        ("mean over seeds", _dot(colour)),
        ("chance", {"color": MUTED, "linestyle": "--"}),
    ]
    ax.legend(*_legend(entries), loc="lower right")
    ax.set_title("The nnPU positive weight, simulated")
    return ax
