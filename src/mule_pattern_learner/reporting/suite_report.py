"""A control-experiment suite's figures and report.md, from its tables and runs' files.

write_suite_report reads the suite's summary.csv and comparison.csv and the validation
audits and epochs of its complete runs, draws the comparison figures
(reporting.comparison) and rewrites its report.md: the variants ranked by the validation
audit, their seed ensembles, the tables and the figures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..artifacts import (
    DELTA_METRIC,
    ENSEMBLE,
    PROXY_METRIC,
    SEED_MEAN,
    atomic_write,
    read_audit_scores,
    read_comparison,
    read_epochs,
    read_json,
    read_run_provenance,
    read_summary,
)
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import INTERVAL, REVIEW_BUDGETS, budget_name
from ..paths import BASELINE_VARIANT, SuitePaths
from .comparison import (
    MeanCurve,
    VariantSeeds,
    mean_capture,
    plot_budget_recall,
    plot_capture_overlay,
    plot_comparison,
    plot_paired_delta,
    plot_proxy_vs_audit,
    plot_validation_overlay,
)
from .document import (
    Drawing,
    draw,
    figure_links,
    one,
    rows_size,
    table,
    value_or_none,
)
from .ranking import SplitScores, share_label
from .style import BASELINE, MUTED, PANEL, SPLIT_COLOURS, estimate, number

# The figures of a control-experiment suite, and what its report.md calls them.
SUITE_FIGURES = {
    "comparison_ap": "Audit AP per variant on validation and test",
    "comparison_delta": "Validation audit AP against the baseline, paired",
    "comparison_budget": "Validation audit recall at the review budgets",
    "comparison_capture": "Seed-mean validation audit capture against the baseline",
    "comparison_validation": "Seed-mean proxy validation AP per epoch against the baseline",
    "comparison_proxy_vs_audit": "Selected proxy AP against the validation audit AP, per run",
}
# The small multiples of a suite: panels per row.
PANELS_PER_ROW = 3


@dataclass(frozen=True)
class SuiteFiles:
    """What a suite's tables and its complete runs saved, as its figures read them.

    ``comparison`` holds comparison.csv's seed-mean rows, ranked by the seed-mean
    validation audit AP, best first, with the variants the audit could not rank last in
    the suite's order; ``ensembles`` its seed-ensemble rows, ranked by their own
    validation audit AP. ``captures`` and ``epochs`` hold each complete run's validation
    audit sample and epochs.csv, by variant and seed.
    """

    summary: pd.DataFrame
    comparison: pd.DataFrame
    ensembles: pd.DataFrame
    captures: dict[str, dict[int, SplitScores]]
    epochs: dict[str, dict[int, pd.DataFrame]]
    prevalence: float | None


def complete_runs(summary: pd.DataFrame) -> list[tuple[str, int]]:
    """The (variant, seed) of every complete run of a suite, in summary.csv's order."""
    complete = summary[summary.status == "complete"]
    return list(dict.fromkeys(zip(complete.variant, complete.seed.astype(int), strict=True)))


def suite_files(suite: SuitePaths) -> SuiteFiles:
    """summary.csv, comparison.csv, and the validation audits and epochs of complete runs."""
    summary = read_summary(suite.summary)
    table = read_comparison(suite.comparison)
    ranked = table.sort_values("validation_ap", ascending=False, na_position="last")
    comparison = ranked[ranked.estimate == SEED_MEAN].reset_index(drop=True)
    ensembles = ranked[ranked.estimate == ENSEMBLE].reset_index(drop=True)
    captures: dict[str, dict[int, SplitScores]] = {}
    epochs: dict[str, dict[int, pd.DataFrame]] = {}
    prevalence = None
    for variant, seed in complete_runs(summary):
        run = suite.run(variant, seed)
        report = read_json(run.audit_report("validation"))
        frame = read_audit_scores(run.audit_scores("validation"))
        captures.setdefault(variant, {})[seed] = SplitScores(
            y=frame.is_mule.to_numpy(np.int64),
            score=frame.score.to_numpy(np.float64),
            weight=1 / frame.inclusion_probability.to_numpy(np.float64),
            metrics=report["metrics"],
        )
        epochs.setdefault(variant, {})[seed] = read_epochs(run.epochs)
        if prevalence is None:
            prevalence = read_json(run.metrics)["validation_proxy"].get("prevalence")
    return SuiteFiles(summary, comparison, ensembles, captures, epochs, prevalence)


def _bounds(record: Mapping[str, Any], interval: str | None) -> tuple[float, float] | None:
    """A row's interval, from the columns ``interval`` names without _low and _high."""
    if interval is None:
        return None
    low, high = record[f"{interval}_low"], record[f"{interval}_high"]
    return None if pd.isna(low) or pd.isna(high) else (float(low), float(high))


def variant_seeds(
    files: SuiteFiles,
    split: str,
    metric: str,
    column: str,
    *,
    interval: str | None = None,
    audit: str | None = None,
    ensemble: bool = False,
) -> list[VariantSeeds]:
    """Each ranked variant's per-seed values of a summary metric and its comparison column.

    ``interval`` names the comparison columns of the interval without their _low and
    _high endings, and ``audit`` those of a second, audit-only interval. With
    ``ensemble`` each variant also gets its seed ensemble's value of the column, and its
    interval, where the suite has one.
    """
    summary = files.summary
    chosen = summary[(summary.split == split) & (summary.metric == metric)]
    chosen = chosen[chosen.status == "complete"]
    combined = {str(row["variant"]): row for row in records(files.ensembles)}
    rows = []
    for record in records(files.comparison):
        name = str(record["variant"])
        mine = chosen[chosen.variant == name]
        joint = combined.get(name) if ensemble else None
        rows.append(
            VariantSeeds(
                name,
                dict(zip(mine.seed.astype(int), mine.value.astype(float), strict=True)),
                value_or_none(record[column]),
                _bounds(record, interval),
                str(record["consistent"]) == "True",
                ensemble=None if joint is None else value_or_none(joint[column]),
                ensemble_interval=None if joint is None else _bounds(joint, interval),
                audit_interval=_bounds(record, audit),
            )
        )
    return rows


def panels(
    draws: list[Callable[[Axes], Axes]],
    legend: tuple[list[Line2D], list[str]],
    labels: tuple[str, str],
) -> Drawing:
    """Small multiples: one panel per drawing, sharing axes, with one legend below them."""

    def drawing(figure: Figure) -> None:
        rows = math.ceil(len(draws) / PANELS_PER_ROW)
        columns = min(len(draws), PANELS_PER_ROW)
        axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
        flat = list(axes.flat)
        for ax, draw_panel in zip(flat, draws, strict=False):
            draw_panel(ax)
        for index, ax in enumerate(flat[len(draws) :], start=len(draws)):
            ax.set_visible(False)
            # The panel above an empty place keeps its x tick labels.
            flat[index - columns].xaxis.set_tick_params(labelbottom=True)
        figure.supxlabel(labels[0], fontsize=9)
        figure.supylabel(labels[1], fontsize=9)
        # Above the panels, clear of the x label below them.
        figure.legend(*legend, loc="outside upper center", ncols=len(legend[0]))

    return drawing


def panels_size(count: int) -> tuple[float, float]:
    """Small multiples of count panels: at least a panel's width, taller with more rows."""
    rows = math.ceil(count / PANELS_PER_ROW)
    return (max(PANEL[0], 2.5 * min(count, PANELS_PER_ROW) + 0.8), 2.2 * rows + 1.0)


def overlay_legend(split: str, reference: str) -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D([], [], color=BASELINE, linewidth=1.4),
        Line2D([], [], color=SPLIT_COLOURS[split], linewidth=1.8),
        Line2D([], [], color=MUTED, linestyle="--", linewidth=0.9),
    ]
    return handles, ["baseline, mean over seeds", "the panel's variant", reference]


def suite_drawings(files: SuiteFiles) -> dict[str, tuple[Drawing, tuple[float, float]]]:
    """The drawing and size of each suite figure its complete runs allow."""
    drawings: dict[str, tuple[Drawing, tuple[float, float]]] = {}
    if not files.captures:
        return drawings
    count = len(files.comparison)
    splits = {
        split: variant_seeds(
            files,
            split,
            "average_precision",
            f"{split}_ap",
            interval=f"{split}_ap",
            ensemble=True,
        )
        for split in HELD_OUT_SPLITS
    }

    def both(figure: Figure) -> None:
        axes = figure.subplots(1, len(splits), sharey=True, squeeze=False)[0]
        for ax, (split, rows) in zip(axes, splits.items(), strict=True):
            baseline = next((row.mean for row in rows if row.name == BASELINE_VARIANT), None)
            plot_comparison(ax, rows, split=split, baseline=baseline)

    drawings["comparison_ap"] = (both, rows_size(count, width=9.0))
    deltas = variant_seeds(
        files,
        "validation",
        DELTA_METRIC,
        "validation_ap_delta",
        interval="validation_ap_delta",
        audit="validation_ap_delta_audit",
    )
    deltas = sorted(
        (row for row in deltas if row.mean is not None),
        key=lambda row: -(row.mean if row.mean is not None else 0.0),
    )
    if deltas:
        drawings["comparison_delta"] = (
            one(lambda ax: plot_paired_delta(ax, deltas)),
            rows_size(len(deltas)),
        )
    budgets = {
        fraction: variant_seeds(
            files,
            "validation",
            f"recall_at_{budget_name(fraction)}",
            f"validation_recall_at_{budget_name(fraction)}",
        )
        for fraction in REVIEW_BUDGETS
    }
    drawings["comparison_budget"] = (
        one(lambda ax: plot_budget_recall(ax, budgets)),
        rows_size(count, per_row=0.55),
    )
    # Every panel holds the baseline beside its variant, so the baseline has no panel of
    # its own unless it is alone.
    audited = [name for name in files.comparison.variant if name in files.captures]
    order = [name for name in audited if name != BASELINE_VARIANT] or audited
    prevalences = [
        scores.prevalence for runs in files.captures.values() for scores in runs.values()
    ]
    grid = np.geomspace(min([1e-4, *(p / 2 for p in prevalences)]), 1.0, 200)
    captures = {
        name: mean_capture(name, list(files.captures[name].values()), grid) for name in audited
    }
    drawings["comparison_capture"] = (
        panels(
            [
                partial(
                    plot_capture_overlay,
                    variant=captures[name],
                    baseline=captures.get(BASELINE_VARIANT),
                )
                for name in order
            ],
            overlay_legend("validation", "random ranking"),
            ("Top share of accounts reviewed, by score (log scale)", "Share of mules found"),
        ),
        panels_size(len(order)),
    )
    curves = {name: mean_epochs(name, files.epochs[name]) for name in audited}
    last = max(int(curve.x.max()) for curve in curves.values())
    drawings["comparison_validation"] = (
        panels(
            [
                partial(
                    plot_validation_overlay,
                    variant=curves[name],
                    baseline=curves.get(BASELINE_VARIANT),
                    prevalence=files.prevalence,
                    last_epoch=last,
                )
                for name in order
            ],
            overlay_legend("validation", "prevalence: a random ranking's AP"),
            ("Epoch", "Proxy validation AP"),
        ),
        panels_size(len(order)),
    )
    points = proxy_points(files.summary)
    if len(points):
        drawings["comparison_proxy_vs_audit"] = (
            one(
                lambda ax: plot_proxy_vs_audit(
                    ax,
                    points.variant.tolist(),
                    points[PROXY_METRIC].to_numpy(np.float64),
                    points.average_precision.to_numpy(np.float64),
                )
            ),
            PANEL,
        )
    return drawings


def mean_epochs(name: str, epochs: Mapping[int, pd.DataFrame]) -> MeanCurve:
    """A variant's proxy validation AP per epoch, averaged over its seeds.

    The curve ends at the last epoch every seed trained: early stopping ends seeds at
    different epochs, and a mean over the seeds that went on would read as the same one.
    """
    reached = min(int(frame.epoch.max()) for frame in epochs.values())
    joined = pd.concat(
        [frame.loc[frame.epoch <= reached, ["epoch", "validation_ap"]] for frame in epochs.values()]
    )
    mean = joined.groupby("epoch").validation_ap.mean()
    return MeanCurve(name, mean.index.to_numpy(np.float64), mean.to_numpy(np.float64), len(epochs))


def proxy_points(summary: pd.DataFrame) -> pd.DataFrame:
    """Each complete run's selected proxy AP and validation audit AP (variant, seed, both)."""
    chosen = summary[
        (summary.status == "complete")
        & (summary.split == "validation")
        & summary.metric.isin([PROXY_METRIC, "average_precision"])
    ]
    wide = chosen.pivot_table(index=["variant", "seed"], columns="metric", values="value")
    wanted = [PROXY_METRIC, "average_precision"]
    if not set(wanted) <= set(wide.columns):
        return pd.DataFrame(columns=["variant", "seed", *wanted])
    return wide.dropna(subset=wanted).reset_index()


def write_suite_report(suite: SuitePaths) -> dict[str, Any]:
    """Draw a suite's figures from its tables and runs, then its report.md.

    What the runner writes after comparing a suite (experiments.runner), and what
    `mule report` redraws offline from a suite directory.
    """
    files = suite_files(suite)
    return draw(suite, suite_drawings(files), lambda: write_suite_text(suite, files))


def _interval_text(value: Any, low: Any, high: Any) -> str:
    interval = None if pd.isna(low) or pd.isna(high) else [float(low), float(high)]
    return estimate(value_or_none(value), interval)


def suite_text(suite: SuitePaths, files: SuiteFiles) -> str:
    """A suite's report.md: runs, validation ranking, seed ensembles, test audit, figures."""
    summary, comparison = files.summary, files.comparison
    runs = summary[summary.status != ENSEMBLE].drop_duplicates(["variant", "seed"])
    statuses = runs.status.value_counts()
    counted = ", ".join(f"{number(int(n))} {status}" for status, n in statuses.items())
    seeds = ", ".join(str(seed) for seed in sorted(runs.seed.astype(int).unique()))
    lines = [
        f"# Suite {suite.root.name}",
        "",
        f"{len(comparison)} variants with the seeds {seeds}: {len(runs)} runs, {counted}.",
    ]
    provenance = [read_run_provenance(suite.run(v, s).config) for v, s in complete_runs(summary)]
    if provenance:
        datasets = sorted({str(p.get("dataset_id"))[:12] for p in provenance})
        commits = sorted({str(p.get("git_commit") or "unknown")[:12] for p in provenance})
        dirty = sum(bool(p.get("git_dirty")) for p in provenance)
        lines[-1] += f" Dataset `{'`, `'.join(datasets)}`, commit `{'`, `'.join(commits)}`" + (
            f"; {dirty} of the complete runs had uncommitted changes." if dirty else "."
        )
    unpaired = comparison.unpaired_accounts.dropna()
    # The comparisons with the baseline: the variants with a delta.
    compared = int(comparison.validation_ap_delta.notna().sum())
    lines += [
        "",
        "Decisions use the validation audit; the test audit is for reporting, not selection. "
        "The pool groups (pool_activity and pool_internal_inflows) were designed after "
        "reading test-split mules and the data generator's mule typology, so the test "
        "audit is optimistic for every variant that keeps them.",
        "",
        "## Validation audit, for decisions",
        "",
        f"Variants ranked by their mean validation audit AP over seeds. In parentheses: the "
        f"{INTERVAL:.0%} interval of the mean over paired bootstrap replicates, which draw one "
        "resample of the accounts every audit scored and apply it to every run. It covers "
        "the audit sample's uncertainty for these seeds, not the spread between seeds (the "
        "standard deviation beside it). The delta is the variant's mean AP minus the "
        "baseline's over the seeds both completed, each seed paired with the baseline's run "
        "of the same seed on those same accounts. Its interval covers both sources of "
        "uncertainty: each replicate also resamples the seeds, so the interval widens with "
        "the spread between them. The audit-only interval beside it resamples the accounts "
        "alone, for these seeds. The seeds that agree are those whose own delta has the "
        "sign of the mean. A delta is consistent when its interval over both sources "
        f"excludes zero. The suite makes {compared} comparison"
        f"{'' if compared == 1 else 's'} with the baseline, so at {INTERVAL:.0%} about "
        f"{compared * (1 - INTERVAL):.1f} would exclude zero by chance even if no variant "
        "differed from it: a single consistent delta is a lead to repeat, not a finding.",
        "",
    ]
    if len(unpaired) and unpaired.iloc[0] > 0:
        lines += [
            f"{number(int(unpaired.iloc[0]))} validation accounts that some run's audit "
            "rejected are left out of the pairing.",
            "",
        ]
    budgets = [budget_name(f) for f in REVIEW_BUDGETS]
    header = [
        "Rank",
        "Variant",
        "Seeds",
        "AP",
        "Standard deviation",
        "Delta from the baseline",
        "Audit-only interval",
        "Seeds that agree",
        "Consistent",
        "ROC AUC",
        "Recall in the top " + " / ".join(share_label(f) for f in REVIEW_BUDGETS),
    ]
    rows = []
    for rank, row in enumerate(records(comparison), start=1):
        compared_here = not pd.isna(row["validation_ap_delta"])
        delta = audit = agree = ""
        if compared_here:
            delta = _interval_text(
                row["validation_ap_delta"],
                row["validation_ap_delta_low"],
                row["validation_ap_delta_high"],
            )
            low, high = row["validation_ap_delta_audit_low"], row["validation_ap_delta_audit_high"]
            audit = "" if pd.isna(low) or pd.isna(high) else f"{number(low)} to {number(high)}"
            agree = (
                f"{int(row['validation_ap_delta_agreeing'])} of "
                f"{int(row['validation_ap_delta_seeds'])}"
            )
        rows.append(
            [
                str(rank),
                f"**{row['variant']}**" if row["variant"] == BASELINE_VARIANT else row["variant"],
                row["seeds"] or "none",
                _interval_text(
                    row["validation_ap"], row["validation_ap_low"], row["validation_ap_high"]
                ),
                number(value_or_none(row["validation_ap_spread"])),
                delta,
                audit,
                agree,
                {True: "yes", False: "no"}.get(row["consistent"], "") if compared_here else "",
                number(value_or_none(row["validation_roc_auc"])),
                " / ".join(
                    number(value_or_none(row[f"validation_recall_at_{b}"])) for b in budgets
                ),
            ]
        )
    lines += [*table(header, rows), "", *ensemble_section(files)]
    test_rows = [
        [
            row["variant"],
            _interval_text(row["test_ap"], row["test_ap_low"], row["test_ap_high"]),
            number(value_or_none(row["test_ap_spread"])),
            number(value_or_none(row["test_roc_auc"])),
            " / ".join(number(value_or_none(row[f"test_recall_at_{b}"])) for b in budgets),
        ]
        for row in records(comparison)
    ]
    lines += [
        "## Test audit, for reporting, not selection",
        "",
        "In the validation ranking's order; never rank or choose variants by these.",
        "",
        *table(["Variant", "AP", "Standard deviation", "ROC AUC", header[-1]], test_rows),
        "",
        "## Variants and runs",
        "",
    ]
    variant_rows = [
        [
            row["variant"],
            row["question"],
            row["changes"],
            "" if pd.isna(row["best_epoch"]) else f"{row['best_epoch']:.1f}",
            number(None if pd.isna(row["parameter_count"]) else int(row["parameter_count"])),
            "" if pd.isna(row["training_hours"]) else f"{row['training_hours']:.2f}",
            row["differs"],
        ]
        for row in records(comparison)
    ]
    lines += [
        *table(
            [
                "Variant",
                "Question",
                "Changes",
                "Mean best epoch",
                "Parameters",
                "Mean hours",
                "Runs that differ from the others",
            ],
            variant_rows,
        ),
        "",
    ]
    others = runs[runs.status != "complete"]
    if len(others):
        listed = ", ".join(
            f"{v} seed {int(s)} ({status})"
            for v, s, status in zip(others.variant, others.seed, others.status, strict=True)
        )
        lines += [f"Not compared: {listed}.", ""]
    lines += figure_links(suite, SUITE_FIGURES)
    return "\n".join(lines).rstrip("\n") + "\n"


def ensemble_section(files: SuiteFiles) -> list[str]:
    """report.md's seed ensembles, ranked by their validation audit AP; none without one."""
    if not len(files.ensembles):
        return []
    budgets = [budget_name(f) for f in REVIEW_BUDGETS]
    means = {str(row["variant"]): row for row in records(files.comparison)}
    rows = []
    for rank, row in enumerate(records(files.ensembles), start=1):
        name = str(row["variant"])
        seed_mean = means[name]["validation_ap"] if name in means else None
        rows.append(
            [
                str(rank),
                f"**{name}**" if name == BASELINE_VARIANT else name,
                row["seeds"],
                _interval_text(
                    row["validation_ap"], row["validation_ap_low"], row["validation_ap_high"]
                ),
                number(value_or_none(seed_mean)),
                number(value_or_none(row["validation_roc_auc"])),
                " / ".join(
                    number(value_or_none(row[f"validation_recall_at_{b}"])) for b in budgets
                ),
                _interval_text(row["test_ap"], row["test_ap_low"], row["test_ap_high"]),
            ]
        )
    header = [
        "Rank",
        "Variant",
        "Seeds",
        "Ensemble AP",
        "Mean of its seeds' AP",
        "ROC AUC",
        "Recall in the top " + " / ".join(share_label(f) for f in REVIEW_BUDGETS),
        "Test AP, for reporting",
    ]
    return [
        "## Seed ensembles",
        "",
        "Each variant's seeds combined into one model: their scores of the accounts every "
        "audit scored, averaged on the log-odds scale, then audited on validation and test "
        "as a run is. On that scale a seed that is confident about an account weighs more "
        "than a hesitant one, where averaging the seeds' ranks would give each the same say. "
        "Ranked by the ensemble's validation audit AP, with its "
        f"{INTERVAL:.0%} interval over the same paired replicates (the audit sample's "
        "uncertainty, for these seeds), beside the mean of its seeds' own AP: an ensemble "
        "above that mean gains from the seeds' disagreement.",
        "",
        *table(header, rows),
        "",
    ]


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A table's rows as dicts by column name."""
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def write_suite_text(suite: SuitePaths, files: SuiteFiles) -> Path:
    """Replace the suite's report.md with suite_text."""
    with atomic_write(suite.report) as pending:
        pending.write_text(suite_text(suite, files))
    return suite.report
