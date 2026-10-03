"""A diagnostic study's figures and report.md, from the long tables of `mule diagnose`.

write_diagnostics_report reads study.json, the feature table and each analysis' table,
draws the figures the tables it has allow (reporting.diagnostics) and rewrites the
study's report.md, one section per analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from ..artifacts import (
    DIAGNOSTIC_TABLES,
    atomic_write,
    read_diagnostic_table,
    read_feature_table,
    read_json,
)
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import INTERVAL, REVIEW_BUDGETS, budget_name
from ..paths import DiagnosticsPaths
from .diagnostics import (
    baseline_rows,
    plot_ap_concentration,
    plot_baselines,
    plot_drift,
    plot_learning_curve,
    plot_nnpu_simulation,
    plot_proxy_validity,
    plot_reveal_spread,
    plot_ring_coverage,
    plot_univariate,
    strongest_features,
    strongest_shifts,
)
from .document import (
    Drawing,
    Reported,
    draw,
    figure_links,
    one,
    rows_size,
    table,
    value_or_none,
)
from .ranking import share_label
from .style import PANEL, estimate, number

# The figures of a diagnostic study, and what its report.md calls them.
DIAGNOSTICS_FIGURES = {
    "baselines": "Each baseline's audit AP beside the run's",
    "learning_curve": "Audit AP against the oracle-labelled train mules a learner was fitted on",
    "univariate_auc": "Each feature's ROC AUC alone",
    "drift": "The non-mules' feature drift from the train cutoff",
    "ap_concentration": "How few mules make the audit AP",
    "ring_coverage": "Rings with a member in the review budgets",
    "proxy_validity": "The proxy predictions against the ground truth",
    "reveal_spread": "The label reveal replayed over salts",
    "nnpu_simulation": "The nnPU positive weight, simulated",
}
# The figures each analysis' table draws.
ANALYSIS_FIGURES = {
    "baselines": ("baselines",),
    "learning_curve": ("learning_curve",),
    "univariate": ("univariate_auc",),
    "drift": ("drift",),
    "subgroups": ("ap_concentration", "ring_coverage"),
    "proxy_validity": ("proxy_validity",),
    "reveal_spread": ("reveal_spread",),
    "nnpu_simulation": ("nnpu_simulation",),
}
# The rows report.md lists of the longest tables.
REPORTED_ROWS = 10


@dataclass(frozen=True)
class StudyFiles:
    """What a diagnostic study saved: study.json, the feature table and its analyses' tables.

    ``tables`` holds the analyses whose table exists, by DIAGNOSTIC_TABLES key.
    """

    record: dict[str, Any]
    features: pd.DataFrame | None
    tables: dict[str, pd.DataFrame]


def study_files(study: DiagnosticsPaths) -> StudyFiles:
    """study.json, features.parquet and every analysis' table the study holds."""
    record = read_json(study.study) if study.study.exists() else {}
    features = read_feature_table(study.features) if study.features.exists() else None
    tables = {
        name: read_diagnostic_table(study.table(name), name)
        for name in DIAGNOSTIC_TABLES
        if study.table(name).exists()
    }
    return StudyFiles(record, features, tables)


def paired(
    draws: tuple[Callable[[Axes], Axes], Callable[[Axes], Axes]], *, stacked: bool
) -> Drawing:
    """The drawing of two panels, validation's then test's: side by side sharing the y
    axis, or ``stacked`` one above the other, each with its own axes and legend."""

    def drawing(figure: Figure) -> None:
        first, second = figure.subplots(2, 1) if stacked else figure.subplots(1, 2, sharey=True)
        draws[0](first)
        draws[1](second)

    return drawing


def study_drawings(files: StudyFiles) -> dict[str, tuple[Drawing, tuple[float, float]]]:
    """The drawing and size of each figure the study's tables allow."""
    tables = files.tables
    drawings: dict[str, tuple[Drawing, tuple[float, float]]] = {}
    if "baselines" in tables:
        table = tables["baselines"]
        drawings["baselines"] = (
            paired(
                (
                    partial(plot_baselines, table=table, split="validation"),
                    partial(plot_baselines, table=table, split="test", labels=False),
                ),
                stacked=False,
            ),
            rows_size(len(baseline_rows(table)), per_row=0.3, width=12.0),
        )
    if "learning_curve" in tables:
        table = tables["learning_curve"]
        drawings["learning_curve"] = (
            paired(
                (
                    partial(plot_learning_curve, table=table, split="validation"),
                    partial(plot_learning_curve, table=table, split="test"),
                ),
                stacked=True,
            ),
            (7.0, 10.0),
        )
    if "univariate" in tables:
        table = tables["univariate"]
        drawings["univariate_auc"] = (
            one(partial(plot_univariate, table=table)),
            rows_size(len(strongest_features(table)), per_row=0.3, width=8.0),
        )
    if "drift" in tables:
        table = tables["drift"]
        drawings["drift"] = (
            one(partial(plot_drift, table=table)),
            rows_size(len(strongest_shifts(table)), per_row=0.3, width=8.0),
        )
    if "subgroups" in tables:
        table = tables["subgroups"]
        drawings["ap_concentration"] = (one(partial(plot_ap_concentration, table=table)), PANEL)
        if (_metric(table, "rings").value > 0).any():
            drawings["ring_coverage"] = (one(partial(plot_ring_coverage, table=table)), PANEL)
    if "proxy_validity" in tables:
        table = tables["proxy_validity"]
        drawings["proxy_validity"] = (one(partial(plot_proxy_validity, table=table)), PANEL)
    if "reveal_spread" in tables:
        reveal = files.record.get("reveal", {})
        drawings["reveal_spread"] = (
            one(
                partial(
                    plot_reveal_spread,
                    table=tables["reveal_spread"],
                    budget=reveal.get("budget"),
                    salt=reveal.get("salt"),
                )
            ),
            PANEL,
        )
    if "nnpu_simulation" in tables:
        table = tables["nnpu_simulation"]
        drawings["nnpu_simulation"] = (one(partial(plot_nnpu_simulation, table=table)), PANEL)
    return drawings


def write_diagnostics_report(study: DiagnosticsPaths) -> dict[str, Any]:
    """Draw a diagnostic study's figures from its tables, then its report.md.

    What `mule diagnose` writes after its analyses (diagnostics.study), and what
    `mule report` redraws offline from a diagnostics directory.
    """
    files = study_files(study)
    return draw(study, study_drawings(files), lambda: write_study_text(study, files))


def _metric(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    return table[table.metric == metric]


def _value(frame: pd.DataFrame, **keys: Any) -> float | None:
    """The one value of a long table's rows with these keys, None if there is none."""
    chosen = frame
    for column, value in keys.items():
        chosen = chosen[chosen[column] == value]
    return None if chosen.empty else float(chosen.value.iloc[0])


def features_section(features: pd.DataFrame) -> list[str]:
    """report.md's lines on the feature table: each split's sample."""
    rows = []
    for split, part in features.groupby("split", sort=False):
        mules = part.is_mule.eq(1)
        weight = part.weight.to_numpy(np.float64)
        prevalence = float(weight[mules.to_numpy()].sum() / weight.sum())
        rows.append(
            [
                str(split),
                str(part.date.iloc[0]),
                number(len(part)),
                number(int(mules.sum())),
                number(int((mules & part.revealed.astype(bool)).sum())),
                number(int(part.rejected.astype(bool).sum())),
                number(prevalence),
            ]
        )
    header = ["Split", "Cutoff", "Accounts", "Mules", "Revealed", "Rejected", "Prevalence"]
    return [
        "## Feature table",
        "",
        "Each split's sample is its audit sample: every mule and a uniform sample of the "
        "other accounts, each weighted to the split's population. Prevalence is weighted.",
        "",
        *table(header, rows),
        "",
    ]


def baselines_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the baselines: AP and ROC AUC on validation and test."""
    rows = []
    keys = [*baseline_rows(frame), ("chance (a random ranking)", "chance", "", "")]
    for label, baseline, features, model in keys:
        mine = frame[
            (frame.baseline == baseline) & (frame.features == features) & (frame.model == model)
        ]
        cells = [label]
        for split in HELD_OUT_SPLITS:
            for metric in ("average_precision", "roc_auc"):
                found = mine[(mine.split == split) & (mine.metric == metric)]
                if found.empty:
                    cells.append("")
                    continue
                row = found.iloc[0]
                bounds = None if pd.isna(row.low) else [float(row.low), float(row.high)]
                cells.append(estimate(float(row.value), bounds))
        rows.append(cells)
    header = ["Ranking", "Validation AP", "Validation ROC AUC", "Test AP", "Test ROC AUC"]
    return [
        "## Baselines",
        "",
        "How well does a table of the account's own activity rank mules, with no neighbour, "
        "association or pool input (the question of the retired no_graph control)? The PU "
        "baselines are fitted at the train cutoff on the revealed train mules, against every "
        "other sampled train account weighted to the population; `account` reads only the "
        "account's own history, which no model reads, `model` the root's own model inputs, "
        "`messages` its candidate pool, and `all` everything. In parentheses: the "
        f"ring-clustered {INTERVAL:.0%} interval.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"baselines": DIAGNOSTICS_FIGURES["baselines"]}),
    ]


def curve_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the learning curve: the mean AP and ROC AUC of each count."""
    rows = []
    learners = [m for m in ("lr", "hgb", "model") if (frame.model == m).any()]
    for model in learners:
        mine = frame[frame.model == model]
        for labels, mules in sorted({*zip(mine["labels"], mine.mules.astype(int), strict=True)}):
            part = mine[(mine["labels"] == labels) & (mine.mules == mules)]
            cells = [model, str(labels), number(mules), number(int(part.repeat.nunique()))]
            for split in HELD_OUT_SPLITS:
                for metric in ("average_precision", "roc_auc"):
                    values = part[(part.split == split) & (part.metric == metric)].value
                    cells.append(number(float(values.mean())) if len(values) else "")
            rows.append(cells)
    header = ["Learner", "Labels", "Train mules", "Draws", "Validation AP", "Validation ROC AUC"]
    header += ["Test AP", "Test ROC AUC"]
    return [
        "## Learning curve",
        "",
        "A learner fitted at the train cutoff on k train mules with oracle labels, drawn at "
        "random, against every sampled train non-mule (the mean of the draws), beside the "
        "same learner on the revealed train mules alone and the run's audit.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"learning_curve": DIAGNOSTICS_FIGURES["learning_curve"]}),
    ]


def univariate_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on each feature alone: the strongest on validation."""
    auc = _metric(frame, "roc_auc").pivot_table(index="feature", columns="split", values="value")
    ap = _metric(frame, "average_precision")
    rows = []
    for name in strongest_features(frame, REPORTED_ROWS):
        cells = [f"`{name}`"]
        cells += [
            number(value_or_none(auc.loc[name].get(split))) for split in ("train", *HELD_OUT_SPLITS)
        ]
        cells.append(number(_value(ap, feature=name, split="validation")))
        rows.append(cells)
    header = ["Feature", "Train ROC AUC", "Validation ROC AUC", "Test ROC AUC", "Validation AP"]
    return [
        "## Each feature alone",
        "",
        f"The {REPORTED_ROWS} features whose validation ROC AUC is farthest from 0.5. The AP "
        "ranks by the value in the direction the train split gives it.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"univariate_auc": DIAGNOSTICS_FIGURES["univariate_auc"]}),
    ]


def drift_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on drift: the largest shifts, and what the shift costs."""
    shifts = frame[frame.feature != ""]
    rows = []
    for name in strongest_shifts(frame, REPORTED_ROWS):
        mine = shifts[shifts.feature == name]
        rows.append(
            [
                f"`{name}`",
                number(_value(mine, split="validation", metric="smd")),
                number(_value(mine, split="test", metric="smd")),
                number(_value(mine, split="test", metric="shift_auc")),
                number(_value(mine, split="test", metric="above_train_q90")),
            ]
        )
    cost = frame[frame.feature == ""]
    cost_rows = []
    for (model, setup), part in cost.groupby(["model", "setup"], sort=False):
        cells = [str(model), str(setup)]
        for split in HELD_OUT_SPLITS:
            for metric in ("average_precision", "roc_auc"):
                cells.append(number(_value(part, split=split, metric=metric)))
        cost_rows.append(cells)
    return [
        "## Drift",
        "",
        "Each split is read at its own cutoff, so its accounts have seen more history. The "
        "standardised mean difference of a held-out split's non-mules from train's, the ROC "
        "AUC of telling the two apart by the value (0.5 means no shift) and the share above "
        "train's 90th percentile (10% without shift):",
        "",
        *table(
            ["Feature", "Validation SMD", "Test SMD", "Test shift AUC", "Test above train q90"],
            rows,
        ),
        "",
        "What the shift costs a learner fitted with every train mule labelled: at the train "
        "cutoff on raw values, on each split's own percentiles, and fitted inside each "
        "held-out split by cross-validation, which no shift touches:",
        "",
        *table(
            ["Learner", "Setup", "Validation AP", "Validation ROC AUC", "Test AP", "Test ROC AUC"],
            cost_rows,
        ),
        "",
        *figure_links(home, {"drift": DIAGNOSTICS_FIGURES["drift"]}),
    ]


def subgroups_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the run's revealed and hidden mules and its rings."""
    budgets = [budget_name(f) for f in REVIEW_BUDGETS]
    rows = []
    for split in HELD_OUT_SPLITS:
        mine = frame[frame.split == split]
        for subset in ("revealed", "hidden"):
            part = mine[mine.subset == subset]
            if part.empty:
                continue
            rows.append(
                [
                    split,
                    subset,
                    number(_int(_value(part, metric="mules"))),
                    number(_value(part, metric="roc_auc")),
                    " / ".join(number(_int(_value(part, metric=f"in_top_{b}"))) for b in budgets),
                    number(_int(_value(part, metric="median_population_rank"))),
                ]
            )
    ring_rows = []
    for split in HELD_OUT_SPLITS:
        rings = frame[(frame.split == split) & (frame.subset == "rings")]
        if rings.empty:
            continue
        ring_rows.append(
            [
                split,
                number(_int(_value(rings, metric="rings"))),
                " / ".join(number(_value(rings, metric=f"coverage_at_{b}")) for b in budgets),
            ]
        )
    shares = " / ".join(share_label(f) for f in REVIEW_BUDGETS)
    lines = [
        "## Revealed and hidden mules, AP concentration and rings",
        "",
        "From the run's audit samples. A ranking that finds the revealed mules and not the "
        "hidden ones measures the reveal, not mule detection. The population rank is the "
        "estimated number of non-mules scoring at least as high.",
        "",
        *table(
            [
                "Split",
                "Mules",
                "Number",
                "ROC AUC against the non-mules",
                f"In the top {shares}",
                "Median population rank",
            ],
            rows,
        ),
        "",
    ]
    if ring_rows:
        lines += [
            *table(["Split", "Rings", f"Rings with a member in the top {shares}"], ring_rows),
            "",
        ]
    figures = {name: DIAGNOSTICS_FIGURES[name] for name in ANALYSIS_FIGURES["subgroups"]}
    return [*lines, *figure_links(home, figures)]


def proxy_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the proxy predictions against the ground truth."""
    rows = []
    for split in HELD_OUT_SPLITS:
        for subset in ("all", "hidden", "revealed"):
            part = frame[(frame.split == split) & (frame.subset == subset)]
            if part.empty:
                continue
            rows.append(
                [
                    split,
                    subset,
                    number(_int(_value(part, metric="n"))),
                    number(_int(_value(part, metric="positives"))),
                    number(_value(part, metric="average_precision")),
                    number(_value(part, metric="roc_auc")),
                ]
            )
    return [
        "## Proxy validity",
        "",
        "The run's proxy predictions (its observed positives and a sample of unlabelled "
        "accounts) scored against the ground truth, unweighted: all of them, the hidden "
        "mules against the non-mules, and the revealed mules against them.",
        "",
        *table(["Split", "Subset", "Accounts", "Mules", "AP", "ROC AUC"], rows),
        "",
        *figure_links(home, {"proxy_validity": DIAGNOSTICS_FIGURES["proxy_validity"]}),
    ]


def reveal_section(home: Reported, frame: pd.DataFrame, record: Mapping[str, Any]) -> list[str]:
    """report.md's lines on the reveal over salts: each split's spread."""
    reveal = record.get("reveal", {})
    salt = reveal.get("salt")
    rows = []
    for split in ("train", *HELD_OUT_SPLITS):
        mine = frame[frame.split == split]
        if mine.empty:
            continue
        cells = [split, number(_int(_value(mine, metric="mules")))]
        for metric in ("eligible", "revealed"):
            values = mine[mine.metric == metric].value.to_numpy(np.float64)
            low, middle, high = (f"{q:g}" for q in np.percentile(values, [5, 50, 95]))
            configured = _value(mine, metric=metric, salt=salt) if salt is not None else None
            cells += [f"{middle} ({low} to {high})", number(_int(configured))]
        rows.append(cells)
    header = ["Split", "Mules", "Discovered by the cutoff", f"Salt {salt}", "Revealed"]
    header.append(f"Salt {salt}")
    return [
        "## The label reveal over salts",
        "",
        f"The reveal's mirror replayed for {frame.salt.nunique():,} salts: the median over "
        "salts and, in parentheses, its 5th to 95th percentile, beside the configured salt's "
        f"outcome. The budget is {reveal.get('budget', 'n/a')} mules per split.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"reveal_spread": DIAGNOSTICS_FIGURES["reveal_spread"]}),
    ]


def nnpu_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the nnPU simulation: each weight's mean over seeds."""
    wide = frame.pivot_table(index=["positive_weight", "seed"], columns="metric", values="value")
    rows = []
    for weight in sorted({float(w) for w in wide.index.get_level_values("positive_weight")}):
        part = wide.xs(weight, level="positive_weight")
        rows.append(
            [
                f"{weight:g}",
                number(len(part)),
                number(float(part.test_roc_auc.mean())),
                number(float(part.test_average_precision.mean())),
                number(int(part.collapsed.sum())),
                number(float(part.labelled_positive_mean_score.mean())),
            ]
        )
    header = ["Positive weight", "Seeds", "Test ROC AUC", "Test AP", "Collapsed"]
    header.append("Labelled positives' mean score")
    return [
        "## The nnPU positive weight, simulated",
        "",
        "A synthetic problem of the dataset's proportions trained with the repository's "
        "loss (offline; the means over seeds). A run collapsed when its labelled positives "
        "score below 0.05 on average.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"nnpu_simulation": DIAGNOSTICS_FIGURES["nnpu_simulation"]}),
    ]


def _int(value: float | None) -> int | None:
    """A count a long table holds as a float, rounded to the account."""
    return None if value is None else round(value)


def study_text(study: DiagnosticsPaths, files: StudyFiles) -> str:
    """A study's report.md: what ran, then each analysis' tables with its figures."""
    record, tables = files.record, files.tables
    dataset = str(record.get("dataset_id", study.root.name))
    lines = [f"# Diagnostics of dataset `{dataset[:12]}`", ""]
    run = record.get("run")
    if run is not None:
        compared = record.get("run_compared")
        lines += [
            f"Compared with the run {run}."
            if compared
            else f"No run is compared: {run} was not trained on this dataset, or is missing.",
            "",
        ]
    lines += [
        "Ground truth chooses the samples and labels the rows, for analysis only; nothing "
        "here feeds a model. Decisions use the validation split; test is for reporting, not "
        "selection. The pool groups (pool_activity and pool_internal_inflows) were designed "
        "after reading test-split mules and the data generator's mule typology, so test "
        "results that depend on them are optimistic.",
        "",
    ]
    outcomes = record.get("analyses", {})
    if outcomes:
        rows = [
            [
                name,
                str(outcome.get("status")),
                str(outcome.get("finished", "")),
                str(outcome.get("reason", "")),
            ]
            for name, outcome in outcomes.items()
        ]
        lines += [*table(["Analysis", "Last outcome", "Finished (UTC)", "Reason"], rows), ""]
    if files.features is not None:
        lines += features_section(files.features)
    sections: list[tuple[str, Callable[[pd.DataFrame], list[str]]]] = [
        ("baselines", partial(baselines_section, study)),
        ("learning_curve", partial(curve_section, study)),
        ("univariate", partial(univariate_section, study)),
        ("drift", partial(drift_section, study)),
        ("subgroups", partial(subgroups_section, study)),
        ("proxy_validity", partial(proxy_section, study)),
        ("reveal_spread", lambda frame: reveal_section(study, frame, record)),
        ("nnpu_simulation", partial(nnpu_section, study)),
    ]
    for name, section in sections:
        if name in tables:
            lines += section(tables[name])
    return "\n".join(lines).rstrip("\n") + "\n"


def write_study_text(study: DiagnosticsPaths, files: StudyFiles) -> Path:
    """Replace the study's report.md with study_text."""
    with atomic_write(study.report) as pending:
        pending.write_text(study_text(study, files))
    return study.report
