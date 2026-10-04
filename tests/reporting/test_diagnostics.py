"""A diagnostic study's figures, each on a Figure of its own, then its PNGs and report.md."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
import struct

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.paths import DiagnosticsPaths
from mule_pattern_learner.reporting import diagnostics as figures
from mule_pattern_learner.reporting.report import report_directory
from mule_pattern_learner.reporting.study_report import (
    ANALYSIS_FIGURES,
    DIAGNOSTICS_FIGURES,
    write_diagnostics_report,
)
from mule_pattern_learner.reporting.style import BASELINE, DPI, MUTED
from mule_pattern_learner.testing.builders import diagnostic_tables, write_study_files

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return diagnostic_tables()


def xs(line: Line2D) -> list[float]:
    """The x values of a drawn line."""
    return [float(value) for value in np.ravel(np.asarray(line.get_xdata()))]


def ys(line: Line2D) -> list[float]:
    """The y values of a drawn line."""
    return [float(value) for value in np.ravel(np.asarray(line.get_ydata()))]


def drawn(plot: Callable[[Axes], Axes]) -> Axes:
    """The Axes a plot function returns, drawn on a new Figure's one panel."""
    ax = Figure().add_subplot()
    assert plot(ax) is ax
    assert ax.get_title() and ax.get_xlabel()
    return ax


def test_the_features_are_ranked_on_validation_and_named_by_family(
    tables: dict[str, pd.DataFrame],
) -> None:
    table = tables["univariate"]
    # Ranked by the ROC AUC of validation's hidden mules.
    auc = table[table.metric == "hidden_roc_auc"].pivot_table(
        index="feature", columns="split", values="value"
    )
    distance = (auc.validation - 0.5).abs().sort_values(ascending=False)
    assert figures.strongest_features(table, 3) == distance.index[:3].tolist()
    ax = drawn(lambda ax: figures.plot_univariate(ax, table, count=4))
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == [figures.feature_label(name) for name in distance.index[:4]]
    assert figures.feature_label("account__30d_in_count") == "account: 30d_in_count"


def test_the_drift_shows_the_largest_shifts_on_a_symmetric_log_axis(
    tables: dict[str, pd.DataFrame],
) -> None:
    table = tables["drift"]
    smd = table[table.metric == "smd"]
    largest = smd.assign(size=smd.value.abs()).groupby("feature")["size"].max()
    expected = largest.sort_values(ascending=False).index[:3].tolist()
    assert figures.strongest_shifts(table, 3) == expected
    ax = drawn(lambda ax: figures.plot_drift(ax, table))
    assert ax.get_xscale() == "symlog"
    # Every point lies inside the axis, the largest shift included.
    low, high = ax.get_xlim()
    assert low < smd.value.min() and smd.value.max() < high


def test_the_baselines_mark_chance_at_the_prevalence_and_the_run_in_ink(
    tables: dict[str, pd.DataFrame],
) -> None:
    table = tables["baselines"]
    rows = figures.baseline_rows(table)
    assert rows[0] == ("the run: baseline/seed-42", "model", "baseline/seed-42", "audit")
    assert all(baseline != "chance" for _, baseline, _, _ in rows)
    for split in ("validation", "test"):
        ax = drawn(lambda ax, split=split: figures.plot_baselines(ax, table, split=split))
        assert ax.get_xscale() == "log" and split.capitalize() in ax.get_title()
        # Chance at the prevalence of the hidden mules, the AP the figure draws.
        chance = table[
            (table.baseline == "chance")
            & (table.split == split)
            & (table.metric == "hidden_average_precision")
        ].value.item()
        vertical = [line for line in ax.get_lines() if line.get_color() == MUTED]
        assert [xs(line)[0] for line in vertical] == [pytest.approx(chance)]
        ink = [line for line in ax.get_lines() if line.get_color() == BASELINE]
        assert ink, "the run's audit"
        assert len(ax.get_yticklabels()) == len(rows)
    bare = drawn(lambda ax: figures.plot_baselines(ax, table, split="test", labels=False))
    assert not any(label.get_visible() for label in bare.get_yticklabels() if label.get_text())
    # The intervals are named as the table has them: ring-clustered unless told otherwise.
    for ax, name in (
        (bare, "ring-clustered 90% interval"),
        (
            drawn(
                lambda ax: figures.plot_baselines(ax, table, split="test", interval="stratified")
            ),
            "stratified",
        ),
    ):
        legend = ax.get_legend()
        assert legend is not None and name in [text.get_text() for text in legend.get_texts()]


def test_the_learning_curve_marks_the_run_at_the_revealed_count(
    tables: dict[str, pd.DataFrame],
) -> None:
    table = tables["learning_curve"]
    for split in ("validation", "test"):
        ax = drawn(lambda ax, split=split: figures.plot_learning_curve(ax, table, split=split))
        assert ax.get_xscale() == "log"
        run = table[(table.model == "model") & (table.split == split)]
        run = run[run.metric == "hidden_average_precision"].iloc[0]
        ink = [line for line in ax.get_lines() if line.get_color() == BASELINE]
        assert {ys(line)[0] for line in ink} >= {run.value}
        assert {xs(line)[0] for line in ink} >= {run.mules}
    # The ROC AUC version marks chance, and the run's audit AUC.
    ax = drawn(lambda ax: figures.plot_learning_curve(ax, table, metric="roc_auc"))
    assert ax.get_ylabel() == "Test audit ROC AUC"
    assert [ys(line) for line in ax.get_lines() if line.get_color() == MUTED] == [[0.5, 0.5]]
    run = table[(table.model == "model") & (table.split == "test") & (table.metric == "roc_auc")]
    legend = ax.get_legend()
    assert legend is not None
    texts = [text.get_text() for text in legend.get_texts()]
    assert f"the run's audit: ROC AUC {run.value.iloc[0]:.3f}" in " ".join(texts)


def test_the_run_figures_draw_each_audited_split(tables: dict[str, pd.DataFrame]) -> None:
    concentration = drawn(lambda ax: figures.plot_ap_concentration(ax, tables["subgroups"]))
    legend = concentration.get_legend()
    assert legend is not None
    texts = [text.get_text() for text in legend.get_texts()]
    assert texts[0] == "validation" and any(text.startswith("test AP ") for text in texts)
    coverage = drawn(lambda ax: figures.plot_ring_coverage(ax, tables["subgroups"]))
    # Two splits, three review budgets: a bar each.
    assert len(coverage.patches) == 6
    proxy = drawn(lambda ax: figures.plot_proxy_validity(ax, tables["proxy_validity"]))
    assert len(proxy.patches) == 6
    labels = [text.get_text() for text in proxy.texts]
    assert len(labels) == 6 and all(text.startswith("AP ") for text in labels)


def test_the_reveal_spread_marks_the_configured_salt_and_the_budget(
    tables: dict[str, pd.DataFrame],
) -> None:
    table = tables["reveal_spread"]
    ax = drawn(lambda ax: figures.plot_reveal_spread(ax, table, budget=20, salt=42))
    assert "50 salts" in ax.get_title()
    names = [label.get_text() for label in ax.get_yticklabels()]
    assert names[0] == "train: discovered by the cutoff" and names[-1] == "test: revealed"
    diamonds = [line for line in ax.get_lines() if line.get_marker() == "D"]
    configured = table[table.salt == 42]
    expected = configured[configured.metric != "mules"].value.tolist()
    assert sorted(xs(line)[0] for line in diamonds) == sorted(expected)
    budget = [line for line in ax.get_lines() if xs(line) == [20.0, 20.0]]
    assert budget
    # Without a configured salt or budget, neither is drawn.
    plain = drawn(lambda ax: figures.plot_reveal_spread(ax, table, budget=None, salt=None))
    assert not [line for line in plain.get_lines() if line.get_marker() == "D"]


def test_the_nnpu_figure_labels_each_weight_with_its_collapsed_seeds(
    tables: dict[str, pd.DataFrame],
) -> None:
    table = tables["nnpu_simulation"]
    ax = drawn(lambda ax: figures.plot_nnpu_simulation(ax, table))
    wide = table.pivot_table(index=["positive_weight", "seed"], columns="metric", values="value")
    ticks = [label.get_text() for label in ax.get_xticklabels()]
    assert len(ticks) == wide.index.get_level_values(0).nunique()
    assert ticks[0].startswith("0.001 (the prior)\n") and "(balanced)" in ticks[-1]
    collapsed = wide.collapsed.groupby(level="positive_weight").sum().astype(int)
    seeds = wide.groupby(level="positive_weight").size()
    for tick, count, total in zip(ticks, collapsed, seeds, strict=True):
        assert tick.endswith(f"\n{count} of {total} collapsed")


@pytest.fixture(scope="module")
def reported(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[DiagnosticsPaths, dict[str, object]]:
    """A synthetic study's files, reported once."""
    study = DiagnosticsPaths.of("abcdef0123456789", tmp_path_factory.mktemp("results"))
    write_study_files(study)
    return study, write_diagnostics_report(study)


def test_every_study_figure_renders_to_a_png_under_its_fixed_name(
    reported: tuple[DiagnosticsPaths, dict[str, object]],
) -> None:
    study, result = reported
    assert list(DIAGNOSTICS_FIGURES) == [
        "baselines",
        "learning_curve",
        "univariate_auc",
        "drift",
        "ap_concentration",
        "ring_coverage",
        "proxy_validity",
        "reveal_spread",
        "nnpu_simulation",
    ]
    assert sorted(n for names in ANALYSIS_FIGURES.values() for n in names) == sorted(
        DIAGNOSTICS_FIGURES
    )
    assert result["figures"] == [str(study.figure(name)) for name in DIAGNOSTICS_FIGURES]
    for name in DIAGNOSTICS_FIGURES:
        data = study.figure(name).read_bytes()
        assert data[:8] == PNG and len(data) > 20_000, name
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 7 * DPI and height >= 4 * DPI, name
    # The paired figures are two panels wide or tall.
    width, _ = struct.unpack(">II", study.figure("baselines").read_bytes()[16:24])
    _, height = struct.unpack(">II", study.figure("learning_curve").read_bytes()[16:24])
    assert width >= 12 * DPI and height >= 10 * DPI


def test_the_study_report_has_each_analysis_with_its_figures(
    reported: tuple[DiagnosticsPaths, dict[str, object]],
) -> None:
    study, result = reported
    assert result["report"] == str(study.report)
    text = study.report.read_text()
    assert text.startswith("# Diagnostics of dataset `abcdef012345`\n\nCompared with the run")
    assert "pool groups (pool_activity and pool_internal_inflows) were designed after" in text
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert headings == [
        "Feature table",
        "Baselines",
        "Learning curve",
        "Each feature alone",
        "Drift",
        "Revealed and hidden mules, AP concentration and rings",
        "Proxy validity",
        "The label reveal over salts",
        "The nnPU positive weight, simulated",
    ]
    assert "| chance (a random ranking) |" in text and "no_graph" in text
    links = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
    assert links == [f"plots/{name}.png" for name in DIAGNOSTICS_FIGURES]
    # The reveal's counts are whole, and its configured salt's outcome is beside them.
    (row,) = re.findall(r"^\| train \| 160 \| .+$", text, flags=re.MULTILINE)
    assert ".000" not in row


def test_report_redraws_a_study_directory_from_the_tables_it_holds(tmp_path: Path) -> None:
    study = write_study_files(DiagnosticsPaths.of("abc", tmp_path))
    for name in ("subgroups", "nnpu_simulation"):
        study.table(name).unlink()
    result = report_directory(study.root)
    drawn = {Path(str(path)).stem for path in result["figures"]}
    assert drawn == set(DIAGNOSTICS_FIGURES) - {
        "ap_concentration",
        "ring_coverage",
        "nnpu_simulation",
    }
    assert "## Proxy validity" in study.report.read_text()
    assert "## The nnPU positive weight" not in study.report.read_text()
