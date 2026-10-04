"""A suite's figures and report.md, drawn offline from the synthetic files of its runs."""

from __future__ import annotations

from pathlib import Path
import re
import struct

from matplotlib.figure import Figure
import numpy as np
import pytest

from mule_pattern_learner.experiments.tables import write_tables
from mule_pattern_learner.experiments.variants import BASELINE, VARIANTS
from mule_pattern_learner.paths import RunPaths, SuitePaths
from mule_pattern_learner.reporting.comparison import (
    VariantSeeds,
    plot_comparison,
    plot_paired_delta,
    rank_correlation,
)
from mule_pattern_learner.reporting.report import report_directory
from mule_pattern_learner.reporting.style import DPI
from mule_pattern_learner.reporting.suite_report import SUITE_FIGURES
from mule_pattern_learner.testing.builders import write_run_files, write_suite_runs

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory) -> tuple[SuitePaths, dict[str, object]]:
    """A synthetic suite of three variants over two seeds, compared and reported once."""
    suite = SuitePaths.of("demo", tmp_path_factory.mktemp("results"))
    variants = [BASELINE, VARIANTS["prior_weight"], VARIANTS["no_attention"]]
    found = {"baseline": 0.62, "no_attention": 0.5, "prior_weight": 0.2}
    write_tables(suite, write_suite_runs(suite, variants, (42, 43), found))
    return suite, report_directory(suite.root)


def test_every_suite_figure_renders_to_a_png_under_its_fixed_name(
    reported: tuple[SuitePaths, dict[str, object]],
) -> None:
    suite, result = reported
    assert list(SUITE_FIGURES) == [
        "comparison_ap",
        "comparison_delta",
        "comparison_budget",
        "comparison_capture",
        "comparison_validation",
        "comparison_proxy_vs_audit",
    ]
    assert result["figures"] == [str(suite.figure(name)) for name in SUITE_FIGURES]
    assert sorted(p.name for p in suite.plots.iterdir()) == sorted(
        f"{n}.png" for n in SUITE_FIGURES
    )
    for name in SUITE_FIGURES:
        data = suite.figure(name).read_bytes()
        assert data[:8] == PNG and len(data) > 20_000, name
        width, _ = struct.unpack(">II", data[16:24])
        # Every figure is at least as wide as a panel at 150 dpi.
        assert width >= 7 * DPI, name


def test_the_suite_report_ranks_by_validation_and_keeps_test_for_reporting(
    reported: tuple[SuitePaths, dict[str, object]],
) -> None:
    suite, result = reported
    assert result["report"] == str(suite.report)
    text = suite.report.read_text()
    assert text.startswith("# Suite demo\n\n3 variants with the seeds 42, 43: 6 runs, 6 complete.")
    assert "## Validation audit, for decisions" in text
    assert "## Test audit, for reporting, not selection" in text
    assert "pool groups (pool_activity and pool_internal_inflows) were designed after" in text
    # Ranked by the mean validation audit AP: the variants' found shares order them here,
    # and their seed ensembles in the same order.
    validation, rest = text.split("## Seed ensembles\n")
    ensembles, _ = rest.split("## Test audit")
    expected = [("1", "**baseline**"), ("2", "no_attention"), ("3", "prior_weight")]
    for section in (validation, ensembles):
        assert re.findall(r"^\| (\d) \| (\S+) \|", section, flags=re.MULTILINE) == expected
    assert "averaged on the log-odds scale" in ensembles
    # Two comparisons with the baseline, each over both sources of uncertainty, with the
    # audit-only interval and the seeds that agree beside it.
    assert "The suite makes 2 comparisons with the baseline, so at 90% about 0.2" in validation
    assert "| Delta from the baseline | Audit-only interval | Seeds that agree |" in validation
    assert len(re.findall(r"\| \d of 2 \|", validation)) == 2
    # The proxy's reliability, each correlation with its n.
    reliability = ensembles.split("## Proxy reliability\n")[1]
    assert "In real use only the proxy exists" in reliability
    for label, n in (("runs", 6), ("variants, by their seed means", 3)):
        assert re.search(rf"^\| {label} \| -?\d\.\d\d \| {n} \|$", reliability, re.M), label
    assert "| runs selected on validation_ap |" in reliability
    assert "| 1 | **baseline** | 42 43 |" in ensembles
    links = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
    assert links == [f"plots/{name}.png" for name in SUITE_FIGURES]
    assert all((suite.root / link).exists() for link in links)


def test_report_redraws_a_run_directory_as_a_run(tmp_path: Path) -> None:
    run = write_run_files(RunPaths(tmp_path / "baseline" / "seed-42"))
    result = report_directory(run.root)
    assert result["report"] == str(run.report)
    assert run.report.read_text().startswith("# Run baseline/seed-42\n")


def test_rank_correlation_is_spearmans_with_tied_ranks_shared() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_correlation(x, x**3) == pytest.approx(1.0)
    assert rank_correlation(x, -x) == pytest.approx(-1.0)
    # Ranks 1, 2.5, 2.5, 4 against 1, 2, 3, 4.
    tied = rank_correlation(x, np.array([1.0, 5.0, 5.0, 9.0]))
    assert tied == pytest.approx(np.corrcoef([0, 1.5, 1.5, 3], [0, 1, 2, 3])[0, 1])
    assert rank_correlation(x[:2], x[:2]) is None
    assert rank_correlation(x, np.ones(4)) is None


def test_the_comparison_draws_each_seed_ensemble_below_its_mean() -> None:
    ax = Figure().add_subplot()
    rows = [
        VariantSeeds(
            "baseline",
            {42: 0.2, 43: 0.4},
            0.3,
            (0.2, 0.4),
            ensemble=0.5,
            ensemble_interval=(0.45, 0.6),
        ),
        VariantSeeds("prior_weight", {42: 0.1}, 0.1, None),
    ]
    plot_comparison(ax, rows, split="validation", baseline=0.3)
    # The baseline's row is on top (y = 1); its ensemble is a diamond below it.
    diamonds = [line for line in ax.get_lines() if line.get_marker() == "D"]
    points = [
        np.ravel(np.asarray([line.get_xdata(), line.get_ydata()], dtype=float)).tolist()
        for line in diamonds
    ]
    assert points == [[0.5, pytest.approx(0.7)]]
    legend = ax.get_legend()
    assert legend is not None
    assert "seed ensemble, its interval" in [text.get_text() for text in legend.get_texts()]
    # Without an ensemble the legend does not name one.
    ax = Figure().add_subplot()
    plot_comparison(ax, rows[1:], split="validation", baseline=None)
    legend = ax.get_legend()
    assert legend is not None
    assert "seed ensemble, its interval" not in [t.get_text() for t in legend.get_texts()]


def test_the_delta_figure_draws_both_intervals() -> None:
    ax = Figure().add_subplot()
    row = VariantSeeds(
        "prior_weight",
        {42: -0.3, 43: -0.1},
        -0.2,
        (-0.35, -0.05),
        consistent=True,
        audit_interval=(-0.25, -0.15),
    )
    plot_paired_delta(ax, [row])
    lines = [
        np.ravel(np.asarray([line.get_xdata(), line.get_ydata()], dtype=float)).tolist()
        for line in ax.get_lines()
    ]
    # The two-source interval on the row, the audit-only one a little above it.
    assert [-0.35, -0.05, 0.0, 0.0] in lines and [-0.25, -0.15, 0.25, 0.25] in lines
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()][3:5] == [
        "90% interval over seeds and accounts",
        "over the audit's accounts alone",
    ]
