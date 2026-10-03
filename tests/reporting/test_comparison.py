"""A suite's figures and report.md, drawn offline from the synthetic files of its runs."""

from __future__ import annotations

from pathlib import Path
import re
import struct

import numpy as np
import pytest

from mule_pattern_learner.experiments.tables import write_tables
from mule_pattern_learner.experiments.variants import BASELINE, VARIANTS
from mule_pattern_learner.paths import RunPaths, SuitePaths
from mule_pattern_learner.reporting.comparison import rank_correlation
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
    # Ranked by the mean validation audit AP: the variants' found shares order them here.
    ranked = re.findall(r"^\| (\d) \| (\S+) \|", text, flags=re.MULTILINE)
    assert ranked == [("1", "**baseline**"), ("2", "no_attention"), ("3", "prior_weight")]
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
