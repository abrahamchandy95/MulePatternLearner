"""A run's figures and report.md, drawn offline from synthetic files of a complete run."""

from __future__ import annotations

from pathlib import Path
import re
import struct
import subprocess
import sys

import pytest

from mule_pattern_learner.paths import REPOSITORY_ROOT, RunPaths
from mule_pattern_learner.reporting import run_report
from mule_pattern_learner.reporting.run_report import (
    AUDIT_FIGURES,
    TRAINING_FIGURES,
    report_run,
    write_audit_report,
    write_training_report,
)
from mule_pattern_learner.reporting.style import DPI, PANEL, STACKED
from mule_pattern_learner.testing.builders import write_run_files

PNG = b"\x89PNG\r\n\x1a\n"


def png_size(path: Path) -> tuple[int, int]:
    """The width and height a PNG's header records."""
    data = path.read_bytes()
    assert data[:8] == PNG, path
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunPaths, dict[str, object]]:
    """A complete, audited run with every figure and report.md drawn once."""
    run = write_run_files(RunPaths(tmp_path_factory.mktemp("run") / "baseline" / "seed-42"))
    return run, report_run(run)


def test_every_figure_renders_to_a_png_at_150_dpi_under_its_fixed_name(
    reported: tuple[RunPaths, dict[str, object]],
) -> None:
    run, result = reported
    names = [*TRAINING_FIGURES, *AUDIT_FIGURES]
    assert names == [
        "training_objective",
        "training_corrections",
        "validation_ranking",
        "training_throughput",
        "proxy_precision_recall",
        "run_health",
        "audit_precision_recall",
        "audit_roc",
        "audit_capture",
        "audit_threshold",
        "audit_score_distribution",
        "audit_revealed_hidden",
    ]
    assert result["figures"] == [str(run.plots / f"{name}.png") for name in names]
    assert sorted(path.name for path in run.plots.iterdir()) == sorted(f"{n}.png" for n in names)
    for name in names:
        path = run.figure(name)
        assert path.stat().st_size > 20_000, name
        size = STACKED if name == "training_throughput" else PANEL
        assert png_size(path) == (round(size[0] * DPI), round(size[1] * DPI)), name


def test_report_md_holds_the_tables_and_links_every_figure_relatively(
    reported: tuple[RunPaths, dict[str, object]],
) -> None:
    run, result = reported
    assert result["report"] == str(run.report)
    text = run.report.read_text()
    assert text.startswith("# Run baseline/seed-42\n")
    links = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
    assert links == [f"plots/{name}.png" for name in [*AUDIT_FIGURES, *TRAINING_FIGURES]]
    assert all((run.root / link).exists() for link in links)
    # The audit's point estimates with their intervals, then the proxy's, as recorded.
    assert "| validation (decisions) | test (reporting) |" in text
    assert "| Population accounts | 47,120 | 47,749 |" in text
    assert "| Audit sample: mules / accounts | 38 / 2,038 | 40 / 2,040 |" in text
    assert re.search(r"\| Average precision \| 0\.\d{3} \(0\.\d+ to 0\.\d+\) \|", text)
    assert "| Observed positives / accounts | 11 / 2,011 | 12 / 2,012 |" in text
    assert "| Epochs run (selected) | 11 (5) |" in text


def test_training_and_evaluation_each_draw_their_own_figures(tmp_path: Path) -> None:
    run = write_run_files(RunPaths(tmp_path / "run"), audited=False)
    trained = write_training_report(run)
    assert trained["figures"] == [str(run.figure(name)) for name in TRAINING_FIGURES]
    text = run.report.read_text()
    assert "## Ground-truth audit" not in text and "## Training" in text
    # Without an audit, `mule evaluate`'s part draws nothing, and report.md stays the same.
    assert write_audit_report(run)["figures"] == []
    assert run.report.read_text() == text
    # A run with only its validation audit gets the figures of every split, not the test's.
    audited = write_run_files(RunPaths(tmp_path / "audited"))
    audited.audit_report("test").unlink()
    drawn = [Path(path).stem for path in write_audit_report(audited)["figures"]]
    assert drawn == ["audit_precision_recall", "audit_roc", "audit_capture"]
    assert "| validation (decisions) |\n" in audited.report.read_text()


def test_a_failing_figure_leaves_the_others_and_report_md_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = write_run_files(RunPaths(tmp_path / "run"), audited=False)

    def broken(*args: object) -> None:
        raise ValueError("no ink")

    monkeypatch.setattr(run_report, "plot_corrections", broken)
    with pytest.raises(RuntimeError, match=r"training_corrections: ValueError\('no ink'\)"):
        write_training_report(run)
    drawn = {path.stem for path in run.plots.iterdir()}
    assert drawn == set(TRAINING_FIGURES) - {"training_corrections"}
    assert "plots/training_corrections.png" not in run.report.read_text()
    assert "plots/training_objective.png" in run.report.read_text()
    # A redraw that fails leaves no older drawing of the figure for report.md to link.
    monkeypatch.undo()
    write_training_report(run)
    assert "plots/training_corrections.png" in run.report.read_text()
    monkeypatch.setattr(run_report, "plot_corrections", broken)
    with pytest.raises(RuntimeError, match="training_corrections"):
        write_training_report(run)
    assert not run.figure("training_corrections").exists()
    assert "plots/training_corrections.png" not in run.report.read_text()


def test_report_needs_a_complete_run_or_an_audit(tmp_path: Path) -> None:
    run = RunPaths(tmp_path / "empty")
    run.root.mkdir()
    with pytest.raises(ValueError, match="neither a complete run nor an audit"):
        report_run(run)
    assert list(run.root.iterdir()) == []


def test_drawing_never_imports_pyplot(tmp_path: Path) -> None:
    # In a fresh interpreter, so no other test's imports count.
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from mule_pattern_learner.paths import RunPaths\n"
        "from mule_pattern_learner.reporting.report import report_directory\n"
        "from mule_pattern_learner.testing.builders import write_run_files\n"
        f"report_directory(write_run_files(RunPaths(Path({str(tmp_path)!r}))).root)\n"
        "print(sorted(m for m in sys.modules if m.startswith(('matplotlib.pyplot', 'pylab'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    assert result.stdout.strip() == "[]"
