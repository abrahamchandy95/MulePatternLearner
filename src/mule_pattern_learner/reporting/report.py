"""`mule report`: redraw the figures and report.md of a run, a suite or a diagnostic study.

Each report reads only the files its directory holds: reporting.run_report a run's,
reporting.suite_report a control-experiment suite's and reporting.study_report a
diagnostic study's; reporting.document saves their figures. report_directory tells them
apart by the file each has: a suite's summary.csv, a study's study.json, else a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import DiagnosticsPaths, RunPaths, SuitePaths
from .run_report import report_run
from .study_report import write_diagnostics_report
from .suite_report import write_suite_report


def report_directory(directory: Path) -> dict[str, Any]:
    """`mule report`: redraw the suite's or study's report directory holds, else its run's."""
    suite = SuitePaths(directory)
    if suite.summary.exists():
        return write_suite_report(suite)
    study = DiagnosticsPaths(directory)
    if study.study.exists():
        return write_diagnostics_report(study)
    return report_run(RunPaths(directory))
