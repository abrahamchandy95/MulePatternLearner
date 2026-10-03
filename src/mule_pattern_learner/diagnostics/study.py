"""`mule diagnose`: the diagnostic study of a prepared dataset, one analysis at a time.

The study of a dataset lives in results/diagnostics/<dataset id>/ (paths.DiagnosticsPaths):
the feature table (features.parquet), one long table per analysis (<analysis>.csv),
study.json, which records what each analysis last did, its events.jsonl, and the
figures and report.md that reporting.study_report.write_diagnostics_report draws from them.

The analyses, in the order `mule diagnose` runs them all:
- features: the feature table (diagnostics.feature_table), read from the graph. A table
  read with this code's query texts and columns is kept, since the frozen source would
  give the same one again; delete features.parquet to read it anew.
- univariate, drift, baselines, learning-curve: analyses of the feature table, which
  they build first when it is missing or stale. The baselines and the curve add the
  run's audits when it has them.
- subgroups: the run's audit samples (`mule evaluate` writes them).
- proxy-validity: the run's proxy predictions against the graph's truth.
- reveal-spread: the reveal's inputs, read from the graph, replayed over salts.
- nnpu-simulation: offline.

The run compared with is the one the caller names (the built-in run for the command
line), and only if it was trained on this dataset. An analysis whose inputs are missing
(no such run, no audit) is skipped with its reason, and the study's status is then
incomplete. The graph is read only through a StudyReader, which pipeline.diagnose
builds, and ground truth only through its TruthReader, for analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Protocol

import pandas as pd

from ..artifacts import (
    read_audit_scores,
    read_feature_table,
    read_json,
    read_run_provenance,
    write_diagnostic_table,
    write_feature_table,
    write_json,
)
from ..config import RunConfig
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..data.contexts import ContextReader, close_source
from ..data.hub_registry import load_hub_registry
from ..data.manifest import read_manifest
from ..data.ports import ScopeReader
from ..evaluation.truth import TruthReader
from ..paths import RESULTS_DIR, DatasetPaths, DiagnosticsPaths, RunPaths
from ..reporting.study_report import write_diagnostics_report
from ..runtime.progress import emit, recording
from .baselines import baselines
from .drift import drift
from .feature_table import AnalyticsFetcher, build_feature_table, current
from .learning_curve import learning_curve
from .nnpu_simulation import nnpu_simulation
from .proxy_validity import proxy_validity
from .reveal_spread import reveal_spread
from .subgroups import subgroups
from .univariate import univariate

# Every analysis of `mule diagnose`, in the order it runs them.
ANALYSES = (
    "features",
    "univariate",
    "drift",
    "baselines",
    "learning-curve",
    "subgroups",
    "proxy-validity",
    "reveal-spread",
    "nnpu-simulation",
)
# What an analysis did: wrote its table, kept a current one, or was skipped.
WRITTEN, KEPT, SKIPPED = "written", "kept", "skipped"
COMPLETE, INCOMPLETE = "complete", "incomplete"


class StudyReader(Protocol):
    """What a study reads from the graph (pipeline.diagnose.TigerGraphStudyReader).

    The reads connect on first use. oracle is the reader of the ground truth, which
    reads it once; contexts a context source of the study's configuration on the
    dataset's frozen source, which the caller closes; analytics the analytics context
    fetcher, its queries installed; reveal_inputs the rows the reveal decided from, and
    reveal_parameters the reveal's parameters with apply off, which need no graph.
    """

    def oracle(self) -> TruthReader: ...
    def scope(self) -> ScopeReader: ...
    def contexts(self) -> ContextReader: ...
    def analytics(self) -> AnalyticsFetcher: ...
    def reveal_inputs(self) -> list[dict[str, Any]]: ...
    def reveal_parameters(self) -> dict[str, Any]: ...


class Skipped(Exception):
    """An analysis' inputs are missing; the message says which."""


def analyses(name: str | None) -> tuple[str, ...]:
    """The analyses `mule diagnose [ANALYSIS]` runs: all of them, or the one named."""
    if name is None:
        return ANALYSES
    if name not in ANALYSES:
        raise ValueError(f"Unknown analysis {name!r}; the analyses are {list(ANALYSES)}")
    return (name,)


def finished() -> dict[str, str]:
    """When an analysis finished, in UTC to the second, as study.json records it."""
    return {"finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run_name(run: RunPaths) -> str:
    """How the study names a run: <variant>/seed-<n>."""
    return f"{run.root.parent.name}/{run.root.name}"


@dataclass
class Study:
    """One `mule diagnose` call: its settings, inputs and the outcome of each analysis.

    ``run`` is the run compared with, when it can be (unusable_run says why not);
    ``frame`` is the feature table once this call has read or built it.
    """

    config: RunConfig
    dataset: DatasetPaths
    run: RunPaths
    paths: DiagnosticsPaths
    graph: StudyReader
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    frame: pd.DataFrame | None = None

    @property
    def unusable_run(self) -> str | None:
        """Why the run cannot be compared with this dataset, or None when it can."""
        if not self.run.config.exists():
            return f"{self.run.root} holds no run"
        trained = read_run_provenance(self.run.config).get("dataset_id")
        if trained != self.dataset.root.name:
            return f"{self.run.root} was trained on another dataset ({str(trained)[:12]})"
        return None

    def audits(self) -> dict[str, dict[str, Any]]:
        """The run's audit reports by audited split; none when the run is not comparable."""
        if self.unusable_run is not None:
            return {}
        return {
            split: read_json(self.run.audit_report(split))
            for split in HELD_OUT_SPLITS
            if self.run.audit_report(split).exists()
        }

    def features(self) -> pd.DataFrame:
        """The feature table: kept if current, else read from the graph and written."""
        if self.frame is not None:
            return self.frame
        plan = self.config.feature_plan()
        if self.paths.features.exists():
            found = read_feature_table(self.paths.features)
            if current(found, plan):
                self.frame = found
                self.outcomes["features"] = {"status": KEPT, "rows": len(found), **finished()}
                return found
        manifest = read_manifest(self.dataset)
        hubs = load_hub_registry(self.dataset, manifest)
        graph = self.graph
        truth, scope, analytics = graph.oracle().read(), graph.scope(), graph.analytics()
        contexts = graph.contexts()
        failed = True
        try:
            frame = build_feature_table(
                self.config,
                manifest,
                hubs,
                scope=scope,
                truth=truth,
                contexts=contexts,
                analytics=analytics,
            )
            failed = False
        finally:
            close_source(contexts, failed=failed)
        write_feature_table(self.paths.features, frame)
        self.frame = frame
        self.outcomes["features"] = {"status": WRITTEN, "rows": len(frame), **finished()}
        return frame

    def table(self, name: str) -> pd.DataFrame:
        """The table of one analysis other than the feature table, computed."""
        match name:
            case "univariate":
                return univariate(self.features())
            case "drift":
                return drift(self.features())
            case "baselines":
                return baselines(self.features(), run=run_name(self.run), audits=self.audits())
            case "learning-curve":
                return learning_curve(self.features(), audits=self.audits())
            case "subgroups":
                reason = self.unusable_run
                if reason is not None:
                    raise Skipped(reason)
                audited = [s for s in HELD_OUT_SPLITS if self.run.audit_report(s).exists()]
                if not audited:
                    raise Skipped(f"{self.run.root} has no audit; run `mule evaluate`")
                return subgroups(
                    {split: read_audit_scores(self.run.audit_scores(split)) for split in audited}
                )
            case "proxy-validity":
                reason = self.unusable_run
                if reason is not None:
                    raise Skipped(reason)
                if not self.run.metrics.exists():
                    raise Skipped(f"{self.run.root} is not complete: it has no metrics.json")
                return proxy_validity(self.run, self.graph.oracle().read())
            case "reveal-spread":
                graph = self.graph
                return reveal_spread(graph.reveal_inputs(), graph.reveal_parameters())
            case "nnpu-simulation":
                return nnpu_simulation()
            case other:
                raise ValueError(f"Unknown analysis {other!r}")

    def analyse(self, name: str) -> dict[str, Any]:
        """Run one analysis, write what it found, and say what it did."""
        started = time.monotonic()
        try:
            if name == "features":
                self.features()
                outcome = self.outcomes["features"]
            else:
                table = self.table(name)
                write_diagnostic_table(self.paths.table(name), name.replace("-", "_"), table)
                outcome = {"status": WRITTEN, "rows": len(table)}
        except Skipped as reason:
            outcome = {"status": SKIPPED, "reason": str(reason)}
        outcome = {**outcome, "seconds": round(time.monotonic() - started, 3), **finished()}
        self.outcomes[name] = outcome
        emit({"event": "diagnose", "analysis": name, **outcome})
        return outcome

    def record(self) -> dict[str, Any]:
        """study.json: the dataset, the run, the reveal's settings and every analysis' outcome.

        An analysis not run this time keeps the outcome a former call recorded.
        """
        former = read_json(self.paths.study) if self.paths.study.exists() else {}
        return {
            "dataset_id": self.dataset.root.name,
            "run": run_name(self.run),
            "run_compared": self.unusable_run is None,
            "reveal": {
                "salt": self.config.scope.reveal_salt,
                "budget": self.config.scope.reveal_per_split,
            },
            "analyses": {**former.get("analyses", {}), **self.outcomes},
        }


def diagnose(
    names: Sequence[str],
    *,
    config: RunConfig,
    dataset: DatasetPaths,
    run: RunPaths,
    graph: StudyReader,
    results: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run the named analyses of a prepared dataset's study, then draw its report.

    The study's directory is results/diagnostics/<dataset id>/. The lines the analyses
    print go to its events.jsonl, and study.json is written once they have run, or once
    one of them failed, recording those that ran; the figures and report.md come after
    every analysis, so a figure that fails loses no table.
    """
    paths = DiagnosticsPaths.of(dataset.root.name, results)
    study = Study(config, dataset, run, paths, graph)
    paths.root.mkdir(parents=True, exist_ok=True)
    with recording(paths.events):
        try:
            for name in names:
                study.analyse(name)
        finally:
            write_json(paths.study, study.record())
    skipped = {n: o["reason"] for n, o in study.outcomes.items() if o["status"] == SKIPPED}
    report = write_diagnostics_report(paths)
    return {
        "status": INCOMPLETE if skipped else COMPLETE,
        "directory": str(paths.root),
        "dataset_id": dataset.root.name,
        "run": run_name(run),
        "analyses": {name: study.outcomes[name] for name in names},
        **({"skipped": skipped} if skipped else {}),
        **report,
    }
