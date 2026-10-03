"""A control-experiment suite: train its variants over the seeds, audit them, compare them.

run_suite does, in order:
1. resolves the named suites and variants, the baseline always among them
   (variants.select);
2. validates every variant offline (check_variants): each seed's configuration, feature
   plan and model on the CPU, one dataset id for all, and a fingerprint of its own;
3. prepares the base run's dataset, as `mule train` does, on the suite's one session,
   which connects only once something needs the graph;
4. plans each run (plan_run): a complete run of the same settings and dataset is kept,
   an interrupted one resumes, and one whose settings or dataset differ is moved to
   results/archive/ (never deleted) and trained again; the plan is a `suite` event with
   an upper bound on the training time (time_bound);
5. trains the runs, seeds outer and variants inner, into results/<variant>/seed-<n>/;
6. audits validation and test for every complete run that lacks them, reading truth once
   (pipeline.evaluate.SharedTruth);
7. writes summary.csv and comparison.csv (experiments.tables) and the figures and
   report.md (reporting.suite_report.write_suite_report) under results/experiments/<suite>/.

An error of one run's own is recorded against it and the suite goes on. An outage
(tigergraph.executor.TigerGraphUnavailableError) stops the training and the audits,
since every later run would fail the same way; the tables and the report are still
written from the runs there are. An error while preparing the dataset, an outage
included, stops the suite before any run, as it would `mule train`. The result's status
is complete only when every run is trained and audited and none recorded an error. A
run whose step failed after its numbers were saved (a training figure, say) is complete
in the tables, with its error beside it, and the suite then fails, as `mule train` fails
when a figure does; the next suite keeps the run, and `mule report` redraws its figures.

The suite's own events (the plan, each run's step that finished or failed, the archives,
an outage) and those of the dataset's preparation are recorded in the suite's
events.jsonl; each run's go to the run's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ..artifacts import read_history, read_json, read_run_config, read_run_provenance
from ..config import DEFAULT_CONFIG, RunConfig
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..data.manifest import dataset_id
from ..model.build import build_model
from ..paths import DATA_DIR, RESULTS_DIR, RunPaths, SuitePaths, archived_run
from ..pipeline.connect import Session
from ..pipeline.evaluate import SharedTruth, evaluate_run
from ..pipeline.prepare import prepare_dataset
from ..pipeline.train import train_run
from ..reporting.suite_report import write_suite_report
from ..runtime.progress import emit, recording
from ..tigergraph.executor import TigerGraphUnavailableError, error_summary
from ..training.checkpoint import changed_settings, run_started
from ..training.trainer import check_limits
from .tables import COMPLETE, FAILED, STOPPED, SuiteRun, audited, write_tables
from .variants import SEEDS, Variant, select

# What the suite does with a run: keep a complete one, train a new one, resume an
# interrupted one, or move one of other settings or another dataset aside and train it.
KEEP, TRAIN, RESUME, ARCHIVE = "keep", "train", "resume", "archive"


@dataclass
class PlannedRun:
    """One run of a suite: its variant, seed, settings and directory, and what befalls it.

    ``differs`` names what an archived run had otherwise: the settings, or its dataset.
    """

    variant: Variant
    seed: int
    config: RunConfig
    paths: RunPaths
    action: str
    differs: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])

    def outcome(self, stopped: bool) -> SuiteRun:
        """The run's record for the tables: complete, failed on its own, or stopped.

        A run is complete once it is audited, even when a step of its own failed after its
        numbers were saved; its error is kept beside it.
        """
        if audited(self.paths):
            status = COMPLETE
        elif self.errors:
            status = FAILED
        else:
            status = STOPPED if stopped else FAILED
        error = "; ".join(self.errors) or None
        return SuiteRun(self.variant, self.seed, self.paths, status, error)


def check_variants(variants: Sequence[Variant], base: RunConfig, seeds: Sequence[int]) -> None:
    """Refuse a suite before any graph work unless each variant builds and names its own run.

    Every seed's configuration of a variant must validate, give a feature plan within the
    batch limits and build its model on the CPU; every variant must train on the base
    run's dataset (one dataset id), and no two variants may share a fingerprint. The
    error names the variant.
    """
    expected = dataset_id("", base)
    for variant in variants:
        for seed in seeds:
            try:
                config = variant.config(base, seed)
                plan = config.feature_plan()
                check_limits(config, plan)
                build_model(config.model, plan, config.sampler.fanouts[0])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Variant {variant.name} does not build: {error}") from error
            if dataset_id("", config) != expected:
                raise ValueError(
                    f"Variant {variant.name} changes dataset settings; every variant of a "
                    "suite trains on the base run's dataset"
                )
    for seed in seeds:
        named: dict[str, str] = {}
        for variant in variants:
            fingerprint = variant.config(base, seed).fingerprint()
            if fingerprint in named:
                raise ValueError(
                    f"Variants {named[fingerprint]} and {variant.name} have the same settings"
                )
            named[fingerprint] = variant.name


def plan_run(
    variant: Variant, seed: int, base: RunConfig, results: Path, dataset: str
) -> PlannedRun:
    """What the suite does with a variant's run of a seed, from the files it has.

    A run that has written nothing trains. One whose results-relevant settings differ
    from the variant's, or that trained on another dataset than ``dataset`` (its id), is
    archived and trained again; a complete one otherwise is kept and an interrupted one
    resumes.
    """
    config = variant.config(base, seed)
    paths = RunPaths.of(variant.name, seed, results)
    planned = PlannedRun(variant, seed, config, paths, TRAIN)
    if not run_started(paths):
        return planned
    if paths.config.exists():
        try:
            planned.differs = changed_settings(config, paths)
            recorded = read_run_provenance(paths.config).get("dataset_id")
        except (KeyError, ValueError) as error:
            planned.differs, recorded = [f"unreadable config.json: {error}"], None
        if recorded is not None and recorded != dataset:
            planned.differs.append(f"dataset {str(recorded)[:12]}")
    if planned.differs:
        planned.action = ARCHIVE
    else:
        planned.action = KEEP if paths.metrics.exists() else RESUME
    return planned


def archive(run: RunPaths, results: Path, moment: datetime) -> RunPaths:
    """Move a run to results/archive/<variant>/seed-<n>/<moment>/, a directory of its own."""
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    target, count = archived_run(run, stamp, results), 1
    while target.root.exists():
        count += 1
        target = archived_run(run, f"{stamp}-{count}", results)
    target.root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(run.root, target.root)
    return target


def latest_graph_history(results: Path) -> Path | None:
    """The history.csv most recently written by a graph model's run under results.

    A run whose config.json or history.csv this code cannot read is passed over, and so
    is a history with no timed interval: a run restarted without resume state rewrites
    its history.csv as a bare header until it logs again.
    """
    found = []
    for history in results.glob("*/seed-*/history.csv"):
        run = RunPaths(history.parent)
        try:
            architecture = read_run_config(run.config).model.architecture
            seconds = read_history(history).seconds_per_step.to_numpy(dtype=np.float64)
        except (KeyError, OSError, ValueError):
            continue
        if architecture == "tgat" and np.isfinite(seconds).any():
            found.append((history.stat().st_mtime, history))
    return max(found)[1] if found else None


def time_bound(runs: Sequence[PlannedRun], results: Path) -> dict[str, Any]:
    """An upper bound on the hours the runs still to train take, and where it comes from.

    Every run is taken to train all its epochs (early stopping ends most sooner) at the
    median seconds per step of the latest graph run's history.csv (latest_graph_history)
    that timed a step; the summary models take far less. None without such a history, or
    when an epoch has no fixed steps.
    """
    pending = [run for run in runs if run.action != KEEP]
    history = latest_graph_history(results)
    steps = [run.config.training.steps_per_epoch for run in pending]
    if history is None or any(step is None for step in steps):
        return {"bound_hours": None, "timed_from": None if history is None else str(history)}
    seconds = float(np.nanmedian(read_history(history).seconds_per_step))
    total = sum(
        run.config.training.epochs * (run.config.training.steps_per_epoch or 0) for run in pending
    )
    return {"bound_hours": round(total * seconds / 3600, 2), "timed_from": str(history)}


def is_outage(error: BaseException) -> bool:
    """Whether an error is, or was caused by, TigerGraph staying unavailable."""
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, TigerGraphUnavailableError):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def finished(run: PlannedRun, step: str) -> dict[str, Any]:
    """The `run_finished` event of a run's step (train or audit), with the numbers it saved.

    A trained run gives its best epoch, validation proxy AP and seconds (metrics.json),
    an audited one the AP of each audited split; a number the run has no file for is None.
    """
    record: dict[str, Any] = {
        "event": "run_finished",
        "variant": run.variant.name,
        "seed": run.seed,
        "step": step,
    }
    if step == "train":
        metrics = read_json(run.paths.metrics) if run.paths.metrics.exists() else {}
        proxy = metrics.get("observed_label_proxy", {}).get("validation", {})
        record |= {
            "best_epoch": metrics.get("best_epoch"),
            "validation_proxy_ap": proxy.get("average_precision"),
            "seconds": metrics.get("elapsed_seconds"),
        }
    else:
        for split in HELD_OUT_SPLITS:
            report = run.paths.audit_report(split)
            metrics = read_json(report)["metrics"] if report.exists() else {}
            record[f"{split}_ap"] = metrics.get("average_precision")
    return record


def attempt(run: PlannedRun, step: str, work: Callable[[], object]) -> str | None:
    """Do one step of a run (train or audit); the summary of an outage that stopped it.

    A step that finished is a `run_finished` event. Any other error is the run's own: it
    is recorded against the run, and the suite goes on.
    """
    try:
        work()
    except Exception as error:  # an error of this run's own, or an outage
        summary = error_summary(error)
        where = {"variant": run.variant.name, "seed": run.seed, "step": step, "error": summary}
        if is_outage(error):
            emit({"event": "suite_stopped", **where})
            return summary
        run.errors.append(f"{step}: {summary}")
        emit({"event": "run_failed", **where})
    else:
        emit(finished(run, step))
    return None


def run_suite(
    names: Sequence[str] = (),
    *,
    base: RunConfig = DEFAULT_CONFIG,
    seeds: Sequence[int] = SEEDS,
    results: Path = RESULTS_DIR,
    data: Path = DATA_DIR,
) -> dict[str, Any]:
    """Train, audit and compare the suites and variants names choose (see the module).

    ``base`` is the run the variants change, the built-in run for the experiments
    script; ``seeds``, ``results`` and ``data`` are the script's too unless a test gives
    others. Returns the suite's directory, its status, every run's action, status and
    errors, and the files written.
    """
    suite_name, variants = select(names)
    check_variants(variants, base, seeds)
    suite = SuitePaths.of(suite_name, results)
    suite.root.mkdir(parents=True, exist_ok=True)
    with recording(suite.events):
        return _run(suite, variants, base=base, seeds=seeds, results=results, data=data)


def _run(
    suite: SuitePaths,
    variants: Sequence[Variant],
    *,
    base: RunConfig,
    seeds: Sequence[int],
    results: Path,
    data: Path,
) -> dict[str, Any]:
    """run_suite's steps once the suite is chosen and checked, its events recorded."""
    suite_name = suite.root.name
    session = Session(base.transport)
    dataset = prepare_dataset(base, data, session=session)
    runs = [
        plan_run(variant, seed, base, results, dataset.root.name)
        for seed in seeds
        for variant in variants
    ]
    emit(
        {
            "event": "suite",
            "suite": suite_name,
            "dataset_id": dataset.root.name,
            "runs": {
                variant.name: {str(r.seed): r.action for r in runs if r.variant is variant}
                for variant in variants
            },
            **time_bound(runs, results),
        }
    )
    stopped: str | None = None
    for run in (run for run in runs if run.action != KEEP):
        if run.action == ARCHIVE:
            moved = archive(run.paths, results, datetime.now(timezone.utc))
            emit(
                {
                    "event": "run_archived",
                    "run": str(run.paths.root),
                    "archive": str(moved.root),
                    "differs": run.differs,
                }
            )
        options = {"config": run.config, "data": data, "resume": True, "session": session}
        stopped = attempt(run, "train", partial(train_run, run.paths, **options))
        if stopped is not None:
            break
    truth = SharedTruth(session)
    for run in runs:
        if stopped is not None:
            break
        if run.paths.metrics.exists() and not audited(run.paths):
            audit = partial(evaluate_run, run.paths, truth=truth, data=data, session=session)
            stopped = attempt(run, "audit", audit)
    outcomes = [run.outcome(stopped is not None) for run in runs]
    tables = write_tables(suite, outcomes, base)
    report = write_suite_report(suite)
    if stopped is not None:
        status = STOPPED
    else:
        clean = all(o.status == COMPLETE and o.error is None for o in outcomes)
        status = COMPLETE if clean else FAILED
    return {
        "suite": suite_name,
        "directory": str(suite.root),
        "status": status,
        "stopped_by": stopped,
        "dataset_id": dataset.root.name,
        "runs": [
            {
                "variant": o.variant.name,
                "seed": o.seed,
                "action": run.action,
                "status": o.status,
                "error": o.error,
            }
            for run, o in zip(runs, outcomes, strict=True)
        ],
        **tables,
        **report,
    }
