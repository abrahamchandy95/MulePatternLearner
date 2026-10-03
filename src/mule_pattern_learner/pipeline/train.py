"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..data.contexts import ContextOpener, ContextSource
from ..paths import BASELINE_VARIANT, DATA_DIR, DatasetPaths, RunPaths
from ..reporting.run_report import write_training_report
from ..runtime.progress import emit
from ..training.checkpoint import check_resumable, completed_run, run_started
from ..training.trainer import train
from .connect import Session, open_context_source
from .prepare import prepare_dataset

# The directory of the built-in run, results/baseline/seed-42/, which `mule train` writes.
BASELINE_RUN = RunPaths.of(BASELINE_VARIANT, DEFAULT_CONFIG.training.seed)


def train_run(
    output: RunPaths | None = None,
    *,
    config: RunConfig = DEFAULT_CONFIG,
    data: Path = DATA_DIR,
    resume: bool = False,
    session: Session | None = None,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and write the run into output.

    `mule train` runs DEFAULT_CONFIG into BASELINE_RUN, which is also the output
    without ``output``, but only for a configuration of the built-in run's
    results-relevant settings: another run names its own directory
    (RunPaths.of(variant, seed)). The dataset is config's in data
    (pipeline.prepare.prepare_dataset). With ``resume`` (what the command
    passes) an interrupted run continues from its resume.pt, and a complete run of the
    same settings is reported from its metrics.json, with an `already_complete` event. Both are checked before anything
    connects or is written: a run of other settings, complete or not, is an error that
    names them. A run trained here then gets its training figures and report.md
    (reporting.run_report.write_training_report); a complete run reported is left as it was.
    With a ``session`` the dataset is prepared and the contexts are requested on its
    connection, which a suite of runs shares.
    """
    if output is None:
        if config.fingerprint() != DEFAULT_CONFIG.fingerprint():
            raise ValueError(
                f"Only the built-in run trains into {BASELINE_RUN.root}; name the directory of "
                "a run of other settings (RunPaths.of(variant, seed))"
            )
        output = BASELINE_RUN
    if not resume and run_started(output):
        raise FileExistsError(f"Run already exists: {output.root}; pass resume=True")
    if resume:
        recorded = completed_run(config, output)
        if recorded is not None:
            emit({"event": "already_complete", "run": str(output.root)})
            return recorded
        check_resumable(config, output)
    dataset = prepare_dataset(config, data, session=session)
    result = train(config, dataset, output, open_contexts=opener(session), resume=resume)
    # model.pt and every other file of the run are saved by now, so a figure that fails
    # loses nothing: the error comes after the other figures and report.md are written.
    write_training_report(output)
    return result


def opener(session: Session | None) -> ContextOpener:
    """The pipeline's context opener: open_context_source, on the session's connection if any."""
    if session is None:
        return open_context_source

    def open_on_session(
        dataset: DatasetPaths, manifest: dict[str, Any], config: RunConfig
    ) -> ContextSource:
        return open_context_source(dataset, manifest, config, session=session)

    return open_on_session
