"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..paths import BASELINE_VARIANT, DATA_DIR, RunPaths
from ..training.checkpoint import check_resumable, completed_run, run_started
from ..training.trainer import train
from .connect import open_context_source
from .prepare import prepare_dataset

# The directory of the built-in run, results/baseline/seed-42/, which `mule train` writes.
BASELINE_RUN = RunPaths.of(BASELINE_VARIANT, DEFAULT_CONFIG.training.seed)


def train_run(
    output: RunPaths | None = None,
    *,
    config: RunConfig = DEFAULT_CONFIG,
    data: Path = DATA_DIR,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and write the run into output.

    `mule train` runs DEFAULT_CONFIG into BASELINE_RUN, which is also the output
    without ``output``, but only for a configuration of the built-in run's
    results-relevant settings: another run names its own directory
    (RunPaths.of(variant, seed)). The dataset is config's in data
    (pipeline.prepare.prepare_dataset). With ``resume`` (what the command
    passes) an interrupted run continues from its resume.pt, and a complete run of the
    same settings is reported from its metrics.json. Both are checked before anything
    connects or is written: a run of other settings, complete or not, is an error that
    names them.
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
            return recorded
        check_resumable(config, output)
    dataset = prepare_dataset(config, data)
    return train(config, dataset, output, open_contexts=open_context_source, resume=resume)
