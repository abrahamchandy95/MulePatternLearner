"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..paths import BASELINE_VARIANT, DATA_DIR, RunPaths
from ..training.checkpoint import completed_run, run_started
from ..training.trainer import train
from .connect import open_context_source
from .prepare import prepare_dataset

# The directory of the built-in run, results/baseline/seed-42/, which `mule train` writes.
BASELINE_RUN = RunPaths.of(BASELINE_VARIANT, DEFAULT_CONFIG.training.seed)


def train_run(
    output: RunPaths = BASELINE_RUN,
    *,
    config: RunConfig = DEFAULT_CONFIG,
    data: Path = DATA_DIR,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and write the run into output.

    `mule train` runs DEFAULT_CONFIG into BASELINE_RUN. The dataset is
    config's in data (pipeline.prepare.prepare_dataset). With ``resume`` (what the command
    passes) an interrupted run continues from its resume.pt, and a complete run of the
    same settings is reported from its metrics.json before anything connects or is
    written; a complete run of other settings is an error that names them.
    """
    if not resume and run_started(output):
        raise FileExistsError(f"Run already exists: {output.root}; pass resume=True")
    if resume:
        recorded = completed_run(config, output)
        if recorded is not None:
            return recorded
    dataset = prepare_dataset(config, data)
    return train(config, dataset, output, open_contexts=open_context_source, resume=resume)
