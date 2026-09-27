"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..paths import BASELINE_VARIANT, DATA_DIR, RunPaths
from ..training.checkpoint import run_started
from ..training.trainer import train
from .connect import open_context_source
from .prepare import prepare_live

# The directory of the built-in run, results/baseline/seed-42/, which `mule train` writes.
BASELINE_RUN = RunPaths.of(BASELINE_VARIANT, DEFAULT_CONFIG.training.seed)


def run(
    output: RunPaths = BASELINE_RUN,
    *,
    config: RunConfig = DEFAULT_CONFIG,
    data: Path = DATA_DIR,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and write the run into output.

    `mule-temporal train` runs DEFAULT_CONFIG into BASELINE_RUN. The dataset is
    config's in data (pipeline.prepare.prepare_live). With ``resume`` (what the command
    passes) an interrupted run continues from its resume.pt; a finished one is an error.
    """
    if not resume and run_started(output):
        raise FileExistsError(f"Run already exists: {output.root}; pass resume=True")
    dataset = prepare_live(config, data)
    return train(config, dataset, output, open_contexts=open_context_source, resume=resume)
