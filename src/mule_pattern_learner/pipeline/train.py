"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..paths import DATA_DIR, DEFAULT_MODEL, output_paths
from ..training.trainer import train
from .connect import open_context_source
from .prepare import prepare_live


def run(
    output: Path = DEFAULT_MODEL,
    *,
    config: RunConfig = DEFAULT_CONFIG,
    data: Path = DATA_DIR,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and save the selected model at output.

    `mule-temporal train` runs DEFAULT_CONFIG. The dataset is config's in data
    (pipeline.prepare.prepare_live). With ``resume`` (what the command passes) an
    interrupted run continues from its checkpoint_last.pt; a finished one is an error.
    """
    checkpoint, _ = output_paths(output)
    if not resume and checkpoint.exists():
        raise FileExistsError(f"Experiment already exists: {output}; pass resume=True")
    dataset = prepare_live(config, data)
    return train(config, dataset, output, open_contexts=open_context_source, resume=resume)
