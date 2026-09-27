"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, RunConfig
from ..paths import DEFAULT_MODEL, dataset_path, output_paths
from ..training.trainer import train
from .connect import open_context_source
from .prepare import prepare_live


def run(
    output: Path = DEFAULT_MODEL,
    *,
    config: RunConfig = DEFAULT_CONFIG,
    dataset: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and save the selected model at output.

    `mule-temporal train` runs DEFAULT_CONFIG. With ``resume`` (what the command
    passes) an interrupted run continues from its checkpoint_last.pt; a finished one is
    an error. An explicit ``dataset`` is an existing prepared dataset, trained on as it
    is; its settings must match the configuration's.
    """
    checkpoint, _ = output_paths(output)
    if not resume and checkpoint.exists():
        raise FileExistsError(f"Experiment already exists: {output}; pass resume=True")
    if dataset is None:
        dataset = dataset_path(output)
        prepare_live(config, dataset)
    return train(config, dataset, output, open_contexts=open_context_source, resume=resume)
