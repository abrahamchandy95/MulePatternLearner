"""Prepare if needed, then train with the built-in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import run_config
from ..data.manifest import read_manifest
from ..paths import DEFAULT_MODEL, dataset_path, output_paths
from ..training.trainer import train
from .prepare import prepare_live


def prepared_config(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The run settings with the identity preparation resolved (dataset_id).

    A pinned dataset_id is kept, so preparation_mismatches reports it when it
    differs from the prepared dataset.
    """
    return {**config, "dataset_id": config.get("dataset_id") or manifest["source"]["dataset_id"]}


def run(
    output: Path = DEFAULT_MODEL,
    *,
    config_path: Path | None = None,
    dataset: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare if needed, train with nnPU, and save the selected model at output.

    With ``resume`` (what `mule-temporal train` passes) an interrupted run continues
    from its checkpoint_last.pt; a finished one is an error.
    """
    checkpoint, _ = output_paths(output)
    if not resume and checkpoint.exists():
        raise FileExistsError(f"Experiment already exists: {output}; pass resume=True")
    config = run_config(config_path)
    if dataset is None:
        dataset = dataset_path(config, output)
        manifest = prepare_live(config, dataset)
    else:
        # Explicit datasets are immutable pre-existing caches, useful for experiments.
        manifest = read_manifest(dataset)
    return train(prepared_config(config, manifest), dataset, output, resume=resume)
