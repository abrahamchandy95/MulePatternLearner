"""Repository paths, and the run and dataset path policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPOSITORY_ROOT / "models/temporal/model.pt"


def dataset_path(config: dict[str, Any], output: Path = DEFAULT_MODEL) -> Path:
    """The prepared cache of a run: <run directory>/prepared.

    An explicit prepared_id instead names a shared cache under artifacts/temporal,
    for experiments that train several models on one preparation.
    """
    if config.get("prepared_id"):
        return REPOSITORY_ROOT / "artifacts/temporal" / config["prepared_id"]
    return output_paths(output)[1] / "prepared"


def output_paths(output: Path) -> tuple[Path, Path]:
    """A .pt output names the model; directory outputs retain the experiment API."""
    if output.suffix == ".pt":
        return output, output.with_name(output.stem + "_run")
    return output / "model.pt", output
