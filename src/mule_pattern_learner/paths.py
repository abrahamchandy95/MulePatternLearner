"""Repository paths, the run and dataset path policy, and the TOML/JSON reader of run_config."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML or JSON table as written, without any schema.

    Training reads override files through config.run_config, which
    merges them into the built-in settings and validates the result.
    """
    if path.suffix.lower() == ".toml":
        with path.open("rb") as stream:
            value: object = tomllib.load(stream)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    else:
        value = None
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a TOML table or a local JSON object")
    config: dict[str, Any] = value
    return config


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
