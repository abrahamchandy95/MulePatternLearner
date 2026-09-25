"""Repository paths and the raw TOML/JSON reader behind run_config."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path; relative paths are relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML or JSON table as written, without any schema.

    Live training reads override files through config_schema.run_config, which
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
