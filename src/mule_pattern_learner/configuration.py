"""Tracked TOML configuration and backward-compatible local JSON inputs."""

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


def load_config(path: Path, *, live: bool = False) -> dict[str, Any]:
    """Read a TOML or JSON table; `live=True` also applies the live training schema.

    The schema is opt-in because this loader also serves the offline temporal
    commands, whose configuration keys differ.
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
    if live:
        from mule_pattern_learner.temporal.live.config_schema import validate_config

        return validate_config(config)
    return config
