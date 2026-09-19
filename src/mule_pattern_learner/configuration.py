"""Tracked TOML configuration and backward-compatible local JSON inputs."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        with path.open("rb") as stream:
            return tomllib.load(stream)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            return value
    raise ValueError("Configuration must be a TOML table or a local JSON object")
