"""Where the repository is, so every command behaves the same from any working directory."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# The GSQL of the graph: its schema, and the queries the code installs.
GSQL_DIR = REPOSITORY_ROOT / "gsql"
