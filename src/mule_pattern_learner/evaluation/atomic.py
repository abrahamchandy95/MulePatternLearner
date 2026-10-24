"""Files written whole or not at all (docs/design.md, "Results" and "Cache"): through a temporary
file in the same folder and a rename.

These helpers live apart from evaluation.store, which reads the truth through the answer key, so
that the benchmark can write its tables without importing the answer key's module (see the import
contract of the answer key's parser in pyproject.toml).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import uuid

import pyarrow as pa
import pyarrow.parquet as pq


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_bytes(path: Path, write: Callable[[Path], object]) -> None:
    """Write a file through a temporary file and a rename; write takes the temporary path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet(path: Path, table: pa.Table) -> None:
    """Write a Parquet table through a temporary file and a rename."""
    write_bytes(path, lambda temporary: pq.write_table(table, temporary))
