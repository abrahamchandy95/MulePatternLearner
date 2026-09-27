"""Dataset and run files: the one atomic write and the one file digest."""

from __future__ import annotations

from collections.abc import Generator
import contextlib
import hashlib
import os
from pathlib import Path


def file_digest(path: Path) -> str:
    """The sha256 of a file's bytes, in hex; manifests and saved models record it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pending_path(path: Path) -> Path:
    """Where ``atomic_write`` writes ``path`` before replacing it."""
    return path.with_name(path.name + ".pending")


@contextlib.contextmanager
def atomic_write(path: Path) -> Generator[Path]:
    """Yield a pending path beside ``path`` for the block to write; it then replaces ``path``.

    The pending file (``pending_path``) replaces ``path`` only when the block ends
    without an error, so a crash never leaves a truncated file. A block that writes
    nothing, or removes what it wrote, leaves ``path`` as it was. The pending file
    never outlives the block.
    """
    pending = pending_path(path)
    try:
        yield pending
        if pending.exists():
            os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
