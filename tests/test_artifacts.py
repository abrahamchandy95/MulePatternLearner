"""The one atomic write and the one file digest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mule_pattern_learner.artifacts import atomic_write, file_digest, pending_path


def test_atomic_writes_replace_the_file_only_after_the_block_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("old")
    with pytest.raises(RuntimeError, match="crash"):
        with atomic_write(path) as pending:
            assert pending == pending_path(path) == tmp_path / "manifest.json.pending"
            pending.write_text("half")
            raise RuntimeError("crash")
    assert path.read_text() == "old" and not pending_path(path).exists()
    with atomic_write(path) as pending:
        pending.write_text("new")
        assert path.read_text() == "old"
    assert path.read_text() == "new" and not pending_path(path).exists()
    # A block that writes nothing, or removes what it wrote, leaves the file alone.
    with atomic_write(path) as pending:
        pending.write_text("discarded")
        pending.unlink()
    with atomic_write(tmp_path / "absent.txt"):
        pass
    assert path.read_text() == "new" and not (tmp_path / "absent.txt").exists()


def test_file_digests_are_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "accounts.parquet"
    path.write_bytes(b"\x00parquet\xff" * 1000)
    assert file_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
