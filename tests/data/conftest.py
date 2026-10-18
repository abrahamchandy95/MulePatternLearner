"""What the data tests share: a folder of identity files and a data folder, each the test's
own."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mule_pattern_learner import paths


@pytest.fixture
def records(tmp_path: Path) -> Callable[..., Path]:
    """A writer of identity files into a folder of the test's own: records(name, text)."""
    folder = tmp_path / "records"
    folder.mkdir()

    def write(name: str, text: str) -> Path:
        path = folder / name
        path.write_text(text)
        return path

    return write


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The folder the cache and the examples are written under, the test's own."""
    folder = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_DIR", folder)
    return folder
