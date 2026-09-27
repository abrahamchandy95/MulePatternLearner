"""The prepared dataset lives inside the run."""

from __future__ import annotations

from pathlib import Path

from mule_pattern_learner import paths


def test_the_prepared_dataset_is_inside_the_run(tmp_path: Path) -> None:
    assert paths.dataset_path(tmp_path / "m.pt") == tmp_path / "m_run" / "prepared"
    assert paths.dataset_path(tmp_path / "run") == tmp_path / "run" / "prepared"
