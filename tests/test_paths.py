"""The prepared cache lives inside the run unless a prepared id names it."""

from __future__ import annotations

from pathlib import Path

from mule_pattern_learner import paths


def test_prepared_directory_is_inside_the_run_unless_prepared_id_is_set(tmp_path: Path) -> None:
    assert paths.dataset_path({}, tmp_path / "m.pt") == tmp_path / "m_run" / "prepared"
    shared = paths.dataset_path({"prepared_id": "p2"}, tmp_path / "m.pt")
    assert shared.name == "p2" and shared.parent.name == "temporal"
