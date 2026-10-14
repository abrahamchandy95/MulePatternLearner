"""The repository root is found from the package, not from the working directory."""

from __future__ import annotations

from mule_pattern_learner.paths import REPOSITORY_ROOT


def test_the_repository_root_holds_the_project_and_its_design() -> None:
    assert REPOSITORY_ROOT.is_absolute()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert (REPOSITORY_ROOT / "docs" / "design.md").is_file()
