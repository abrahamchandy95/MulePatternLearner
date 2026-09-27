"""Prepared datasets live in data/<dataset id>/, and DatasetPaths names their files."""

from __future__ import annotations

from pathlib import Path

from mule_pattern_learner import paths
from mule_pattern_learner.paths import DatasetPaths


def test_a_dataset_directory_is_named_by_its_dataset_id(tmp_path: Path) -> None:
    assert paths.DATA_DIR == paths.REPOSITORY_ROOT / "data"
    assert paths.GSQL_DIR == paths.REPOSITORY_ROOT / "gsql"
    assert DatasetPaths.of("abc").root == paths.DATA_DIR / "abc"
    dataset = DatasetPaths.of("abc", tmp_path)
    assert dataset.root == tmp_path / "abc"
    names = [dataset.manifest, dataset.accounts, dataset.observed_labels, dataset.hubs]
    assert [path.relative_to(dataset.root).as_posix() for path in names] == [
        "manifest.json",
        "accounts.parquet",
        "observed_labels.parquet",
        "hubs.parquet",
    ]


def test_datasets_are_the_directories_that_hold_a_manifest(tmp_path: Path) -> None:
    assert paths.datasets(tmp_path / "absent") == []
    for name in ("b", "a"):
        DatasetPaths.of(name, tmp_path).root.mkdir()
        DatasetPaths.of(name, tmp_path).manifest.write_text("{}")
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("")
    assert paths.datasets(tmp_path) == [
        DatasetPaths.of("a", tmp_path),
        DatasetPaths.of("b", tmp_path),
    ]
