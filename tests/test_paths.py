"""Datasets live in data/<dataset id>/ and runs in results/<variant>/seed-<n>/."""

from __future__ import annotations

from pathlib import Path

from mule_pattern_learner import paths
from mule_pattern_learner.paths import DatasetPaths, RunPaths


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


def test_a_run_directory_names_every_file_of_the_run_directory_table(tmp_path: Path) -> None:
    assert paths.RESULTS_DIR == paths.REPOSITORY_ROOT / "results"
    assert RunPaths.of("baseline", 42).root == paths.RESULTS_DIR / "baseline" / "seed-42"
    run = RunPaths.of("no_attention", 43, tmp_path)
    assert run.root == tmp_path / "no_attention" / "seed-43"
    names = [
        run.config,
        run.model,
        run.resume,
        run.history,
        run.epochs,
        run.events,
        run.metrics,
        run.predictions("validation"),
        run.audit_report("test"),
        run.audit_scores("test"),
        run.audit_rejected("test"),
        run.scores("new_accounts", "2025-01-01"),
        run.scores_rejected("new_accounts", "2025-01-01"),
        run.plots,
        run.report,
    ]
    assert [path.relative_to(run.root).as_posix() for path in names] == [
        "config.json",
        "model.pt",
        "resume.pt",
        "history.csv",
        "epochs.csv",
        "events.jsonl",
        "metrics.json",
        "predictions/validation.parquet",
        "audit/test.json",
        "audit/test.parquet",
        "audit/test_rejected.txt",
        "scores/new_accounts_2025-01-01.parquet",
        "scores/new_accounts_2025-01-01_rejected.txt",
        "plots",
        "report.md",
    ]
    # Suites, diagnostics and the archive have their own directories under results/.
    assert paths.suite_dir("controls", tmp_path) == tmp_path / "experiments" / "controls"
    assert paths.diagnostics_dir("abc", tmp_path) == tmp_path / "diagnostics" / "abc"
    assert paths.archive_dir(tmp_path) == tmp_path / "archive"
    assert paths.suite_dir("controls").parent.parent == paths.RESULTS_DIR
