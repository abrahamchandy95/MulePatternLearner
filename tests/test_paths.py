"""Every file location has one accessor, found from the package rather than the working
directory; the two folders outside the repository come from .env or the environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner import paths
from mule_pattern_learner.paths import (
    DATA_DIR,
    ENV_FILE,
    GSQL_DIR,
    REPOSITORY_ROOT,
    RESULTS_DIR,
    bench_dir,
    cache_dir,
    evaluation_store_dir,
    examples_dir,
    export_root,
    parity_record,
    pl_gate_record,
    pl_identity_files,
    results_dir,
)
from mule_pattern_learner.tigergraph.connection import ConnectionSettings

IDENTITY = "0123456789abcdef0123456789abcdef"
FOLDERS = ("MULE_EXPORT_DIR", "MULE_PL_RECORDS_DIR")


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty .env of the test's own, with neither folder in the environment."""
    env = tmp_path / ".env"
    env.write_text("")
    monkeypatch.setattr(paths, "ENV_FILE", env)
    for name in FOLDERS:
        monkeypatch.delenv(name, raising=False)
    return env


def test_the_repository_root_holds_the_project_and_its_design() -> None:
    assert REPOSITORY_ROOT.is_absolute()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert (REPOSITORY_ROOT / "docs" / "design.md").is_file()
    assert GSQL_DIR == REPOSITORY_ROOT / "gsql"


def test_the_connection_and_the_folders_read_one_env_file() -> None:
    assert ENV_FILE == REPOSITORY_ROOT / ".env"
    assert ConnectionSettings.model_config.get("env_file") is ENV_FILE


def test_what_the_code_writes_lives_under_the_git_ignored_data_and_results() -> None:
    assert (DATA_DIR, RESULTS_DIR) == (REPOSITORY_ROOT / "data", REPOSITORY_ROOT / "results")
    assert cache_dir(IDENTITY) == DATA_DIR / "cache" / IDENTITY
    assert examples_dir(IDENTITY) == DATA_DIR / "examples" / IDENTITY
    assert evaluation_store_dir(IDENTITY) == DATA_DIR / "evaluation" / IDENTITY
    assert results_dir("dev-2026-10-17") == RESULTS_DIR / "dev-2026-10-17"
    assert bench_dir() == RESULTS_DIR / "bench"
    # The parity record is committed with the code, in TOML since git ignores JSON.
    assert parity_record() == GSQL_DIR / "parity.toml"
    ignored = (REPOSITORY_ROOT / ".gitignore").read_text().splitlines()
    assert {"/data/", "/results/", "*.json"} <= set(ignored)


@pytest.mark.parametrize(
    "identity", ["", "0123", IDENTITY.upper(), f"{IDENTITY}0", "../" + IDENTITY[3:], 7]
)
def test_a_folder_of_a_world_needs_its_identity(identity: object) -> None:
    for accessor in (cache_dir, examples_dir, evaluation_store_dir):
        with pytest.raises(ValueError, match="32 hex digits"):
            accessor(identity)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("run", ["", "bench", "../x", "a/b", ".hidden"])
def test_a_run_is_named_by_a_folder_name_other_than_the_benchmarks(run: str) -> None:
    with pytest.raises(ValueError, match="run's name"):
        results_dir(run)


def test_the_folders_outside_the_repository_come_from_env(
    env_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = tmp_path / "records"
    records.mkdir()
    for name in ("world_2002.identity.toml", "world_2001.identity.toml", "world_2001.gates.toml"):
        (records / name).write_text("")
    env_file.write_text(f"MULE_EXPORT_DIR=../loader/exports\nMULE_PL_RECORDS_DIR={records}\n")
    # A relative folder is taken from the repository root.
    assert export_root() == REPOSITORY_ROOT / "../loader/exports"
    assert pl_identity_files() == [
        records / "world_2001.identity.toml",
        records / "world_2002.identity.toml",
    ]
    assert pl_gate_record(2001) == records / "world_2001.gates.toml"
    # The environment overrides .env.
    monkeypatch.setenv("MULE_EXPORT_DIR", str(tmp_path / "exports"))
    assert export_root() == tmp_path / "exports"


def test_a_folder_that_is_not_set_is_named(env_file: Path) -> None:
    with pytest.raises(ValueError, match=f"MULE_EXPORT_DIR must be set in {env_file}"):
        export_root()
    with pytest.raises(ValueError, match="MULE_PL_RECORDS_DIR must be set"):
        pl_identity_files()
    with pytest.raises(ValueError, match="MULE_PL_RECORDS_DIR must be set"):
        pl_gate_record(2001)


@pytest.mark.parametrize("seed", [-1, True, "2001"])
def test_a_gate_record_needs_a_seed(env_file: Path, seed: object) -> None:
    env_file.write_text("MULE_PL_RECORDS_DIR=/records\n")
    with pytest.raises(ValueError, match="seed"):
        pl_gate_record(seed)  # pyright: ignore[reportArgumentType]
