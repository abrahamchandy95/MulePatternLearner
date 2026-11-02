"""Every file location, each with one accessor, so every command behaves the same from any
working directory (docs/design.md, "Package layout" and "Cache").

Some locations hold labels or values computed from the truth: the evaluation store, the
loader's export folder (its manifest, its verification report and its Account shards) and
PhantomLedger's gate records. Only evaluation/ and testing/ call their accessors, as
tests/test_path_contract.py checks over every module (the Files that carry labels guard).

The two folders outside the repository come from the repository's .env, which environment
variables override: MULE_EXPORT_DIR, the loader's export root, and MULE_PL_RECORDS_DIR, where
the loader writes each world's identity and gate records from PhantomLedger. A relative
folder is taken from the repository root.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# The GSQL of the graph: its schema, and the queries the code installs.
GSQL_DIR = REPOSITORY_ROOT / "gsql"
# The settings outside the code: the TigerGraph connection and the two folders outside
# the repository.
ENV_FILE = REPOSITORY_ROOT / ".env"
# What the code writes, which git never holds: the cache, the training examples and the
# evaluation store; and the results of every run and of the benchmark.
DATA_DIR = REPOSITORY_ROOT / "data"
RESULTS_DIR = REPOSITORY_ROOT / "results"
# A world identity: 32 hex digits of a blake2b hash of read_world's fields.
IDENTITY = re.compile(r"[0-9a-f]{32}")
# The benchmark's folder under results/, which no run may take.
BENCH = "bench"
# A run's name: letters, digits, dots, dashes and underscores, as a folder name.
RUN = re.compile(rf"(?!{BENCH}$)[A-Za-z0-9][A-Za-z0-9._-]*")


class _Folders(BaseSettings):
    """The folders outside the repository, from .env or the environment."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    mule_export_dir: str = ""
    mule_pl_records_dir: str = ""


def _folder(name: str) -> Path:
    """The folder a setting of .env names, from the repository root when it is relative."""
    # pydantic-settings takes the file as _env_file, which its typed signature leaves out.
    folders = _Folders(_env_file=ENV_FILE)  # pyright: ignore[reportCallIssue]
    value: str = getattr(folders, name.lower())
    if not value:
        raise ValueError(f"{name} must be set in {ENV_FILE} or the environment")
    return REPOSITORY_ROOT / Path(value).expanduser()


def _name(what: str, value: object, pattern: re.Pattern[str]) -> str:
    """value when it is a string that pattern matches whole."""
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{what}, got {value!r}")
    return value


def _seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"A world's seed is a non-negative integer, got {seed!r}")
    return seed


def _identity(identity: object) -> str:
    return _name("A world identity is 32 hex digits", identity, IDENTITY)


def cache_dir(identity: str) -> Path:
    """The cache of a world: its folders per query versions and settings, then per scoring time."""
    return DATA_DIR / "cache" / _identity(identity)


def world_registry() -> Path:
    """The file that says, per configured world, which loaded world mule prepare last prepared
    under its name: its identity and its label identity, so that the commands that never query
    TigerGraph (train, evaluate) find the cache and the examples of a world by its name."""
    return DATA_DIR / "worlds.toml"


def examples_dir(identity: str) -> Path:
    """The training examples and weights of a world, in a folder per label identity."""
    return DATA_DIR / "examples" / _identity(identity)


def evaluation_store_dir(identity: str) -> Path:
    """The evaluation store of a world: the answer key's rows and mule audit's pass records.
    Only evaluation/ and testing/ call it."""
    return DATA_DIR / "evaluation" / _identity(identity)


def results_dir(run: str) -> Path:
    """The result tables, model files and manifest of a run."""
    return RESULTS_DIR / _name(f"A run's name is a folder name other than {BENCH!r}", run, RUN)


def bench_dir() -> Path:
    """The tables mule bench writes, one per world."""
    return RESULTS_DIR / BENCH


def parity_record() -> Path:
    """The parity record of the 5,000-person test, committed with the code."""
    return GSQL_DIR / "parity.toml"


def export_root() -> Path:
    """The loader's export root, MULE_EXPORT_DIR. Its exports hold label columns, so only
    evaluation/ and testing/ call it."""
    return _folder("MULE_EXPORT_DIR")


def pl_identity_files() -> list[Path]:
    """Every world's identity file the loader wrote into MULE_PL_RECORDS_DIR from
    PhantomLedger's world record, by name."""
    return sorted(_folder("MULE_PL_RECORDS_DIR").glob("world_*.identity.toml"))


def pl_gate_record(seed: int) -> Path:
    """The gate record of the world of that seed in MULE_PL_RECORDS_DIR. Its pass flags are
    computed from the truth, so only evaluation/ and testing/ call it."""
    return _folder("MULE_PL_RECORDS_DIR") / f"world_{_seed(seed)}.gates.toml"
