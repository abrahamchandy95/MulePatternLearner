"""Repository paths, and the run and dataset path policy.

Prepared datasets live in data/<dataset id>/ (DATA_DIR). DatasetPaths names each file
of one; nothing else joins a dataset file name onto a directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# The GSQL files; the package names each one by its path relative to this folder.
GSQL_DIR = REPOSITORY_ROOT / "gsql"
# Prepared datasets, the inputs to training (gitignored).
DATA_DIR = REPOSITORY_ROOT / "data"
DEFAULT_MODEL = REPOSITORY_ROOT / "models/temporal/model.pt"


@dataclass(frozen=True)
class DatasetPaths:
    """The files of one prepared dataset, in its own directory."""

    root: Path

    @classmethod
    def of(cls, dataset_id: str, data: Path = DATA_DIR) -> DatasetPaths:
        """The directory of a dataset id: <data>/<dataset id>/."""
        return cls(data / dataset_id)

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def accounts(self) -> Path:
        return self.root / "accounts.parquet"

    @property
    def observed_labels(self) -> Path:
        return self.root / "observed_labels.parquet"

    @property
    def hubs(self) -> Path:
        return self.root / "hubs.parquet"


def datasets(data: Path = DATA_DIR) -> list[DatasetPaths]:
    """Every dataset directory under data that holds a manifest, by name."""
    if not data.is_dir():
        return []
    found = (DatasetPaths(path) for path in sorted(data.iterdir()) if path.is_dir())
    return [dataset for dataset in found if dataset.manifest.exists()]


def output_paths(output: Path) -> tuple[Path, Path]:
    """A .pt output names the model; directory outputs retain the experiment API."""
    if output.suffix == ".pt":
        return output, output.with_name(output.stem + "_run")
    return output / "model.pt", output
