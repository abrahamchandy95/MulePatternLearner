"""Repository paths, and where prepared datasets and the commands' outputs live.

Prepared datasets live in data/<dataset id>/ (DATA_DIR), and everything the commands
write lives under results/ (RESULTS_DIR): one training run in
results/<variant>/seed-<n>/, a control-experiment suite in results/experiments/<suite>/,
the diagnostics of a dataset in results/diagnostics/<dataset id>/, and runs moved aside
because their settings changed in results/archive/. DatasetPaths and RunPaths name each
file; nothing else joins a file name onto one of these directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# The GSQL files; the package names each one by its path relative to this folder.
GSQL_DIR = REPOSITORY_ROOT / "gsql"
# Prepared datasets, the inputs to training (gitignored).
DATA_DIR = REPOSITORY_ROOT / "data"
# Everything the commands write (gitignored).
RESULTS_DIR = REPOSITORY_ROOT / "results"
# The variant name of the built-in run, which `mule train` writes.
BASELINE_VARIANT = "baseline"


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


@dataclass(frozen=True)
class RunPaths:
    """The files of one training run, in its own directory."""

    root: Path

    @classmethod
    def of(cls, variant: str, seed: int, results: Path = RESULTS_DIR) -> RunPaths:
        """The directory of one run: <results>/<variant>/seed-<seed>/."""
        return cls(results / variant / f"seed-{seed}")

    @property
    def config(self) -> Path:
        """The configuration, its fingerprint and the run's provenance."""
        return self.root / "config.json"

    @property
    def model(self) -> Path:
        """The selected model (inference.saved_model.SavedModel)."""
        return self.root / "model.pt"

    @property
    def resume(self) -> Path:
        """What an interrupted run continues from (training.checkpoint.ResumeState)."""
        return self.root / "resume.pt"

    @property
    def history(self) -> Path:
        """One row per training log interval."""
        return self.root / "history.csv"

    @property
    def epochs(self) -> Path:
        """One row per epoch, with its validation."""
        return self.root / "epochs.csv"

    @property
    def events(self) -> Path:
        """The structured lines the commands printed for this run, one JSON object each."""
        return self.root / "events.jsonl"

    @property
    def metrics(self) -> Path:
        """The proxy metrics and totals of a complete run."""
        return self.root / "metrics.json"

    def predictions(self, split: str) -> Path:
        """The proxy scores of a split's observed-label rows."""
        return self.root / "predictions" / f"{split}.parquet"

    def audit_report(self, split: str) -> Path:
        """The ground-truth audit of a split: its metrics and constants."""
        return self.root / "audit" / f"{split}.json"

    def audit_scores(self, split: str) -> Path:
        """The scored accounts of a split's audit sample."""
        return self.root / "audit" / f"{split}.parquet"

    def audit_rejected(self, split: str) -> Path:
        """The accounts of a split's audit sample that TigerGraph rejected, one per line."""
        return self.root / "audit" / f"{split}_rejected.txt"

    def scores(self, accounts: str, date: str) -> Path:
        """Scores of the accounts listed in a file with this stem, at a date."""
        return self.root / "scores" / f"{accounts}_{date}.parquet"

    def scores_rejected(self, accounts: str, date: str) -> Path:
        """The accounts of those scores that TigerGraph rejected, one per line."""
        return self.root / "scores" / f"{accounts}_{date}_rejected.txt"

    @property
    def plots(self) -> Path:
        """The directory of the run's figures."""
        return self.root / "plots"

    def figure(self, name: str) -> Path:
        """One of the run's figures, a PNG named <topic>_<figure> (reporting.report)."""
        return self.plots / f"{name}.png"

    @property
    def report(self) -> Path:
        """The run's tables, with links to its figures."""
        return self.root / "report.md"


def suite_dir(suite: str, results: Path = RESULTS_DIR) -> Path:
    """The comparison tables and figures of a control-experiment suite."""
    return results / "experiments" / suite


def diagnostics_dir(dataset_id: str, results: Path = RESULTS_DIR) -> Path:
    """The diagnostic study of a prepared dataset."""
    return results / "diagnostics" / dataset_id


def archive_dir(results: Path = RESULTS_DIR) -> Path:
    """Where results whose settings changed are moved; nothing there is deleted."""
    return results / "archive"
