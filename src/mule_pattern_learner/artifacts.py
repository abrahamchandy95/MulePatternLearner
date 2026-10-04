"""The files of datasets and runs: column schemas, reading and writing.

paths.DatasetPaths, paths.RunPaths, paths.SuitePaths and paths.DiagnosticsPaths say
where each file lives; this module says what the tables and JSON files of runs, suites
and diagnostic studies hold, and holds the one atomic write and the one file digest.
model.pt and resume.pt are torch payloads, which inference.saved_model.SavedModel and
training.checkpoint.ResumeState define.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from typing import Any

import pandas as pd

from .config import RunConfig

# history.csv: one row per training log interval. The loss, objective and timing are
# means over the interval's trained steps; the counters from database_calls on are
# totals of the run so far, over every segment of a resumed run.
HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",  # from 1
    "step",  # the steps of the epoch done when the interval ended
    "date",  # the train cutoff of the interval's last step
    "loss",  # the clamped nnPU loss that training minimised
    "objective",  # the unclamped nnPU risk
    "corrected_steps",  # steps whose non-negative correction fired
    "steps",  # the steps of the epoch's schedule
    "seconds_per_step",
    "batch_wait_seconds",
    "database_calls",
    "contexts_requested",
    "contexts_distinct",
    "memory_hits",  # contexts served from the source's in-memory cache
    "disk_hits",  # contexts read from the dataset's disk cache instead of requested
    "rejected_roots",  # training roots TigerGraph rejected
    "stub_children",
)
# epochs.csv: one row per epoch. validation_ap, validation_roc_auc and
# validation_pu_risk (the run's nnPU risk, lower being better) score the validation proxy
# on observed labels, with the weights named by weights ("averaged" or "raw"); they are
# the criteria of the selection rules (training.selection). selected marks the epoch
# whose weights model.pt holds, and stopped the epoch after which early stopping ended
# the run.
EPOCH_COLUMNS: tuple[str, ...] = (
    "epoch",
    "loss",
    "steps",
    "validation_ap",
    "validation_roc_auc",
    "validation_pu_risk",
    "weights",
    "selected",
    "stopped",
)
# The columns of an epochs.csv written before validation_pu_risk joined them, which
# read_epochs reads with the risk missing: the one earlier layout this code reads, so
# the runs of the first control experiments stay complete (docs/architecture.md).
EARLIER_EPOCH_COLUMNS = tuple(name for name in EPOCH_COLUMNS if name != "validation_pu_risk")
# predictions/<split>.parquet: the scored observed-label rows of a split (float64 scores).
PREDICTION_COLUMNS = ("account_id", "group_id", "date", "observed_label", "score")
# audit/<split>.parquet: the scored accounts of a split's audit sample, with their truth,
# the probability that the sample includes each one, whether the graph revealed the
# account's label before the split's cutoff, and its ring and label source (-1 and the
# label's record for accounts without a ring).
AUDIT_COLUMNS = (
    "account_id",
    "is_mule",
    "inclusion_probability",
    "score",
    "revealed",
    "ring_id",
    "label_source",
)


# A control-experiment suite's tables, which experiments.tables writes and reporting reads.
# summary.csv: one row per run, split and metric, with the run's status (complete,
# failed or stopped) and commit on each. Its metrics are those of the audit reports, the
# run's own values (best_epoch, parameter_count, training_hours: no split) and these:
SUMMARY_COLUMNS = ("variant", "seed", "split", "metric", "value", "status", "commit")
# the validation proxy AP of the epoch training selected;
PROXY_METRIC = "proxy_average_precision"
# a run's validation audit AP on the accounts every audit of the suite scored, and its
# difference from the baseline's of the same seed there.
PAIRED_METRIC = "paired_average_precision"
DELTA_METRIC = "average_precision_delta"


def file_digest(path: Path) -> str:
    """The sha256 of a file's bytes, in hex; manifests and saved models record it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# The file name ending of a file atomic_write has not yet put in place.
PENDING_SUFFIX = ".pending"


def pending_path(path: Path, *, unique: bool = False) -> Path:
    """Where ``atomic_write`` writes ``path`` before replacing it.

    A ``unique`` pending path holds a random part of its own, so writers of one path at
    the same time (the context cache's request workers and processes) never write into
    the same file: the path ends as one of their complete files.
    """
    own = f".{secrets.token_hex(8)}" if unique else ""
    return path.with_name(f"{path.name}{own}{PENDING_SUFFIX}")


@contextlib.contextmanager
def atomic_write(path: Path, *, unique: bool = False) -> Generator[Path]:
    """Yield a pending path beside ``path`` for the block to write; it then replaces ``path``.

    The pending file (``pending_path``, unique or not) replaces ``path`` only when the
    block ends without an error, so a crash never leaves a truncated file. A block that
    writes nothing, or removes what it wrote, leaves ``path`` as it was. The pending file
    never outlives the block, unless its thread dies at interpreter exit.
    """
    pending = pending_path(path, unique=unique)
    try:
        yield pending
        if pending.exists():
            os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    """Replace path with value as indented JSON; NaN and infinity are refused."""
    with atomic_write(path) as pending:
        pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_run_config(path: Path, config: RunConfig, provenance: Mapping[str, Any]) -> None:
    """config.json: the settings, their fingerprint and where and how the run ran."""
    record = {"config": config.to_dict(), "fingerprint": config.fingerprint()}
    write_json(path, {**record, "provenance": dict(provenance)})


def read_run_config(path: Path) -> RunConfig:
    """The settings a config.json records."""
    return RunConfig.from_dict(read_json(path)["config"])


def read_run_provenance(path: Path) -> dict[str, Any]:
    """Where and how the run a config.json records ran (write_run_config)."""
    return dict(read_json(path)["provenance"])


def event_line(record: Mapping[str, Any]) -> str:
    """One events.jsonl line: a JSON object; NaN and infinity are refused."""
    return json.dumps(record, allow_nan=False)


def append_event(path: Path, line: str) -> None:
    """Append an event_line to events.jsonl."""
    with path.open("a") as stream:
        stream.write(line + "\n")


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _checked(row: Mapping[str, Any], columns: tuple[str, ...], name: str) -> None:
    if set(row) != set(columns):
        raise ValueError(f"A {name} row has the columns {sorted(row)}, not {list(columns)}")


def append_history(path: Path, row: Mapping[str, Any]) -> None:
    """Append one log interval to history.csv, writing the header first if it is new."""
    _checked(row, HISTORY_COLUMNS, "history.csv")
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, HISTORY_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def keep_history(path: Path, epoch: int, step: int) -> None:
    """Drop the intervals history.csv logged after a resume position.

    ``epoch`` counts the finished epochs and ``step`` the steps done of the next one, as
    the resume state records them; a resumed run logs the later intervals again. A row
    cut short by a crash is dropped too. A history of other columns is refused rather than
    emptied, since none of its rows would have every column.
    """
    if not path.exists():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if reader.fieldnames is not None and tuple(reader.fieldnames) != HISTORY_COLUMNS:
        raise ValueError(
            f"{path} has the columns {list(reader.fieldnames)}, not {list(HISTORY_COLUMNS)}"
        )
    kept = [
        row
        for row in rows
        if None not in row
        and all(row.get(name) is not None for name in HISTORY_COLUMNS)
        and (int(row["epoch"]) - 1, int(row["step"])) <= (epoch, step)
    ]
    if len(kept) == len(rows):
        return
    with atomic_write(path) as pending, pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, HISTORY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)


def write_epochs(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Replace epochs.csv with one row per epoch so far."""
    rows = list(rows)
    for row in rows:
        _checked(row, EPOCH_COLUMNS, "epochs.csv")
    with atomic_write(path) as pending, pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, EPOCH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    # round_trip parses every float back to the value written.
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) != columns:
        raise ValueError(f"{path} has the columns {list(frame.columns)}, not {list(columns)}")
    return frame


def read_history(path: Path) -> pd.DataFrame:
    return _read_csv(path, HISTORY_COLUMNS)


def read_epochs(path: Path) -> pd.DataFrame:
    """epochs.csv; one written before validation_pu_risk (EARLIER_EPOCH_COLUMNS) has it NaN."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) == EARLIER_EPOCH_COLUMNS:
        frame.insert(EPOCH_COLUMNS.index("validation_pu_risk"), "validation_pu_risk", math.nan)
    if tuple(frame.columns) != EPOCH_COLUMNS:
        raise ValueError(f"{path} has the columns {list(frame.columns)}, not {list(EPOCH_COLUMNS)}")
    return frame


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    """Write a split's predictions (PREDICTION_COLUMNS, in that order)."""
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Predictions have the columns {list(frame.columns)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path) as pending:
        frame.to_parquet(pending, index=False)


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"{path} has the columns {list(frame.columns)}")
    return frame


def write_audit_scores(path: Path, frame: pd.DataFrame) -> None:
    """Write an audit sample's scored accounts (AUDIT_COLUMNS, in that order)."""
    if tuple(frame.columns) != AUDIT_COLUMNS:
        raise ValueError(f"Audit scores have the columns {list(frame.columns)}")
    with atomic_write(path) as pending:
        frame.to_parquet(pending, index=False)


def read_audit_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != AUDIT_COLUMNS:
        raise ValueError(f"{path} has the columns {list(frame.columns)}")
    return frame


def write_rejected(path: Path, ids: Iterable[str]) -> None:
    """The accounts TigerGraph rejected, one ID per line."""
    path.write_text("".join(value + "\n" for value in ids))


def write_table(path: Path, frame: pd.DataFrame) -> None:
    """Replace a CSV table (a suite's summary.csv or comparison.csv); missing values stay empty."""
    with atomic_write(path) as pending:
        frame.to_csv(pending, index=False)


# The text columns of a suite's tables; an empty cell reads as an empty string.
SUMMARY_TEXT = ("variant", "split", "metric", "status", "commit")
COMPARISON_TEXT = ("variant", "question", "changes", "seeds", "differs")


def _read_table(path: Path, text: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip", dtype=dict.fromkeys(text, str))
    for column in text:
        frame[column] = frame[column].fillna("")
    return frame


def read_summary(path: Path) -> pd.DataFrame:
    """A suite's summary.csv (SUMMARY_COLUMNS in order)."""
    frame = _read_table(path, SUMMARY_TEXT)
    if tuple(frame.columns) != SUMMARY_COLUMNS:
        raise ValueError(
            f"{path} has the columns {list(frame.columns)}, not {list(SUMMARY_COLUMNS)}"
        )
    return frame


def read_comparison(path: Path) -> pd.DataFrame:
    """A suite's comparison.csv: one row per variant, in the suite's order."""
    frame = _read_table(path, COMPARISON_TEXT)
    if frame.columns[0] != "variant":
        raise ValueError(
            f"{path} is not a comparison table: its first column is {frame.columns[0]}"
        )
    return frame


# A diagnostic study's tables (results/diagnostics/<dataset id>/<analysis>.csv), which the
# analyses of diagnostics write and reporting reads: long format, as summary.csv, one row
# per measurement. Each table has its analysis' key columns, then the metric and its
# value; the baselines give a metric's bootstrap interval as its low and high ends. A key
# that does not apply to a row is empty.
DIAGNOSTIC_TABLES: dict[str, tuple[str, ...]] = {
    "univariate": ("feature", "family", "split", "metric", "value"),
    "drift": ("feature", "family", "model", "setup", "split", "metric", "value"),
    "baselines": ("baseline", "features", "model", "split", "metric", "value", "low", "high"),
    "learning_curve": ("model", "labels", "mules", "repeat", "split", "metric", "value"),
    "subgroups": ("split", "subset", "rank", "metric", "value"),
    "proxy_validity": ("split", "subset", "metric", "value"),
    "reveal_spread": ("salt", "split", "metric", "value"),
    "nnpu_simulation": ("positive_weight", "seed", "metric", "value"),
}
DIAGNOSTIC_TEXT = (
    "feature",
    "family",
    "model",
    "setup",
    "split",
    "metric",
    "baseline",
    "features",
    "labels",
    "subset",
)
# features.parquet: the diagnostic feature table (diagnostics.feature_table), one row per
# sampled account with these columns first, then its features, each named
# <family>__<name>. weight is 1 / inclusion_probability, and rejected marks an account
# TigerGraph rejected, whose features are missing. The two contracts name the query texts
# the features were read with.
FEATURE_TABLE_COLUMNS = (
    "account_id",
    "split",
    "date",
    "is_mule",
    "revealed",
    "ring_id",
    "label_source",
    "inclusion_probability",
    "weight",
    "rejected",
    "context_contract",
    "analytics_contract",
)


def write_diagnostic_table(path: Path, analysis: str, frame: pd.DataFrame) -> None:
    """Replace an analysis' table (DIAGNOSTIC_TABLES[analysis], its columns in order)."""
    columns = DIAGNOSTIC_TABLES[analysis]
    if tuple(frame.columns) != columns:
        raise ValueError(f"The {analysis} table has the columns {list(frame.columns)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table(path, frame)


def read_diagnostic_table(path: Path, analysis: str) -> pd.DataFrame:
    """An analysis' table; its text keys read as strings, an empty one as an empty string."""
    columns = DIAGNOSTIC_TABLES[analysis]
    frame = _read_table(path, tuple(c for c in DIAGNOSTIC_TEXT if c in columns))
    if tuple(frame.columns) != columns:
        raise ValueError(f"{path} has the columns {list(frame.columns)}, not {list(columns)}")
    return frame


def write_feature_table(path: Path, frame: pd.DataFrame) -> None:
    """Replace features.parquet: FEATURE_TABLE_COLUMNS, then the feature columns."""
    if tuple(frame.columns[: len(FEATURE_TABLE_COLUMNS)]) != FEATURE_TABLE_COLUMNS:
        raise ValueError(f"A feature table starts with {list(FEATURE_TABLE_COLUMNS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path) as pending:
        frame.to_parquet(pending, index=False)


def read_feature_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if tuple(frame.columns[: len(FEATURE_TABLE_COLUMNS)]) != FEATURE_TABLE_COLUMNS:
        raise ValueError(f"{path} is not a feature table: it has the columns {frame.columns}")
    return frame
