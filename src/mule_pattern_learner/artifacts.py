"""The files of datasets and runs: column schemas, reading and writing.

paths.DatasetPaths and paths.RunPaths say where each file lives; this module says what
the tables and JSON files of a run hold, and holds the one atomic write and the one
file digest. model.pt and resume.pt are torch payloads, which
inference.saved_model.SavedModel and training.checkpoint.ResumeState define.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
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
    "cache_hits",
    "rejected_roots",  # training roots TigerGraph rejected
    "stub_children",
)
# epochs.csv: one row per epoch. validation_ap and validation_roc_auc are proxy metrics
# on observed labels, of the weights named by weights ("averaged" or "raw"); selected
# marks the epoch whose weights model.pt holds, and stopped the epoch after which early
# stopping ended the run.
EPOCH_COLUMNS: tuple[str, ...] = (
    "epoch",
    "loss",
    "steps",
    "validation_ap",
    "validation_roc_auc",
    "weights",
    "selected",
    "stopped",
)
# predictions/<split>.parquet: the scored observed-label rows of a split (float64 scores).
PREDICTION_COLUMNS = ("account_id", "group_id", "date", "observed_label", "score")
# audit/<split>.parquet: the scored accounts of a split's audit sample, with their truth
# and the probability that the sample includes each one.
AUDIT_COLUMNS = ("account_id", "split", "is_mule", "inclusion_probability", "score")


def file_digest(path: Path) -> str:
    """The sha256 of a file's bytes, in hex; manifests and saved models record it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pending_path(path: Path) -> Path:
    """Where ``atomic_write`` writes ``path`` before replacing it."""
    return path.with_name(path.name + ".pending")


@contextlib.contextmanager
def atomic_write(path: Path) -> Generator[Path]:
    """Yield a pending path beside ``path`` for the block to write; it then replaces ``path``.

    The pending file (``pending_path``) replaces ``path`` only when the block ends
    without an error, so a crash never leaves a truncated file. A block that writes
    nothing, or removes what it wrote, leaves ``path`` as it was. The pending file
    never outlives the block.
    """
    pending = pending_path(path)
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
    cut short by a crash is dropped too.
    """
    if not path.exists():
        return
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
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
    return _read_csv(path, EPOCH_COLUMNS)


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
