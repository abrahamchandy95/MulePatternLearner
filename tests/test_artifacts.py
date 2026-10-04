"""The run files' schemas, the one atomic write and the one file digest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.artifacts import (
    AUDIT_COLUMNS,
    DIAGNOSTIC_TABLES,
    EARLIER_EPOCH_COLUMNS,
    EPOCH_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    HISTORY_COLUMNS,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    append_history,
    atomic_write,
    file_digest,
    hidden_rows,
    keep_history,
    pending_path,
    read_audit_report,
    read_comparison,
    read_epochs,
    read_history,
    read_json,
    read_predictions,
    read_run_config,
    write_epochs,
    write_json,
    write_predictions,
    write_run_config,
)
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import REPOSITORY_ROOT, DiagnosticsPaths, RunPaths
from mule_pattern_learner.reporting.run_report import AUDIT_FIGURES, TRAINING_FIGURES
from mule_pattern_learner.reporting.study_report import DIAGNOSTICS_FIGURES
from mule_pattern_learner.reporting.suite_report import SUITE_FIGURES


def test_atomic_writes_replace_the_file_only_after_the_block_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("old")
    with pytest.raises(RuntimeError, match="crash"):
        with atomic_write(path) as pending:
            assert pending == pending_path(path) == tmp_path / "manifest.json.pending"
            pending.write_text("half")
            raise RuntimeError("crash")
    assert path.read_text() == "old" and not pending_path(path).exists()
    with atomic_write(path) as pending:
        pending.write_text("new")
        assert path.read_text() == "old"
    assert path.read_text() == "new" and not pending_path(path).exists()
    # A block that writes nothing, or removes what it wrote, leaves the file alone.
    with atomic_write(path) as pending:
        pending.write_text("discarded")
        pending.unlink()
    with atomic_write(tmp_path / "absent.txt"):
        pass
    assert path.read_text() == "new" and not (tmp_path / "absent.txt").exists()
    # Writers of one path at once each write a unique pending file of their own.
    with atomic_write(path, unique=True) as first, atomic_write(path, unique=True) as second:
        assert first != second and first.parent == tmp_path
        assert first.name.startswith("manifest.json.") and first.suffix == ".pending"
        first.write_text("first")
        second.write_text("second")
    assert path.read_text() == "first" and list(tmp_path.glob("*.pending")) == []


def test_file_digests_are_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "accounts.parquet"
    path.write_bytes(b"\x00parquet\xff" * 1000)
    assert file_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def history_row(epoch: int, step: int, **changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(HISTORY_COLUMNS, 0)
    return {**row, "epoch": epoch, "step": step, "date": "2024-07-01", "loss": 0.1, **changes}


def test_history_rows_append_and_a_resume_drops_the_later_ones(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    keep_history(path, 0, 0)
    assert not path.exists()
    positions = [(1, 2), (1, 4), (2, 2), (2, 4), (3, 2)]
    for epoch, step in positions:
        append_history(path, history_row(epoch, step, loss=1 / 3 + epoch))
    frame = read_history(path)
    assert tuple(frame.columns) == HISTORY_COLUMNS and len(frame) == 5
    # Floats come back exactly as written.
    assert frame.loss.tolist() == [1 / 3 + epoch for epoch, _ in positions]
    with pytest.raises(ValueError, match="columns"):
        append_history(path, {"epoch": 1})
    # A crash cut the last row short.
    path.write_text(path.read_text() + "3,4,2024-07")
    # Resumed after epoch 2 and 2 steps of epoch 3, as resume.pt records it (2, 2).
    keep_history(path, 2, 2)
    assert read_history(path)[["epoch", "step"]].to_numpy().tolist() == [list(p) for p in positions]
    keep_history(path, 1, 2)
    assert read_history(path)[["epoch", "step"]].to_numpy().tolist() == [[1, 2], [1, 4], [2, 2]]
    keep_history(path, 0, 0)
    assert read_history(path).empty
    # A history of other columns is refused, not emptied.
    other = tmp_path / "other.csv"
    other.write_text("epoch,step,loss\n1,2,0.5\n")
    with pytest.raises(ValueError, match="columns"):
        keep_history(other, 0, 0)
    assert other.read_text() == "epoch,step,loss\n1,2,0.5\n"


def test_epochs_are_rewritten_whole_and_read_with_their_types(tmp_path: Path) -> None:
    path = tmp_path / "epochs.csv"
    row = {
        "epoch": 1,
        "loss": 0.5,
        "steps": 4,
        "validation_ap": 0.25,
        "validation_roc_auc": None,
        "validation_pu_risk": 0.75,
        "weights": "averaged",
        "selected": True,
        "stopped": False,
    }
    write_epochs(path, [row, {**row, "epoch": 2, "selected": False, "stopped": True}])
    frame = read_epochs(path)
    assert tuple(frame.columns) == EPOCH_COLUMNS
    assert frame.selected.tolist() == [True, False] and frame.stopped.tolist() == [False, True]
    assert frame.validation_roc_auc.isna().all() and frame.weights.tolist() == ["averaged"] * 2
    assert frame.validation_pu_risk.tolist() == [0.75, 0.75]
    write_epochs(path, [row])
    assert len(read_epochs(path)) == 1
    with pytest.raises(ValueError, match="columns"):
        write_epochs(path, [{"epoch": 1}])


def test_an_epochs_csv_written_before_the_risk_reads_without_it(tmp_path: Path) -> None:
    # The layout of the runs of the first control experiments: every column but the risk.
    path = tmp_path / "epochs.csv"
    assert EARLIER_EPOCH_COLUMNS == tuple(c for c in EPOCH_COLUMNS if c != "validation_pu_risk")
    path.write_text(",".join(EARLIER_EPOCH_COLUMNS) + "\n1,0.5,4,0.25,0.9,averaged,True,False\n")
    frame = read_epochs(path)
    assert tuple(frame.columns) == EPOCH_COLUMNS
    assert frame.validation_pu_risk.isna().all() and frame.validation_ap.tolist() == [0.25]
    # No other layout is read: the earlier columns in another order, or one column fewer.
    for columns in (
        ["loss", "epoch", *EARLIER_EPOCH_COLUMNS[2:]],
        list(EARLIER_EPOCH_COLUMNS[:-1]),
    ):
        path.write_text(",".join(columns) + "\n")
        with pytest.raises(ValueError, match="columns"):
            read_epochs(path)


def test_predictions_keep_their_columns_and_json_refuses_nan(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "account_id": ["a"],
            "group_id": [0],
            "date": ["2024-10-01"],
            "observed_label": [1],
            "score": [0.5],
        }
    )
    path = tmp_path / "predictions" / "validation.parquet"
    write_predictions(path, frame)
    pd.testing.assert_frame_equal(read_predictions(path), frame)
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    with pytest.raises(ValueError, match="columns"):
        write_predictions(path, frame[["score", "account_id"]])
    with pytest.raises(ValueError):
        write_json(tmp_path / "metrics.json", {"ap": float("nan")})
    assert not (tmp_path / "metrics.json").exists()
    write_json(tmp_path / "metrics.json", {"ap": 0.5})
    assert read_json(tmp_path / "metrics.json") == {"ap": 0.5}


def test_an_audit_of_earlier_code_is_refused_and_the_hidden_rows_leave_the_revealed_out(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit" / "validation.json"
    path.parent.mkdir()
    write_json(path, {"metrics": {"average_precision": 0.3}, "intervals": {}})
    with pytest.raises(ValueError, match=r"audit of earlier code.*mule evaluate"):
        read_audit_report(path)
    report = {"hidden_metrics": {}, "hidden_intervals": {}, "metrics": {}, "intervals": {}}
    write_json(path, report)
    assert read_audit_report(path) == report
    frame = pd.DataFrame(
        {"account_id": ["a", "b", "c"], "is_mule": [1, 1, 0], "revealed": [True, False, False]}
    )
    assert hidden_rows(frame).account_id.tolist() == ["b", "c"]


def test_run_configurations_round_trip_with_their_fingerprint(tmp_path: Path) -> None:
    config = DEFAULT_CONFIG.with_changes({"training": {"epochs": 3}})
    path = tmp_path / "config.json"
    write_run_config(path, config, {"device": "cpu"})
    assert read_run_config(path) == config
    assert read_json(path)["fingerprint"] == config.fingerprint()
    assert read_json(path)["provenance"] == {"device": "cpu"}


def test_the_outputs_reference_names_every_file_column_and_figure() -> None:
    page = (REPOSITORY_ROOT / "docs/reference/outputs.md").read_text()
    # The code spans of the page, outside its fenced block.
    named = set(re.findall(r"`([^`\n]+)`", re.sub(r"^```.*?^```", "", page, flags=re.S | re.M)))
    run = RunPaths(REPOSITORY_ROOT)
    files = [
        path.relative_to(run.root).as_posix()
        for path in (run.config, run.model, run.resume, run.history, run.epochs, run.events)
    ]
    files += [run.metrics.name, run.report.name, DiagnosticsPaths(REPOSITORY_ROOT).study.name]
    columns = [
        *HISTORY_COLUMNS,
        *EPOCH_COLUMNS,
        *PREDICTION_COLUMNS,
        *AUDIT_COLUMNS,
        *SUMMARY_COLUMNS,
        *FEATURE_TABLE_COLUMNS,
        *(column for table in DIAGNOSTIC_TABLES.values() for column in table),
    ]
    figures = [
        f"{name}.png"
        for kind in (TRAINING_FIGURES, AUDIT_FIGURES, SUITE_FIGURES, DIAGNOSTICS_FIGURES)
        for name in kind
    ]
    tables = [f"{name}.csv" for name in DIAGNOSTIC_TABLES]
    assert [name for name in (*files, *columns, *figures, *tables) if name not in named] == []


def test_a_comparison_table_of_earlier_code_is_refused_with_what_to_do(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    # A comparison.csv written before the seed ensembles, without the estimate column.
    path.write_text("variant,question,changes,seeds,validation_ap\nbaseline,q,,42,0.1\n")
    with pytest.raises(ValueError, match="run scripts/run_experiments.py again"):
        read_comparison(path)
    # One written before the audits led with the hidden mules, without their AP.
    path.write_text("variant,estimate,question,changes,seeds,differs\nbaseline,seed_mean,q,,42,\n")
    with pytest.raises(ValueError, match=r"no validation_hidden_ap.*run scripts/run_experiments"):
        read_comparison(path)
    header = "variant,estimate,question,changes,seeds,differs,validation_hidden_ap"
    path.write_text(f"{header}\nbaseline,seed_mean,q,,42,,0.25\n")
    frame = read_comparison(path)
    assert frame.estimate.tolist() == ["seed_mean"] and frame.changes.tolist() == [""]
    assert frame.validation_hidden_ap.tolist() == [0.25]
