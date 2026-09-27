"""The built-in run needs no flag, file or identifier, and writes the files of its tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from mule_pattern_learner.artifacts import (
    read_epochs,
    read_events,
    read_history,
    read_json,
    write_run_config,
)
from mule_pattern_learner.cli import build_parser
from mule_pattern_learner.config import DEFAULT_CONFIG, TransportConfig
from mule_pattern_learner.data.manifest import dataset_id
from mule_pattern_learner.evaluation.truth import ParquetTruth
from mule_pattern_learner.inference.saved_model import SavedModel
from mule_pattern_learner.paths import BASELINE_VARIANT, RESULTS_DIR, DatasetPaths, RunPaths
from mule_pattern_learner.pipeline import connect as pipeline_connect
from mule_pattern_learner.pipeline import evaluate as pipeline_evaluate
from mule_pattern_learner.pipeline import prepare as pipeline_prepare
from mule_pattern_learner.pipeline.connect import open_context_source
from mule_pattern_learner.pipeline.evaluate import evaluate_run
from mule_pattern_learner.pipeline.train import BASELINE_RUN, train_run
from mule_pattern_learner.testing.builders import neighbourhood, scope_population
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph


def test_minimal_command_and_run_defaults(tmp_path: Path) -> None:
    # `mule train` needs no flag, file or identifier.
    args = build_parser().parse_args(["train"])
    assert vars(args) == {"command": "train"}
    # The built-in run goes to results/baseline/seed-42/.
    assert BASELINE_RUN == RunPaths(RESULTS_DIR / "baseline" / "seed-42")
    dataset = DatasetPaths.of("id", tmp_path / "data")
    output = RunPaths(tmp_path / "run")
    with (
        patch("mule_pattern_learner.pipeline.train.prepare_dataset", return_value=dataset) as prep,
        patch(
            "mule_pattern_learner.pipeline.train.train", return_value={"status": "complete"}
        ) as fit,
    ):
        assert train_run(output, data=tmp_path / "data")["status"] == "complete"
        prep.assert_called_once()
        # The built-in run's dataset, in the data directory.
        assert prep.call_args.args == (DEFAULT_CONFIG, tmp_path / "data")
        fit.assert_called_once()
        config, trained_on, written = fit.call_args.args
        assert written == output and trained_on == dataset
        assert fit.call_args.kwargs["open_contexts"] is open_context_source
        assert config is DEFAULT_CONFIG
        # A started run is refused before anything is prepared, unless it is resumed.
        output.root.mkdir()
        write_run_config(output.config, DEFAULT_CONFIG, {})
        with pytest.raises(FileExistsError, match="Run already exists"):
            train_run(output, data=tmp_path / "data")
        assert prep.call_count == 1
        # So is resuming it with other settings, and the error names them.
        changed = DEFAULT_CONFIG.with_changes({"training": {"epochs": 3}})
        with pytest.raises(ValueError, match=r"differs from the run: \['training.epochs'\]"):
            train_run(output, config=changed, data=tmp_path / "data", resume=True)
        assert prep.call_count == 1
        train_run(output, data=tmp_path / "data", resume=True)
        assert fit.call_args.kwargs["resume"] is True
        # Only the built-in run's settings may default to its directory.
        with pytest.raises(ValueError, match="Only the built-in run trains into"):
            train_run(config=changed, data=tmp_path / "data")
        assert prep.call_count == 2


# The source id of the fake graph's data, and its accounts.
FAKE_SOURCE = "end_to_end_fixture"
POPULATION = 200


def fake_graph(monkeypatch: pytest.MonkeyPatch) -> FakeTigerGraph:
    """The fake graph behind every connection of the pipeline.

    Its queries are installed and it holds the built-in run's frozen scope of FAKE_SOURCE,
    whose known mules are revealed (the reveal is a no-op), so preparation only reads.
    """
    scope = DEFAULT_CONFIG.scope.id
    header = {"ready": True, "source_id": FAKE_SOURCE, "split_seed": 42}
    executor = FakeTigerGraph(
        factory=neighbourhood,
        hubs=[("N3", cutoff) for cutoff in (101, 102, 103)],
        statuses={"N5": "history_capacity_exceeded"},
        population=scope_population(POPULATION),
        scopes={scope: header},
    )

    def connect(transport: TransportConfig) -> FakeTigerGraph:
        return executor

    def nothing(*args: Any, **kwargs: Any) -> None:
        return None

    for module in (pipeline_prepare, pipeline_connect, pipeline_evaluate):
        monkeypatch.setattr(module, "connect", connect)
    monkeypatch.setattr(pipeline_prepare, "ensure_revealed_labels", nothing)
    return executor


def files(directory: Path) -> set[str]:
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


def test_train_then_audit_write_exactly_the_files_of_the_run_and_dataset_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_graph(monkeypatch)
    config = DEFAULT_CONFIG.with_changes(
        {
            "dataset": {"seed_limits": {"train": 64, "validation": 24, "test": 24}},
            "training": {"epochs": 2, "steps_per_epoch": 3, "batch_size": 16},
            "runtime": {"device": "cpu", "threads": 1},
        }
    )
    data, results = tmp_path / "data", tmp_path / "results"
    output = RunPaths.of(BASELINE_VARIANT, config.training.seed, results)
    result = train_run(output, config=config, data=data)
    assert result["status"] == "complete"
    # The dataset is data/<dataset id>/, and nothing else is written there.
    identity = dataset_id(FAKE_SOURCE, config)
    prepared = {
        f"{identity}/manifest.json",
        f"{identity}/accounts.parquet",
        f"{identity}/observed_labels.parquet",
        f"{identity}/hubs.parquet",
    }
    assert result["dataset_id"] == identity and files(data) == prepared
    trained = {
        "config.json",
        "model.pt",
        "resume.pt",
        "history.csv",
        "epochs.csv",
        "events.jsonl",
        "metrics.json",
        "predictions/validation.parquet",
        "predictions/test.parquet",
    }
    assert output.root == results / "baseline" / "seed-42"
    assert files(results) == {f"baseline/seed-42/{name}" for name in trained}
    assert SavedModel.load(output.model).dataset(data) == DatasetPaths.of(identity, data)
    events = [event["event"] for event in read_events(output.events)]
    assert events[0] == "start" and events[-1] == "complete"
    # log_every_steps is above the 3 steps of an epoch: one interval per epoch.
    assert read_history(output.history)[["epoch", "step"]].to_numpy().tolist() == [[1, 3], [2, 3]]
    assert read_epochs(output.epochs).epoch.tolist() == [1, 2]
    # `mule train` reports a complete run with the result it printed, before anything
    # is prepared or connected, and leaves the run as it was. Changed settings are named.
    written = {name: (output.root / name).stat().st_mtime_ns for name in trained}
    with patch(
        "mule_pattern_learner.pipeline.train.prepare_dataset", side_effect=AssertionError
    ) as prep:
        report = train_run(output, config=config, data=data, resume=True)
        assert report == result == read_json(output.metrics)
        changed = config.with_changes({"training": {"patience": 1}})
        with pytest.raises(ValueError, match=r"other settings: \['training.patience'\]"):
            train_run(output, config=changed, data=data, resume=True)
        prep.assert_not_called()
    assert {name: (output.root / name).stat().st_mtime_ns for name in trained} == written
    # The audit adds its files to the run's audit/ and reads the model's own dataset.
    truth = pd.DataFrame(scope_population(POPULATION))[["account_id"]]
    truth["is_mule"] = (truth.index % 3 == 0).astype(int)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    audit = evaluate_run(output, truth=ParquetTruth(truth_path), data=data)
    assert audit["rejected_accounts"] == 0
    assert files(output.root) == trained | {"audit/test.json", "audit/test.parquet"}
    assert files(data) == prepared
    # The audit appends its line to the run's events.jsonl, after training's.
    recorded = read_events(output.events)
    assert [event["event"] for event in recorded] == [*events, "audit"]
    assert recorded[-1]["split"] == "test" and recorded[-1]["rejected_accounts"] == 0
