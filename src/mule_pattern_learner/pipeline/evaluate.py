"""Evaluation against oracle truth: saved predictions, and the final population audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evaluation.audit import audit_inputs, audit, evaluate_predictions
from ..evaluation.truth import ParquetTruth, TruthReader
from ..inference.saved_model import SavedModel
from ..paths import DATA_DIR, DatasetPaths, RunPaths
from ..tigergraph.context_query import TigerGraphContextFetcher
from ..tigergraph.oracle import TigerGraphTruth
from ..tigergraph.provenance import verify_frozen_source
from ..tigergraph.scope import TigerGraphScope
from .connect import connect


def evaluate(predictions: Path, checkpoint: Path, truth: Path | None) -> dict[str, Any]:
    """Saved predictions against a supplied truth parquet, else the graph's oracle truth.

    The graph is read on a connection with the checkpoint's retry budgets.
    """
    saved = SavedModel.of(checkpoint)
    reader: TruthReader
    if truth is not None:
        reader = ParquetTruth(truth)
    else:
        reader = TigerGraphTruth(connect(saved.config.transport))
    return evaluate_predictions(predictions, saved, reader)


def evaluate_run(
    run: RunPaths,
    truth: Path | None,
    *,
    dataset: DatasetPaths | None = None,
    data: Path = DATA_DIR,
) -> dict[str, Any]:
    """The audit of a run's frozen model, which connects once its inputs passed their checks.

    The dataset is the model's own in data unless ``dataset`` names another. The
    connection has the model's retry budgets, and its source must still be the frozen
    one the dataset was prepared from. Truth is a supplied parquet, else the graph's
    oracle truth. The audit goes into the run's audit/ files.
    """
    saved, dataset, manifest = audit_inputs(run, dataset, data)
    executor = connect(saved.config.transport)
    verify_frozen_source(executor, manifest)
    reader: TruthReader
    if truth is not None:
        reader = ParquetTruth(truth)
    else:
        reader = TigerGraphTruth(executor)
    return audit(
        run,
        reader,
        scope=TigerGraphScope(executor),
        fetcher=TigerGraphContextFetcher(executor),
        dataset=dataset,
    )
