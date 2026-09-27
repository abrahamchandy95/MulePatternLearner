"""Evaluation against oracle truth: saved predictions, and the final population audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evaluation.audit import audit_inputs, evaluate_final_population, evaluate_predictions
from ..evaluation.truth import EvaluationTruthSource, ParquetEvaluationTruth
from ..inference.saved_model import ModelCheckpoint
from ..tigergraph.oracle import GraphEvaluationTruth
from ..tigergraph.provenance import verify_frozen_source
from .connect import connect


def evaluate(predictions: Path, checkpoint: Path, truth: Path | None) -> dict[str, Any]:
    """Saved predictions against a supplied truth parquet, else the graph's oracle truth.

    The graph is read on a connection with the checkpoint's retry budgets.
    """
    saved = ModelCheckpoint.of(checkpoint)
    reader: EvaluationTruthSource
    if truth is not None:
        reader = ParquetEvaluationTruth(truth)
    else:
        reader = GraphEvaluationTruth(connect(saved.validated_config()))
    return evaluate_predictions(predictions, saved, reader)


def final_audit(
    checkpoint: Path, truth: Path | None, output: Path, *, dataset: Path | None = None
) -> dict[str, Any]:
    """The frozen-model audit, which connects once its inputs passed their checks.

    The connection has the checkpoint's retry budgets, and its source must still be the
    frozen one the dataset was prepared from. Truth is a supplied parquet, else the
    graph's oracle truth.
    """
    saved, dataset, manifest = audit_inputs(checkpoint, output, dataset)
    executor = connect(saved.validated_config())
    verify_frozen_source(executor, manifest)
    reader: EvaluationTruthSource
    if truth is not None:
        reader = ParquetEvaluationTruth(truth)
    else:
        reader = GraphEvaluationTruth(executor)
    return evaluate_final_population(saved, reader, output, executor=executor, dataset=dataset)
