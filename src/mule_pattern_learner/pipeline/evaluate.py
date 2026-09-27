"""The ground-truth audit of a run, which `mule evaluate` writes into the run's audit/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evaluation.audit import audit, audit_inputs
from ..evaluation.truth import TruthReader
from ..paths import DATA_DIR, RunPaths
from ..tigergraph.context_query import TigerGraphContextFetcher
from ..tigergraph.oracle import TigerGraphTruth
from ..tigergraph.provenance import verify_frozen_source
from ..tigergraph.scope import TigerGraphScope
from .connect import connect


def evaluate_run(
    run: RunPaths, *, truth: TruthReader | None = None, data: Path = DATA_DIR
) -> dict[str, Any]:
    """The audit of a run's frozen model, which connects once its inputs passed their checks.

    The dataset is the model's own in data. The connection has the model's retry
    budgets, and its source must still be the frozen one the dataset was prepared from.
    Truth is the graph's oracle truth unless ``truth`` supplies another reader (the
    tests' ParquetTruth). The audit goes into the run's audit/ files.
    """
    saved, dataset, manifest = audit_inputs(run, None, data)
    executor = connect(saved.config.transport)
    verify_frozen_source(executor, manifest)
    return audit(
        run,
        truth if truth is not None else TigerGraphTruth(executor),
        scope=TigerGraphScope(executor),
        fetcher=TigerGraphContextFetcher(executor),
        dataset=dataset,
    )
