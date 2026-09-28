"""The ground-truth audits of a run, which `mule evaluate` writes into the run's audit/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import read_json
from ..data.contexts import close_source
from ..evaluation.audit import AUDIT_SPLITS, audit, audit_inputs
from ..evaluation.truth import TruthReader
from ..paths import DATA_DIR, RunPaths
from ..reporting.report import write_audit_report
from ..runtime.progress import recording
from ..tigergraph.oracle import TigerGraphTruth
from ..tigergraph.provenance import verify_frozen_source
from ..tigergraph.scope import TigerGraphScope
from .connect import connect, context_source


def evaluate_run(
    run: RunPaths, *, truth: TruthReader | None = None, data: Path = DATA_DIR
) -> dict[str, Any]:
    """The audits of a run's frozen model on validation and test, by split.

    Decisions use the validation audit; the test audit is for reporting. A split the
    run has already audited is reported from its audit/<split>.json and left as it was.
    The others are audited on one connection, opened once the model and its dataset
    (the model's own in data) passed their checks. The connection has the model's retry
    budgets, and its source must still be the frozen one the dataset was prepared from.
    Truth is read once for both splits: the graph's oracle truth unless ``truth``
    supplies another reader (the tests' ParquetTruth). The lines the audits print are
    appended to the run's events.jsonl. Once a split is audited here, the audit figures
    and report.md are drawn again (reporting.report.write_audit_report).
    """
    inputs = audit_inputs(run, data=data)
    reports = {
        split: read_json(run.audit_report(split)) for split in AUDIT_SPLITS if inputs.audited(split)
    }
    pending = [split for split in AUDIT_SPLITS if split not in reports]
    if pending:
        with recording(run.events):
            executor = connect(inputs.model.config.transport)
            verify_frozen_source(executor, inputs.manifest)
            answer = (truth if truth is not None else TigerGraphTruth(executor)).read()
            scope = TigerGraphScope(executor)
            contexts = context_source(executor, inputs.model.config)
            failed = True
            try:
                for split in pending:
                    reports[split] = audit(
                        inputs, split, truth=answer, scope=scope, contexts=contexts
                    )
                failed = False
            finally:
                close_source(contexts, failed=failed)
        # After the audits' files: a figure that fails loses no audit.
        write_audit_report(run)
    return {split: reports[split] for split in AUDIT_SPLITS}
