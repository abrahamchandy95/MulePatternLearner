"""The ground-truth audits of a run, which `mule evaluate` writes into the run's audit/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import read_audit_report
from ..data.context_cache import ContextCache
from ..data.contexts import close_source
from ..evaluation.audit import AUDIT_SPLITS, audit, audit_inputs
from ..evaluation.truth import TruthReader
from ..paths import DATA_DIR, RunPaths
from ..reporting.run_report import write_audit_report
from ..runtime.progress import emit, recording
from ..tigergraph.oracle import TigerGraphTruthReader
from ..tigergraph.provenance import verify_frozen_source
from ..tigergraph.scope import TigerGraphScopeReader
from .connect import Session, connect, context_source


class SharedTruth:
    """The graph's oracle truth on a session's connection, read once for all its audits.

    The first audit that needs truth reads it (connecting the session if nothing has
    yet), and every later audit gets the same table, so a suite of runs reads truth
    once, and not at all when every run it names is audited already.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.table: pd.DataFrame | None = None

    def read(self) -> pd.DataFrame:
        if self.table is None:
            self.table = TigerGraphTruthReader(self.session.executor()).read()
        return self.table


def evaluate_run(
    run: RunPaths,
    *,
    truth: TruthReader | None = None,
    data: Path = DATA_DIR,
    session: Session | None = None,
) -> dict[str, Any]:
    """The audits of a run's frozen model on validation and test, by split.

    Decisions use the validation audit; the test audit is for reporting. A split the
    run has already audited is reported from its audit/<split>.json and left as it was,
    and an `already_audited` event names the reports read.
    The others are audited on one connection, opened once the model and its dataset
    (the model's own in data) passed their checks. The connection has the model's retry
    budgets, and its source must still be the frozen one the dataset was prepared from;
    the audits then read and write the dataset's disk tier, which every run of the
    dataset shares, so the audits of the next run request none of the same contexts.
    Truth is read once for both splits: the graph's oracle truth unless ``truth``
    supplies another reader (the tests' ParquetTruthReader, or a suite's SharedTruth).
    With a ``session`` the audits run on its connection, which a suite of runs shares.
    The audits' events are recorded in the run's events.jsonl. Once a split is
    audited here, the audit figures and report.md are drawn again
    (reporting.run_report.write_audit_report).
    """
    inputs = audit_inputs(run, data=data)
    reports = {
        split: read_audit_report(run.audit_report(split))
        for split in AUDIT_SPLITS
        if inputs.audited(split)
    }
    if reports:
        audited = [str(run.audit_report(split)) for split in reports]
        emit({"event": "already_audited", "run": str(run.root), "reports": audited})
    pending = [split for split in AUDIT_SPLITS if split not in reports]
    if pending:
        with recording(run.events):
            transport = inputs.model.config.transport
            executor = session.executor() if session is not None else connect(transport)
            verify_frozen_source(executor, inputs.manifest)
            answer = (truth if truth is not None else TigerGraphTruthReader(executor)).read()
            scope = TigerGraphScopeReader(executor)
            cache = ContextCache.of(inputs.dataset, inputs.manifest)
            contexts = context_source(executor, inputs.model.config, cache)
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
