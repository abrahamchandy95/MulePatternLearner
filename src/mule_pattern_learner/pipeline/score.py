"""Scoring the accounts listed in a file with a run's model, which `mule score` runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..contract.clock import timestamp
from ..inference.saved_model import SavedModel
from ..inference.score_accounts import check_new_outputs, read_account_ids, score_new_accounts
from ..paths import RunPaths
from ..runtime.progress import recording
from ..tigergraph.cutoffs import TigerGraphCutoffReader
from ..tigergraph.hubs import TigerGraphHubReader
from ..tigergraph.installer import verify_sources
from .connect import connect, context_source


def score_accounts(run: RunPaths, accounts: Path, date: str | None = None) -> dict[str, Any]:
    """Score the accounts of a file (one id per line) with the run's model at a date.

    The date defaults to the model's test cutoff, its last test date. The scores go to
    the run's scores/<file stem>_<date>.parquet and the ids TigerGraph rejects to
    scores/<file stem>_<date>_rejected.txt, and the lines scoring prints are appended to
    the run's events.jsonl. Existing outputs are refused before connecting; the
    connection has the model's retry budgets, and its installed queries must be the
    repository's before any account is scored. The graph need not be the frozen source
    of a dataset, so the source has no disk tier.
    """
    if not accounts.is_file():
        raise FileNotFoundError(f"No account file {accounts}")
    saved = SavedModel.load(run.model)
    if date is None:
        date = max(saved.config.dataset.dates.test, key=timestamp)
    try:
        datetime.fromisoformat(date)
    except ValueError:
        raise ValueError(f"DATE must be an ISO date, got {date!r}") from None
    output = run.scores(accounts.stem, date)
    rejected_output = run.scores_rejected(accounts.stem, date)
    check_new_outputs(output, rejected_output)
    with recording(run.events):
        executor = connect(saved.config.transport)
        verify_sources(executor)
        return score_new_accounts(
            saved,
            read_account_ids(accounts),
            date,
            output,
            rejected_output=rejected_output,
            contexts=context_source(executor, saved.config),
            cutoffs=TigerGraphCutoffReader(executor),
            hub_reader=TigerGraphHubReader(executor),
        )
