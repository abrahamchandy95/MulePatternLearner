"""Scoring arbitrary accounts at a date on the graph, with its installed queries checked."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..inference.saved_model import ModelCheckpoint
from ..inference.score_accounts import check_new_outputs, score_new_accounts
from ..tigergraph.installer import verify_sources
from .connect import connect


def score_new(
    checkpoint: Path, account_ids: Iterable[str], date: str, output: Path
) -> dict[str, Any]:
    """Score the accounts at a date on a connection with the checkpoint's retry budgets.

    Existing outputs are refused before connecting, and the installed queries must be
    the repository's before any account is scored.
    """
    check_new_outputs(output)
    saved = ModelCheckpoint.of(checkpoint)
    executor = connect(saved.validated_config())
    verify_sources(executor)
    return score_new_accounts(saved, account_ids, date, output, executor=executor)
