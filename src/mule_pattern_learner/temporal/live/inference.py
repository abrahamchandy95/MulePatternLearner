"""Score an experiment cohort with the same predictor used for new accounts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .checkpoint import ModelCheckpoint
from .dataset import eligible_mask, load_prepared, sample_keys
from .hubs import HubRegistry, load_hub_registry
from .predictor import TemporalPredictor, rejected_path, write_rejected
from .source import ContextSource, close_source, open_context_source, rejection_summary


def score(
    checkpoint: Path | ModelCheckpoint,
    dataset: Path,
    date: str,
    split: str,
    output: Path,
    *,
    contexts: ContextSource | None = None,
    hubs: HubRegistry | None = None,
) -> dict[str, Any]:
    """Score every eligible account of one prepared split and cutoff.

    Roots that TigerGraph rejects are not scored; their IDs go to
    ``<output>.rejected.txt``. Rejected roots and masked child contexts are
    reported separately (see ``source.rejection_summary``). ``contexts``/``hubs``
    replace the dataset's live source and hub registry (tests, offline replays).
    """
    rejected_output = rejected_path(output)
    for path in (output, rejected_output):
        if path.exists():
            raise FileExistsError(path)
    saved = ModelCheckpoint.of(checkpoint)
    manifest, accounts = load_prepared(dataset)
    saved.check_dataset(dataset)
    if date not in saved.config["dates"].get(split, []):
        raise ValueError("Requested split/cutoff was not prepared")
    accounts = accounts[eligible_mask(accounts, split, date)]
    if accounts.empty:
        raise ValueError("No eligible accounts at this cutoff")
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    store = (
        contexts if contexts is not None else open_context_source(dataset, manifest, saved.config)
    )
    failed = True
    try:
        predictor = TemporalPredictor(saved, store, hubs=registry)
        size = predictor.batch_size
        frames, rejected = predictor.score_keys(
            sample_keys(accounts.iloc[start : start + size], date, manifest)
            for start in range(0, len(accounts), size)
        )
        failed = False
    finally:
        close_source(store, failed=failed)
    result = pd.concat(frames, ignore_index=True)
    result["date"] = date
    result["cutoff_utc"] = date
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    if rejected:
        write_rejected(rejected_output, rejected)
    return {
        "accounts": len(result),
        **rejection_summary(store, len(rejected), predictor.totals),
        "rejected_output": str(rejected_output) if rejected else None,
        "device": str(predictor.device),
        "embedding_dimensions": predictor.model.head[0].in_features,
        "output": str(output),
        "cohort": manifest["cohort"],
    }
