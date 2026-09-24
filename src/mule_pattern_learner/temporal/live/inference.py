"""Score an experiment cohort with the same predictor used for new accounts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..common import digest, timestamp
from .dataset import load_prepared, sample_keys
from .hubs import HubRegistry, load_hub_registry
from .predictor import TemporalPredictor, close_source, rejected_path, rejection_summary
from .source import ContextSource, open_context_source


def score(
    checkpoint: Path,
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
    reported separately (see ``predictor.rejection_summary``). ``contexts``/``hubs``
    replace the dataset's live source and hub registry (tests, offline replays).
    """
    rejected_output = rejected_path(output)
    for path in (output, rejected_output):
        if path.exists():
            raise FileExistsError(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest, accounts = load_prepared(dataset)
    if payload["dataset_manifest_sha256"] != digest(dataset / "manifest.json"):
        raise ValueError("Checkpoint belongs to a different prepared dataset")
    if date not in payload["config"]["dates"].get(split, []):
        raise ValueError("Requested split/cutoff was not prepared")
    accounts = accounts[
        (accounts["split"] == split) & (accounts["first_seen_ts_ms"] < timestamp(date))
    ]
    if accounts.empty:
        raise ValueError("No eligible accounts at this cutoff")
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    store = (
        contexts
        if contexts is not None
        else open_context_source(dataset, manifest, payload["config"])
    )
    frames: list[pd.DataFrame] = []
    rejected: list[str] = []
    failed = True
    try:
        predictor = TemporalPredictor(checkpoint, store, hubs=registry)
        size = predictor.batch_size
        batches = (
            sample_keys(accounts.iloc[start : start + size], date, manifest)
            for start in range(0, len(accounts), size)
        )
        for frame, bad in predictor.stream(batches):
            frames.append(frame)
            rejected.extend(key.node_id for key in bad)
        failed = False
    finally:
        close_source(store, failed=failed)
    result = pd.concat(frames, ignore_index=True)
    result["date"] = date
    result["cutoff_utc"] = date
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    if rejected:
        rejected_output.write_text("".join(value + "\n" for value in rejected))
    return {
        "accounts": len(result),
        **rejection_summary(store, len(rejected), predictor.totals),
        "rejected_output": str(rejected_output) if rejected else None,
        "device": str(predictor.device),
        "embedding_dimensions": predictor.model.head[0].in_features,
        "output": str(output),
        "cohort": manifest["cohort"],
    }
