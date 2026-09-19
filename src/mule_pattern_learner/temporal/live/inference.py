"""Score an experiment cohort with the same predictor used for new accounts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..common import digest, timestamp
from .dataset import load_prepared, sample_keys
from .predictor import TemporalPredictor
from .source import open_context_source


def score(checkpoint: Path, dataset: Path, date: str, split: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
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
    store = open_context_source(dataset, manifest)
    predictor = TemporalPredictor(checkpoint, store)
    frames = []
    try:
        for start in range(0, len(accounts), predictor.batch_size):
            selected = accounts.iloc[start : start + predictor.batch_size]
            frames.append(predictor.predict(sample_keys(selected, date, manifest)))
    finally:
        store.close()
    result = pd.concat(frames, ignore_index=True)
    result["date"] = date
    result["cutoff_utc"] = date
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return {
        "accounts": len(result),
        "device": str(predictor.device),
        "embedding_dimensions": predictor.config.get("hidden", 64),
        "output": str(output),
        "cohort": manifest["cohort"],
    }
