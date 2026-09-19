"""Inductive prediction from bounded contexts, independent of training account IDs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mule_pattern_learner.device import choose_device
from ..common import timestamp
from ..encoding import BASIS_ID
from .batching import make_live_batch
from .contract import ContextKey, contract_fingerprint
from .memory import BatchLimits
from .model import LiveTGAT
from .source import (
    ContextSource,
    QueryExecutor,
    TigerGraphExecutor,
    StreamingContextSource,
    checked_rows,
)


class TemporalPredictor:
    """The same feature/weight contract for old and newly arriving accounts."""

    def __init__(
        self,
        checkpoint: Path,
        contexts: ContextSource | None = None,
        device: str = "auto",
        *,
        executor: QueryExecutor | None = None,
    ) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["contract"] != contract_fingerprint() or payload["basis_id"] != BASIS_ID:
            raise ValueError("Checkpoint feature/time contract differs from this sampler")
        self.config = payload["config"]
        self.threshold = float(payload["threshold"])
        if contexts is None:
            if executor is None:
                raise ValueError("Provide a context source or query executor")
            contexts = StreamingContextSource(
                executor, per_relation=int(self.config.get("per_relation", 2))
            )
        self.contexts = contexts
        self.device = choose_device(device)
        self.fanouts = tuple(self.config.get("fanouts", [8, 4]))
        self.batch_size = min(int(self.config.get("batch_size", 64)), 128)
        BatchLimits().validate_model(
            self.batch_size, self.fanouts, int(self.config.get("hidden", 64))
        )
        self.model = LiveTGAT(
            self.config.get("hidden", 64),
            self.config.get("heads", 4),
            self.config.get("dropout", 0.15),
            self.config.get("variant", "temporal"),
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        torch.set_num_threads(int(self.config.get("threads", 4)))

    def predict(self, keys: list[ContextKey]) -> pd.DataFrame:
        """Retain tensors for one bounded batch, return only CPU predictions."""
        BatchLimits().validate_model(len(keys), self.fanouts, int(self.config.get("hidden", 64)))
        with torch.inference_mode():
            batch = make_live_batch(self.contexts, keys, fanouts=self.fanouts, device=self.device)
            hidden = self.model.encode(batch)
            probabilities = torch.sigmoid(self.model.head(hidden).squeeze(-1)).cpu().numpy()
            vectors = hidden.cpu().tolist()
        return pd.DataFrame(
            {
                "account_id": [key.node_id for key in keys],
                "score": probabilities,
                "embedding": vectors,
                "predicted_mule": probabilities >= self.threshold,
            }
        )


def id_batches(ids: Iterable[str], size: int) -> Iterator[list[str]]:
    """Consume input IDs lazily; never allocate a database-wide ID map."""
    iterator = iter(ids)
    while batch := list(islice(iterator, size)):
        yield batch


def read_account_ids(path: Path) -> Iterator[str]:
    with path.open() as stream:
        for line in stream:
            value = line.strip()
            if value:
                yield value


def score_new_accounts(
    checkpoint: Path,
    account_ids: Iterable[str],
    date: str,
    output: Path,
    *,
    executor: QueryExecutor | None = None,
) -> dict[str, Any]:
    """Score arbitrary existing-in-TigerGraph account IDs without a training manifest.

    Inference can use all history available at its cutoff. Strict experiment
    scoring uses scoped ContextKeys through TemporalPredictor instead.
    """
    if output.exists():
        raise FileExistsError(output)
    if executor is None:
        from .installation import verify_sources

        live = TigerGraphExecutor()
        verify_sources(live)
        executor = live
    ms = timestamp(date) - 1
    result = checked_rows(executor.run("temporal_training_cutoffs", {"cutoff_times": [ms]}))
    clocks = next(row["last_visible_seqs"] for row in result if "last_visible_seqs" in row)
    seq = int(clocks[str(ms)]) + 1
    predictor = TemporalPredictor(checkpoint, executor=executor)
    source = predictor.contexts
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(pending)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    try:
        for ids in id_batches(account_ids, predictor.batch_size):
            keys = [ContextKey("Account", value, seq, ms) for value in ids]
            frame = predictor.predict(keys)
            frame["date"] = date
            frame["cutoff_utc"] = date
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(pending, table.schema)
            writer.write_table(table)
            count += len(frame)
        if writer is None:
            raise ValueError("No account IDs supplied")
        writer.close()
        writer = None
        pending.replace(output)
    finally:
        if writer is not None:
            writer.close()
        if pending.exists():
            pending.unlink()
        source.close()
    return {
        "accounts": count,
        "output": str(output),
        "device": str(predictor.device),
        "database_calls": source.query_calls,
        "scope": "available_history_at_prediction",
    }
