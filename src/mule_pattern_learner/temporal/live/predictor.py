"""Inductive prediction from bounded contexts, independent of training account IDs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mule_pattern_learner.device import choose_device

from .batching import RootBatch, build_root_batch
from .checkpoint import ModelCheckpoint
from .config_schema import fanouts, setting
from .contract import ContextKey, FeaturePlan, SamplerPlan
from .dataset import resolve_cutoff
from .executor import QueryExecutor, live_executor
from .hubs import HubRegistry, hub_threshold, query_hub_registry, warn_hub_stubs
from .memory import BatchLimits
from .model import build_model
from .sampling import BatchPrefetcher
from .source import (
    ContextSource,
    check_coverage,
    close_source,
    rejection_summary,
    streaming_source,
)

SCORE_SCHEMA = pa.schema(
    [
        ("account_id", pa.string()),
        ("score", pa.float32()),
        ("embedding", pa.list_(pa.float64())),
        ("predicted_mule", pa.bool_()),
        ("date", pa.string()),
        ("cutoff_utc", pa.string()),
    ]
)


def query_hubs(
    executor: QueryExecutor, cutoff_seqs: list[int], sampler: SamplerPlan
) -> HubRegistry:
    """Unscoped hub registry for arbitrary cutoffs, with the checkpoint's threshold.

    Score-new runs unscoped, so its rows carry visibility phase 3.
    """
    return query_hub_registry(executor, cutoff_seqs, threshold=hub_threshold(sampler))


class TemporalPredictor:
    """The same feature/weight contract for old and newly arriving accounts.

    Scoring uses the deterministic evaluation sampler. ``hubs`` must be the registry
    for the scored cutoff (training's dataset registry or ``query_hubs``); without
    it no child is stubbed and hub children are masked out when TigerGraph rejects them.
    """

    def __init__(
        self,
        checkpoint: Path | ModelCheckpoint,
        contexts: ContextSource | None = None,
        device: str = "auto",
        *,
        executor: QueryExecutor | None = None,
        hubs: HubRegistry | None = None,
    ) -> None:
        saved = ModelCheckpoint.of(checkpoint)
        saved.check_contract()
        self.config: dict[str, Any] = saved.validated_config()
        self.plan = FeaturePlan.from_config(self.config)
        self.sampler = SamplerPlan.from_config(self.config)
        saved.check_inputs(self.plan)
        self.threshold = saved.threshold
        created = contexts is None
        if contexts is None:
            if executor is None:
                raise ValueError("Provide a context source or query executor")
            contexts = streaming_source(executor, self.plan, self.sampler, self.config)
        try:
            check_coverage(contexts, self.plan, self.sampler)
            self.contexts = contexts
            self.hubs = hubs if hubs is not None else HubRegistry.empty()
            warn_hub_stubs(self.hubs, self.plan)
            # Batch statistics of everything streamed (stub and rejected children).
            self.totals: Counter[str] = Counter()
            self.device = choose_device(device)
            # CUDA batches are assembled on the device by the prefetch workers.
            self.batch_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            # Read like RunSettings reads them, so scoring samples training's neighbourhoods.
            self.fanouts = fanouts(self.config)
            self.hidden = int(setting(self.config, "hidden"))
            self.batch_size = min(int(setting(self.config, "batch_size")), 128)
            BatchLimits().validate_model(
                self.batch_size, self.fanouts, self.hidden, self.plan, self.sampler
            )
            self.prefetch = int(setting(self.config, "prefetch_batches"))
            self.model = build_model(self.config, self.plan).to(self.device)
            self.model.load_state_dict(saved.state_dict)
            self.model.eval()
            torch.set_num_threads(int(setting(self.config, "threads")))
        except BaseException:
            if created:
                contexts.close()
            raise

    def prepare(self, keys: list[ContextKey]) -> RootBatch:
        """Fetch and assemble one batch; safe to call from prefetch worker threads."""
        BatchLimits().validate_model(len(keys), self.fanouts, self.hidden, self.plan, self.sampler)
        return build_root_batch(
            self.contexts,
            keys,
            fanouts=self.fanouts,
            device=self.batch_device,
            plan=self.plan,
            sampler=self.sampler,
            hubs=self.hubs,
            mode="eval",
        )

    def infer(self, prepared: RootBatch) -> pd.DataFrame:
        """Scores and embeddings for the accepted roots only; CPU outputs."""
        keys = prepared.keys
        if prepared.batch is None:
            return pd.DataFrame(
                {
                    "account_id": pd.Series([], dtype=object),
                    "score": np.zeros(0, dtype=np.float32),
                    "embedding": pd.Series([], dtype=object),
                    "predicted_mule": np.zeros(0, dtype=bool),
                }
            )
        with torch.inference_mode():
            batch = {k: v.to(self.device) for k, v in prepared.batch.items()}
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

    def predict(self, keys: list[ContextKey]) -> pd.DataFrame:
        """Rows for accepted keys only; rejected keys are counted on the source."""
        return self.infer(self.prepare(keys))

    def stream(
        self, batches: Iterable[list[ContextKey]]
    ) -> Iterator[tuple[pd.DataFrame, list[ContextKey]]]:
        """Score key batches in order, prefetching the next ones on worker threads.

        Integer batch statistics are summed into ``self.totals``.
        """
        with BatchPrefetcher(self.prepare, batches, depth=self.prefetch) as prepared:
            for item in prepared:
                self.totals.update(
                    {
                        k: int(v)
                        for k, v in item.stats.items()
                        if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                    }
                )
                yield self.infer(item), item.rejected

    def score_keys(
        self, batches: Iterable[list[ContextKey]]
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """Frames of the accepted roots and the node IDs of the rejected ones, in order."""
        frames: list[pd.DataFrame] = []
        rejected: list[str] = []
        for frame, bad in self.stream(batches):
            frames.append(frame)
            rejected.extend(key.node_id for key in bad)
        return frames, rejected


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


def rejected_path(output: Path) -> Path:
    return output.with_name(output.name + ".rejected.txt")


def write_rejected(path: Path, ids: list[str]) -> None:
    """One rejected root ID per line."""
    path.write_text("".join(value + "\n" for value in ids))


def score_new_accounts(
    checkpoint: Path | ModelCheckpoint,
    account_ids: Iterable[str],
    date: str,
    output: Path,
    *,
    executor: QueryExecutor | None = None,
    contexts: ContextSource | None = None,
    hubs: HubRegistry | None = None,
) -> dict[str, Any]:
    """Score arbitrary existing-in-TigerGraph account IDs without a training manifest.

    Inference can use all history available at its cutoff. Strict experiment
    scoring uses scoped ContextKeys through TemporalPredictor instead. IDs that
    TigerGraph rejects (missing, not yet visible, over capacity) are not scored:
    they are listed in ``<output>.rejected.txt``. The result reports rejected roots and
    masked child contexts separately (see ``rejection_summary``). A date before the
    first visible event is refused, since no account could be scored at it.
    """
    rejected_output = rejected_path(output)
    pending = output.with_name(output.name + ".pending")
    rejected_pending = rejected_output.with_name(rejected_output.name + ".pending")
    for path in (output, rejected_output, pending, rejected_pending):
        if path.exists():
            raise FileExistsError(path)
    saved = ModelCheckpoint.of(checkpoint)
    if executor is None:
        from .installation import verify_sources

        # The checkpoint's retry budgets (max_query_attempts, max_outage_s).
        live = live_executor(saved.validated_config())
        verify_sources(live)
        executor = live
    seq, ms = resolve_cutoff(executor, date)
    predictor = TemporalPredictor(saved, contexts, executor=executor, hubs=hubs)
    source = predictor.contexts
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    count = rejected = supplied = 0
    examples: list[str] = []
    failed = True
    try:
        if hubs is None:
            predictor.hubs = query_hubs(executor, [seq], predictor.sampler)
            warn_hub_stubs(predictor.hubs, predictor.plan)
        writer = pq.ParquetWriter(pending, SCORE_SCHEMA)
        with rejected_pending.open("w") as rejected_stream:
            batches = (
                [ContextKey("Account", value, seq, ms) for value in ids]
                for ids in id_batches(account_ids, predictor.batch_size)
            )
            for frame, bad in predictor.stream(batches):
                supplied += len(frame) + len(bad)
                for key in bad:
                    rejected_stream.write(key.node_id + "\n")
                    if len(examples) < 20:
                        examples.append(key.node_id)
                rejected += len(bad)
                if len(frame):
                    frame["date"] = date
                    frame["cutoff_utc"] = date
                    writer.write_table(
                        pa.Table.from_pandas(frame, schema=SCORE_SCHEMA, preserve_index=False)
                    )
                    count += len(frame)
        if not supplied:
            raise ValueError("No account IDs supplied")
        writer.close()
        writer = None
        pending.replace(output)
        if rejected:
            rejected_pending.replace(rejected_output)
        failed = False
    finally:
        if writer is not None:
            writer.close()
        for path in (pending, rejected_pending):
            if path.exists():
                path.unlink()
        close_source(source, failed=failed)
    return {
        "accounts": count,
        **rejection_summary(source, rejected, predictor.totals),
        "rejected_examples": examples,
        "rejected_output": str(rejected_output) if rejected else None,
        "output": str(output),
        "device": str(predictor.device),
        "database_calls": source.query_calls,
        "scope": "available_history_at_prediction",
    }
