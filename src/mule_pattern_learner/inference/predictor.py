"""Inductive prediction from bounded contexts, independent of training account IDs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..batching.assemble import RootBatch, build_root_batch
from ..batching.limits import BatchLimits
from ..config import fanouts
from ..contract.feature_groups import FeaturePlan
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..data.contexts import ContextSource, check_coverage, streaming_source
from ..data.hub_registry import HubRegistry, warn_hub_stubs
from ..model.build import build_model, probabilities_from_logits
from ..runtime.device import choose_device
from ..runtime.workers import BatchPrefetcher
from ..tigergraph.executor import QueryExecutor
from .saved_model import ModelCheckpoint


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
            self.hidden = int(self.config["hidden"])
            self.batch_size = min(int(self.config["batch_size"]), 128)
            BatchLimits().validate_model(
                self.batch_size, self.fanouts, self.hidden, self.plan, self.sampler
            )
            self.prefetch = int(self.config["prefetch_batches"])
            self.model = build_model(self.config, self.plan).to(self.device)
            self.model.load_state_dict(saved.state_dict)
            self.model.eval()
            torch.set_num_threads(int(self.config["threads"]))
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
                    "score": np.zeros(0, dtype=np.float64),
                    "embedding": pd.Series([], dtype=object),
                    "predicted_mule": np.zeros(0, dtype=bool),
                }
            )
        with torch.inference_mode():
            batch = {k: v.to(self.device) for k, v in prepared.batch.items()}
            hidden = self.model.encode(batch)
            probabilities = probabilities_from_logits(self.model.head(hidden).squeeze(-1))
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
