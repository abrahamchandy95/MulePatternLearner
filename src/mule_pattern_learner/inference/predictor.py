"""Inductive prediction from bounded contexts, independent of training account IDs.

score_batches is the one scoring loop: training scores validation and test with it,
and Predictor scores prepared splits, audit samples and arbitrary accounts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator
import contextlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from ..batching.assemble import (
    RootBatch,
    batch_counts,
    batch_device,
    build_root_batch,
    to_device,
)
from ..batching.limits import BatchLimits
from ..contract.bounds import BATCH_ROOTS
from ..contract.graph_schema import ContextKey
from ..data.contexts import ContextReader, check_coverage
from ..data.hub_registry import HubRegistry, warn_hub_stubs
from ..model.build import Model, build_model, probabilities_from_logits
from ..runtime.console import plural, show_scoring
from ..runtime.device import choose_device, torch_runtime
from ..runtime.workers import BatchPrefetcher
from .saved_model import SavedModel


class ScoredBatch(NamedTuple):
    """One assembled batch and the logits (and embeddings) of its accepted roots.

    Both are None when TigerGraph rejected every root; embeddings are also None unless
    they were asked for.
    """

    prepared: RootBatch
    logits: torch.Tensor | None
    embeddings: torch.Tensor | None = None


def score_batch(
    model: Model, prepared: RootBatch, device: torch.device, *, embeddings: bool = False
) -> ScoredBatch:
    """The logits of one assembled batch, on ``device``; the model must be in eval mode.

    With ``embeddings`` the model's logits come from the encoder's output, which is kept
    too.
    """
    if prepared.batch is None:
        return ScoredBatch(prepared, None)
    with torch.inference_mode():
        batch = to_device(prepared.batch, device)
        if not embeddings:
            return ScoredBatch(prepared, model(batch))
        hidden = model.encode(batch)
        return ScoredBatch(prepared, model.logits(batch, hidden), hidden)


def score_batches(
    model: Model,
    build: Callable[[list[ContextKey]], RootBatch],
    batches: Iterable[list[ContextKey]],
    *,
    device: torch.device,
    prefetch: int,
    embeddings: bool = False,
) -> Generator[ScoredBatch]:
    """Score key batches in order, building the next ``prefetch`` ones on worker threads.

    ``build`` fetches and assembles one batch; it runs on the workers. Close the
    iterator (``contextlib.closing``) when its consumer fails, so the workers stop
    without waiting for the builds in flight.
    """
    with BatchPrefetcher(build, batches, depth=prefetch) as prepared:
        for item in prepared:
            yield score_batch(model, item, device, embeddings=embeddings)


def accepted_scores(
    logits: list[torch.Tensor], accepted: list[np.ndarray], label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities of every requested root, in order, and the accepted mask.

    Probabilities are float64 (see ``probabilities_from_logits``), so high scores do
    not tie. Roots that TigerGraph rejects are False in the mask (their score is NaN).
    A non-finite probability for an accepted root is an error, never a rejection;
    ``label`` names the scored roots in its message.
    """
    mask = np.concatenate(accepted) if accepted else np.zeros(0, dtype=bool)
    scores = np.full(len(mask), np.nan)
    if logits:
        # One device-to-host copy per call.
        probabilities = probabilities_from_logits(torch.cat(logits))
        if not np.isfinite(probabilities).all():
            roots = plural(int((~np.isfinite(probabilities)).sum()), f"accepted {label} root")
            raise ValueError(f"Non-finite model probability for {roots}")
        scores[mask] = probabilities
    return scores, mask


class Predictor:
    """The same feature/weight contract for old and newly arriving accounts.

    Scoring uses the deterministic evaluation sampler. ``contexts`` must request the
    saved model's inputs with its candidate pools (the pipeline opens it for the model's
    configuration). ``hubs`` must be the registry for the scored cutoff (training's
    dataset registry or ``query_hubs``); without it no child is stubbed and hub children
    are masked out when TigerGraph rejects them. The caller closes ``contexts``. The
    device is the saved configuration's ``runtime.device`` unless ``device`` names one,
    and scoring runs inside ``runtime()``, the saved determinism and CPU threads.
    """

    def __init__(
        self,
        model: Path | SavedModel,
        contexts: ContextReader,
        device: str | None = None,
        *,
        hubs: HubRegistry | None = None,
    ) -> None:
        saved = SavedModel.of(model)
        saved.check_contract()
        self.config = config = saved.config
        self.plan = config.feature_plan()
        self.sampler = config.sampler
        saved.check_inputs(self.plan)
        self.threshold = saved.threshold
        check_coverage(contexts, self.plan, self.sampler)
        self.contexts = contexts
        self.hubs = hubs if hubs is not None else HubRegistry.empty()
        warn_hub_stubs(self.hubs, self.plan)
        # Batch statistics of everything streamed (stub and rejected children).
        self.totals: Counter[str] = Counter()
        self.device = choose_device(config.runtime.device if device is None else device)
        self.batch_device = batch_device(self.device)
        # The fan-outs training sampled, so scoring samples training's neighbourhoods.
        self.fanouts = self.sampler.fanouts
        self.hidden = config.model.hidden
        self.batch_size = min(config.training.batch_size, BATCH_ROOTS.high)
        BatchLimits().validate_model(
            self.batch_size, self.fanouts, self.hidden, self.plan, self.sampler
        )
        self.prefetch = config.runtime.prefetch_batches
        self.model = build_model(config.model, self.plan, self.fanouts[0]).to(self.device)
        self.model.load_state_dict(saved.state_dict)
        self.model.eval()

    def runtime(self) -> contextlib.AbstractContextManager[None]:
        """The saved determinism and CPU threads for the block that scores, then restored."""
        runtime = self.config.runtime
        return torch_runtime(
            self.device, deterministic=runtime.deterministic, threads=runtime.threads
        )

    def prepare(self, keys: list[ContextKey]) -> RootBatch:
        """Fetch and assemble one batch; safe to call from prefetch worker threads."""
        BatchLimits().validate_model(len(keys), self.fanouts, self.hidden, self.plan, self.sampler)
        return build_root_batch(
            self.contexts,
            keys,
            device=self.batch_device,
            plan=self.plan,
            sampler=self.sampler,
            hubs=self.hubs,
            mode="eval",
        )

    def frame(self, scored: ScoredBatch) -> pd.DataFrame:
        """The rows of a scored batch's accepted roots."""
        if scored.logits is None or scored.embeddings is None:
            return pd.DataFrame(
                {
                    "account_id": pd.Series([], dtype=object),
                    "score": np.zeros(0, dtype=np.float64),
                    "embedding": pd.Series([], dtype=object),
                    "predicted_mule": np.zeros(0, dtype=bool),
                }
            )
        probabilities = probabilities_from_logits(scored.logits)
        return pd.DataFrame(
            {
                "account_id": [key.node_id for key in scored.prepared.keys],
                "score": probabilities,
                "embedding": scored.embeddings.cpu().tolist(),
                "predicted_mule": probabilities >= self.threshold,
            }
        )

    def stream(
        self,
        batches: Iterable[list[ContextKey]],
        *,
        shown: str | None = None,
        total: int | None = None,
    ) -> Iterator[tuple[pd.DataFrame, list[ContextKey]]]:
        """Score key batches in order, prefetching the next ones on worker threads.

        Integer batch statistics are summed into ``self.totals``. With ``shown``, a
        terminal shows in place how many of the ``total`` roots are scored so far
        ("scoring <shown> 640/2,011", runtime.console.show_scoring).
        """
        scored = score_batches(
            self.model,
            self.prepare,
            batches,
            device=self.device,
            prefetch=self.prefetch,
            embeddings=True,
        )
        done = 0
        with contextlib.closing(scored):
            for item in scored:
                self.totals.update(batch_counts(item.prepared.stats))
                if shown is not None:
                    done += len(item.prepared.requested)
                    show_scoring(shown, done, total)
                yield self.frame(item), item.prepared.rejected

    def score_keys(
        self,
        batches: Iterable[list[ContextKey]],
        *,
        shown: str | None = None,
        total: int | None = None,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """Frames of the accepted roots and the node IDs of the rejected ones, in order.

        ``shown`` and ``total`` give the progress a terminal shows, as in stream.
        """
        frames: list[pd.DataFrame] = []
        rejected: list[str] = []
        for frame, bad in self.stream(batches, shown=shown, total=total):
            frames.append(frame)
            rejected.extend(key.node_id for key in bad)
        return frames, rejected
